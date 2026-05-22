"""
Diffusion models for polyhedron generation.

Contains:
  - PolyDiffusion_new: Original single-polyhedron DDPM (unchanged)
  - MultiPolyDiffusion: NEW multi-polyhedron diffusion with masked losses

### NEW: MultiPolyDiffusion class for joint generation of all polyhedra per crystal
"""

import math, copy

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from typing import Any, Dict

import hydra
import omegaconf
import pytorch_lightning as pl
from torch_scatter import scatter

from tqdm import tqdm

from equicsp.common.data_utils import MAX_SLOTS, ATOMS_PER_SLOT, TOTAL_ATOMS

from equicsp.pl_modules.diff_utils import (
    d_log_p_wrapped_normal,
    PredefinedNoiseSchedule,
    GammaNetwork,
    assert_mean_zero,
)

MAX_ATOMIC_NUM = 100


class BaseModule(pl.LightningModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        # populate self.hparams with args and kwargs automagically!
        self.save_hyperparameters()

    def configure_optimizers(self):
        opt = hydra.utils.instantiate(
            self.hparams.optim.optimizer, params=self.parameters(), _convert_="partial"
        )
        if not self.hparams.optim.use_lr_scheduler:
            return [opt]
        scheduler = hydra.utils.instantiate(
            self.hparams.optim.lr_scheduler, optimizer=opt
        )

        # return {"optimizer": opt, "lr_scheduler": scheduler, "monitor": "val_loss"}

        ### modification
        # actual scheduler working code
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
                "strict": False,
            },
        }

        # not doing scheduling, but working
        # return {
        #     "optimizer": opt,
        #     "lr_scheduler": {
        #         "scheduler": scheduler,
        #         "monitor": "val_checkpoint_on",  # Changed to val_checkpoint_on
        #         "interval": "epoch",  # Update after each epoch
        #         "frequency": 1,  # Check every epoch
        #         "strict": False,  # Allow if metric is missing initially
        #     },
        #     "monitor": "val_loss",
        # }
        ### modification


def calc_mean_sin_cos(data_tensor):
    m_sin = torch.sin(data_tensor).mean(dim=1)  # Mean of sine values along dim=1
    m_cos = torch.cos(data_tensor).mean(dim=1)  # Mean of cosine values along dim=1
    return m_sin, m_cos


def calc_grouped_angles_mean_in_radians(data_tensor, groups):
    # reflect [0,1] to angle
    data_tensor = data_tensor * 2 * math.pi

    data_tensor_sin = torch.sin(data_tensor)
    data_tensor_cos = torch.cos(data_tensor)

    sum_sin = scatter(data_tensor_sin, groups, dim=0, reduce="sum")
    sum_cos = scatter(data_tensor_cos, groups, dim=0, reduce="sum")

    group_counts = scatter(torch.ones_like(data_tensor), groups, dim=0, reduce="sum")

    mean_sin = sum_sin / group_counts
    mean_cos = sum_cos / group_counts

    mean_angle = torch.atan2(
        mean_sin, mean_cos
    )  # Calculate mean angle in radians using atan2 for stability

    # Adjust mean_angle to be in the range [0, 2*pi)
    mean_angle = torch.where(mean_angle >= 0, mean_angle, mean_angle + 2 * math.pi)

    mean_angle = mean_angle / (2 * math.pi)

    return mean_angle


#  Probabilistic Modeling Process. See Appendix C.2.
def d_log_x(score, d_mean_x, groups, num_atoms):
    score_sum = scatter(score, groups, dim=0, reduce="sum")
    score_sum = score_sum.repeat_interleave(num_atoms, dim=0)
    score_mean = score_sum * (-d_mean_x)
    result = score_mean + score
    return result


# g() function in Eq(36) of our paper
def d_mean_angle_x(data_tensor, groups, num_atoms):
    data_tensor = data_tensor * 2 * math.pi

    data_tensor_sin = torch.sin(data_tensor)
    data_tensor_cos = torch.cos(data_tensor)

    u = scatter(data_tensor_sin, groups, dim=0, reduce="mean")
    v = scatter(data_tensor_cos, groups, dim=0, reduce="mean")

    n = scatter(torch.ones_like(data_tensor), groups, dim=0, reduce="sum")

    u = u.repeat_interleave(num_atoms, dim=0)
    v = v.repeat_interleave(num_atoms, dim=0)
    n = n.repeat_interleave(num_atoms, dim=0)

    result = (v * data_tensor_cos + u * data_tensor_sin) / (u**2 + v**2) / n

    return result


class SinusoidalTimeEmbeddings(nn.Module):
    """Attention is all you need."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


### modfification
### Sinusoidal Position Embedding for continuous time step
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        # x = x.squeeze() * 1000
        # assert len(x.shape) == 1
        x = x.reshape(-1) * 1000.0
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


### modfification


# ---------------------------------------------------------------------------
# PolyDiffusion_new — ORIGINAL single-polyhedron (UNCHANGED)
# ---------------------------------------------------------------------------
class PolyDiffusion_new(BaseModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.decoder = hydra.utils.instantiate(
            self.hparams.decoder,
            latent_dim=self.hparams.latent_dim + self.hparams.time_dim,
            _recursive_=False,
        )
        self.timesteps = self.hparams.timesteps
        self.time_dim = self.hparams.time_dim
        self.time_embedding_cont = SinusoidalPosEmb(self.time_dim)
        self.keep_coords = self.hparams.cost_coord < 1e-5

        self.noise_schedule = "polynomial_2.0"
        self.noise_precision = 1e-5

        self.gamma = PredefinedNoiseSchedule(
            self.noise_schedule,
            timesteps=self.timesteps,
            precision=self.noise_precision,
        )

        self.time_step_style = "continuous"
        self.coord_norm_factor = 1.0

    def forward(self, batch):
        batch_size = batch.num_graphs

        t_int = torch.randint(
            0, self.timesteps + 1, size=(batch_size, 1), device=self.device
        ).float()
        times_cont = t_int / self.timesteps

        gamma_t = self.gamma(times_cont)

        alpha_t = torch.sqrt(torch.sigmoid(-gamma_t))
        sigma_t = torch.sqrt(torch.sigmoid(gamma_t))
        alpha_t = alpha_t[batch.batch]
        sigma_t = sigma_t[batch.batch]

        time_emb_cont = self.time_embedding_cont(times_cont)

        c0 = None
        c1 = None

        cart_coords = batch.cart_coords

        rand_c = torch.randn_like(cart_coords)
        per_graph_noise_mean = scatter(rand_c, batch.batch, dim=0, reduce="mean")
        rand_c = rand_c - per_graph_noise_mean[batch.batch]
        assert_mean_zero(rand_c, batch.batch)

        if self.time_step_style == "continuous":
            input_cart_coords = alpha_t * cart_coords + sigma_t * rand_c
        else:
            input_cart_coords = c0 * cart_coords + c1 * rand_c

        if self.keep_coords:
            input_cart_coords = cart_coords

        pred_c, _ = self.decoder(
            time_emb_cont,
            batch.atom_types,
            input_cart_coords,
            batch.num_atoms,
            batch.batch,
        )

        assert pred_c.shape == rand_c.shape

        per_node_mse = ((pred_c - rand_c) ** 2).sum(dim=-1)
        per_graph_mse = scatter(per_node_mse, batch.batch, dim=0, reduce="sum")
        per_graph_mse = per_graph_mse / (3.0 * batch.num_atoms.float())
        loss_coord = per_graph_mse.mean()

        loss = self.hparams.cost_coord * loss_coord

        with torch.no_grad():
            if self.time_step_style == "continuous":
                cart_coords_reconstructed = (
                    input_cart_coords - sigma_t * pred_c
                ) / alpha_t
            else:
                cart_coords_reconstructed = (input_cart_coords - c1 * pred_c) / c0

            coord_reconstructed_rmse = torch.sqrt(
                F.mse_loss(cart_coords_reconstructed, cart_coords)
            )

        return {
            "loss": loss,
            "loss_coord": loss_coord,
            "coord_reconstructed_rmse": coord_reconstructed_rmse,
        }

    @torch.no_grad()
    def sample(self, batch, step_lr=5e-6):
        batch_size = batch.num_graphs
        device = self.device

        x_T = torch.randn([batch.num_nodes, 3], device=device)
        x_T = x_T - scatter(x_T, batch.batch, dim=0, reduce="mean")[batch.batch]

        if self.keep_coords:
            x_T = batch.cart_coords.to(device)

        time_start = self.timesteps

        traj = {
            time_start: {
                "num_atoms": batch.num_atoms,
                "atom_types": batch.atom_types,
                "cart_coords": x_T,
            }
        }

        x_t = x_T

        for t in tqdm(range(time_start, 0, -1)):
            times_cont = torch.full((batch_size, 1), t, device=device)
            times_cont = times_cont / self.timesteps
            times_cont_prev = torch.full((batch_size, 1), t - 1, device=device)
            times_cont_prev = times_cont_prev / self.timesteps

            gamma_t = self.gamma(times_cont)
            gamma_t_prev = self.gamma(times_cont_prev)

            alpha_t = torch.sqrt(torch.sigmoid(-gamma_t)).to(device)
            sigma_t = torch.sqrt(torch.sigmoid(gamma_t)).to(device)

            alpha_t_prev = torch.sqrt(torch.sigmoid(-gamma_t_prev)).to(device)
            sigma_t_prev = torch.sqrt(torch.sigmoid(gamma_t_prev)).to(device)

            alpha_t_nodes = alpha_t[batch.batch]
            sigma_t_nodes = sigma_t[batch.batch]
            alpha_t_prev_nodes = alpha_t_prev[batch.batch]
            sigma_t_prev_nodes = sigma_t_prev[batch.batch]

            time_emb_cont = self.time_embedding_cont(times_cont)

            pred_eps, _ = self.decoder(
                time_emb_cont, batch.atom_types, x_t, batch.num_atoms, batch.batch
            )

            x_hat = (x_t - sigma_t_nodes * pred_eps) / alpha_t_nodes

            if t > 1:
                z = torch.randn_like(x_t, device=device)
                per_graph_z_mean = scatter(z, batch.batch, dim=0, reduce="mean")
                z = z - per_graph_z_mean[batch.batch]
            else:
                z = torch.zeros_like(x_t, device=device)

            sigma2_t_given_s = -torch.expm1(
                F.softplus(gamma_t_prev) - F.softplus(gamma_t)
            )
            log_alpha2_t = F.logsigmoid(-gamma_t)
            log_alpha2_s = F.logsigmoid(-gamma_t_prev)
            alpha_t_given_s = torch.exp(0.5 * (log_alpha2_t - log_alpha2_s))
            sigma_t_given_s = torch.sqrt(sigma2_t_given_s)

            alpha_t_given_s_nodes = alpha_t_given_s[batch.batch]
            sigma2_t_given_s_nodes = sigma2_t_given_s[batch.batch]
            sigma_t_given_s_nodes = sigma_t_given_s[batch.batch]

            mu = (
                x_t / alpha_t_given_s_nodes
                - (sigma2_t_given_s_nodes / alpha_t_given_s_nodes / sigma_t_nodes)
                * pred_eps
            )

            sigma_posterior = sigma_t_given_s_nodes * sigma_t_prev_nodes / sigma_t_nodes

            x_t_minus_1 = mu + sigma_posterior * z

            x_t = x_t_minus_1

            traj[t - 1] = {
                "num_atoms": batch.num_atoms,
                "atom_types": batch.atom_types,
                "cart_coords": x_t,
            }

        for t_key in traj:
            traj[t_key]["cart_coords"] = (
                traj[t_key]["cart_coords"] * self.coord_norm_factor
            )

        traj_stack = {
            "num_atoms": batch.num_atoms,
            "atom_types": batch.atom_types,
            "all_cart_coords": torch.stack(
                [traj[i]["cart_coords"] for i in range(time_start, -1, -1)]
            ),
        }

        return traj[0], traj_stack

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        output_dict = self(batch)
        loss_coord = output_dict["loss_coord"]
        loss = output_dict["loss"]
        coord_reconstructed_rmse = output_dict["coord_reconstructed_rmse"]

        self.log_dict(
            {
                "train_loss": loss,
                "coord_loss": loss_coord,
                "coord_reconstructed_rmse": coord_reconstructed_rmse,
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        if loss.isnan():
            return None
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        output_dict = self(batch)
        log_dict, loss = self.compute_stats(output_dict, prefix="val")
        self.log_dict(
            log_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        output_dict = self(batch)
        log_dict, loss = self.compute_stats(output_dict, prefix="test")
        self.log_dict(log_dict)
        return loss

    def compute_stats(self, output_dict, prefix):
        loss_coord = output_dict["loss_coord"]
        loss = output_dict["loss"]
        coord_reconstructed_rmse = output_dict["coord_reconstructed_rmse"]

        log_dict = {
            f"{prefix}_loss": loss,
            f"{prefix}_coord_loss": loss_coord,
            f"{prefix}_coord_reconstructed_rmse": coord_reconstructed_rmse,
        }

        return log_dict, loss


# ---------------------------------------------------------------------------
# MultiPolyDiffusion — NEW multi-polyhedron diffusion
# ---------------------------------------------------------------------------
### NEW: entire class
class MultiPolyDiffusion(BaseModule):
    """
    Multi-polyhedron diffusion model.

    Generates ALL unique polyhedra for a crystal jointly.
    Each crystal = 5 slots × 13 atoms = 65 atoms.

    Training losses:
      1. Coordinate loss: MSE on predicted noise, ONLY on real atoms
      2. Mask loss: BCE predicting real vs padding per atom
      3. Type loss: CE predicting element type on real non-center atoms

    Centering is done per-SLOT (each polyhedron independently), not per-graph.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.decoder = hydra.utils.instantiate(
            self.hparams.decoder,
            latent_dim=self.hparams.latent_dim + self.hparams.time_dim,
            multi_poly=True,  # enable multi-poly mode in EGNN
            _recursive_=False,
        )
        self.timesteps = self.hparams.timesteps
        self.time_dim = self.hparams.time_dim
        self.time_embedding_cont = SinusoidalPosEmb(self.time_dim)

        # Noise schedule
        self.noise_schedule = "polynomial_2.0"
        self.noise_precision = 1e-5
        self.gamma = PredefinedNoiseSchedule(
            self.noise_schedule,
            timesteps=self.timesteps,
            precision=self.noise_precision,
        )

        # Loss weights (configurable via hparams)
        self.loss_weight_mask = getattr(self.hparams, "loss_weight_mask", 0.1)
        self.loss_weight_type = getattr(self.hparams, "loss_weight_type", 0.1)

        self.coord_norm_factor = 1.0  # overridden by run.py from datamodule

    def _center_per_slot(self, coords, batch_batch):
        """
        Center coordinates independently for each polyhedron slot.
        All 13 atoms in a slot are centered to zero mean.
        """
        coords = coords.clone()
        batch_size = batch_batch.max().item() + 1

        for b in range(batch_size):
            graph_start = b * TOTAL_ATOMS
            for s in range(MAX_SLOTS):
                start = graph_start + s * ATOMS_PER_SLOT
                end = start + ATOMS_PER_SLOT
                slot_coords = coords[start:end]
                slot_mean = slot_coords.mean(dim=0)
                coords[start:end] = slot_coords - slot_mean

        return coords

    def _center_noise_per_slot(self, noise, real_mask, batch_batch):
        """
        Center noise per-slot on REAL atoms only (padding noise stays zero).
        This ensures translation invariance within each polyhedron.
        """
        noise = noise.clone()
        batch_size = batch_batch.max().item() + 1

        for b in range(batch_size):
            graph_start = b * TOTAL_ATOMS
            for s in range(MAX_SLOTS):
                start = graph_start + s * ATOMS_PER_SLOT
                end = start + ATOMS_PER_SLOT
                slot_real = real_mask[start:end].float()
                n_real = slot_real.sum()
                if n_real > 0:
                    slot_noise = noise[start:end]
                    real_mean = (slot_noise * slot_real.unsqueeze(-1)).sum(
                        dim=0
                    ) / n_real
                    noise[start:end] = noise[start:end] - real_mean.unsqueeze(
                        0
                    ) * slot_real.unsqueeze(-1)

        return noise

    def forward(self, batch):
        """
        Training forward pass with masked losses.
        """
        batch_size = batch.num_graphs

        # Sample timestep
        t_int = torch.randint(
            0, self.timesteps + 1, size=(batch_size, 1), device=self.device
        ).float()
        times_cont = t_int / self.timesteps

        gamma_t = self.gamma(times_cont)
        alpha_t = torch.sqrt(torch.sigmoid(-gamma_t))[batch.batch]  # (N_total, 1)
        sigma_t = torch.sqrt(torch.sigmoid(gamma_t))[batch.batch]

        time_emb_cont = self.time_embedding_cont(times_cont)

        cart_coords = batch.cart_coords
        real_mask_float = batch.real_mask.float()

        # --- Add noise ---
        rand_c = torch.randn_like(cart_coords)
        # Zero noise for padding atoms
        rand_c = rand_c * real_mask_float.unsqueeze(-1)
        # Center noise per-slot on real atoms
        rand_c = self._center_noise_per_slot(rand_c, batch.real_mask, batch.batch)

        input_cart_coords = alpha_t * cart_coords + sigma_t * rand_c

        # --- Forward through decoder with conditioning ---
        pred_c, mask_logits, type_logits = self.decoder(
            time_emb_cont,
            batch.atom_types,
            input_cart_coords,
            batch.num_atoms,
            batch.batch,
            center_mask=batch.center_mask,
            slot_ids=batch.slot_ids,
            comp_vec=batch.comp_vec,
        )

        # --- Loss 1: Coordinate loss (only real atoms) ---
        per_node_mse = ((pred_c - rand_c) ** 2).sum(dim=-1)  # (N_total,)
        per_node_mse = per_node_mse * real_mask_float  # zero out padding

        per_graph_mse = scatter(per_node_mse, batch.batch, dim=0, reduce="sum")
        num_real = scatter(real_mask_float, batch.batch, dim=0, reduce="sum").clamp(
            min=1
        )
        per_graph_mse = per_graph_mse / (3.0 * num_real)
        loss_coord = per_graph_mse.mean()

        # --- Loss 2: Mask loss (all atoms) ---
        loss_mask = F.binary_cross_entropy_with_logits(mask_logits, real_mask_float)

        # --- Loss 3: Type loss (real non-center atoms only) ---
        neighbor_real = batch.real_mask.bool() & (~batch.center_mask.bool())
        if neighbor_real.any():
            loss_type = F.cross_entropy(
                type_logits[neighbor_real], batch.gt_types[neighbor_real]
            )
        else:
            loss_type = torch.tensor(0.0, device=self.device)

        # --- Total loss ---
        loss = (
            loss_coord
            + self.loss_weight_mask * loss_mask
            + self.loss_weight_type * loss_type
        )

        # --- Monitoring metrics ---
        with torch.no_grad():
            pred_mask = (torch.sigmoid(mask_logits) > 0.5).float()
            mask_acc = (pred_mask == real_mask_float).float().mean()

            if neighbor_real.any():
                pred_types = type_logits[neighbor_real].argmax(dim=-1)
                type_acc = (pred_types == batch.gt_types[neighbor_real]).float().mean()
            else:
                type_acc = torch.tensor(1.0, device=self.device)

        return {
            "loss": loss,
            "loss_coord": loss_coord,
            "loss_mask": loss_mask,
            "loss_type": loss_type,
            "mask_acc": mask_acc,
            "type_acc": type_acc,
        }

    @torch.no_grad()
    def sample(self, batch, step_lr=5e-6):
        """
        Sampling with per-slot centering and mask/type extraction.
        """
        batch_size = batch.num_graphs
        device = self.device
        time_start = self.timesteps

        # Initialize from noise
        x_T = torch.randn(batch.num_nodes, 3, device=device)
        x_T = self._center_per_slot(x_T, batch.batch)

        traj = {time_start: {"cart_coords": x_T.clone()}}
        x_t = x_T

        for t in tqdm(range(time_start, 0, -1)):
            times_cont = torch.full((batch_size, 1), t / self.timesteps, device=device)
            times_cont_prev = torch.full(
                (batch_size, 1), (t - 1) / self.timesteps, device=device
            )

            gamma_t = self.gamma(times_cont)
            gamma_t_prev = self.gamma(times_cont_prev)

            alpha_t = torch.sqrt(torch.sigmoid(-gamma_t))
            sigma_t = torch.sqrt(torch.sigmoid(gamma_t))
            alpha_t_prev = torch.sqrt(torch.sigmoid(-gamma_t_prev))
            sigma_t_prev = torch.sqrt(torch.sigmoid(gamma_t_prev))

            alpha_t_nodes = alpha_t[batch.batch]
            sigma_t_nodes = sigma_t[batch.batch]
            sigma_t_prev_nodes = sigma_t_prev[batch.batch]

            time_emb = self.time_embedding_cont(times_cont)

            # Decoder with conditioning
            pred_eps, _, _ = self.decoder(
                time_emb,
                batch.atom_types,
                x_t,
                batch.num_atoms,
                batch.batch,
                center_mask=batch.center_mask,
                slot_ids=batch.slot_ids,
                comp_vec=batch.comp_vec,
            )

            # EDM posterior
            sigma2_t_given_s = -torch.expm1(
                F.softplus(gamma_t_prev) - F.softplus(gamma_t)
            )
            log_alpha2_t = F.logsigmoid(-gamma_t)
            log_alpha2_s = F.logsigmoid(-gamma_t_prev)
            alpha_t_given_s = torch.exp(0.5 * (log_alpha2_t - log_alpha2_s))
            sigma_t_given_s = torch.sqrt(sigma2_t_given_s)

            alpha_t_given_s_nodes = alpha_t_given_s[batch.batch]
            sigma2_t_given_s_nodes = sigma2_t_given_s[batch.batch]
            sigma_t_given_s_nodes = sigma_t_given_s[batch.batch]

            mu = (
                x_t / alpha_t_given_s_nodes
                - (sigma2_t_given_s_nodes / alpha_t_given_s_nodes / sigma_t_nodes)
                * pred_eps
            )
            sigma_posterior = sigma_t_given_s_nodes * sigma_t_prev_nodes / sigma_t_nodes

            if t > 1:
                z = torch.randn_like(x_t)
                x_t_minus_1 = mu + sigma_posterior * z
            else:
                x_t_minus_1 = mu

            # Center per slot
            x_t_minus_1 = self._center_per_slot(x_t_minus_1, batch.batch)

            x_t = x_t_minus_1
            traj[t - 1] = {"cart_coords": x_t.clone()}

        # --- Final pass at t=0 for mask/type predictions ---
        time_emb_0 = self.time_embedding_cont(torch.zeros(batch_size, 1, device=device))
        _, mask_logits, type_logits = self.decoder(
            time_emb_0,
            batch.atom_types,
            x_t,
            batch.num_atoms,
            batch.batch,
            center_mask=batch.center_mask,
            slot_ids=batch.slot_ids,
            comp_vec=batch.comp_vec,
        )

        # Denormalize coordinates
        for t_key in traj:
            traj[t_key]["cart_coords"] = (
                traj[t_key]["cart_coords"] * self.coord_norm_factor
            )

        final_coords = x_t * self.coord_norm_factor
        real_mask_pred = (torch.sigmoid(mask_logits) > 0.5).long()
        pred_atom_types = type_logits.argmax(dim=-1)

        output = {
            "cart_coords": final_coords,
            "atom_types": batch.atom_types,
            "num_atoms": batch.num_atoms,
            "real_mask": real_mask_pred,
            "pred_atom_types": pred_atom_types,
            "mask_logits": mask_logits,
            "type_logits": type_logits,
            "center_mask": batch.center_mask,
            "slot_ids": batch.slot_ids,
        }

        traj_stack = {
            "num_atoms": batch.num_atoms,
            "atom_types": batch.atom_types,
            "all_cart_coords": torch.stack(
                [traj[i]["cart_coords"] for i in range(time_start, -1, -1)]
            ),
        }

        return output, traj_stack

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        output_dict = self(batch)
        loss = output_dict["loss"]

        self.log_dict(
            {
                "train_loss": loss,
                "coord_loss": output_dict["loss_coord"],
                "mask_loss": output_dict["loss_mask"],
                "type_loss": output_dict["loss_type"],
                "mask_acc": output_dict["mask_acc"],
                "type_acc": output_dict["type_acc"],
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        if loss.isnan() or loss.isinf():
            return None
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        output_dict = self(batch)
        log_dict, loss = self._compute_stats(output_dict, prefix="val")
        self.log_dict(log_dict, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        output_dict = self(batch)
        log_dict, loss = self._compute_stats(output_dict, prefix="test")
        self.log_dict(log_dict)
        return loss

    def _compute_stats(self, output_dict, prefix):
        loss = output_dict["loss"]
        log_dict = {
            f"{prefix}_loss": loss,
            f"{prefix}_coord_loss": output_dict["loss_coord"],
            f"{prefix}_mask_loss": output_dict["loss_mask"],
            f"{prefix}_type_loss": output_dict["loss_type"],
            f"{prefix}_mask_acc": output_dict["mask_acc"],
            f"{prefix}_type_acc": output_dict["type_acc"],
        }
        return log_dict, loss


### END NEW
