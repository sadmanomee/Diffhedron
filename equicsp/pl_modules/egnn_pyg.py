"""
EGNN-based denoiser for Cartesian coordinate diffusion of crystals.
Pure PyTorch/PyG implementation — NO DGL dependency.

Faithfully follows the reference EGNN (Satorras et al.) architecture:
  - GCL layers: update node features h only (invariant message passing)
  - EquivariantUpdate: update coordinates x only (once per block)
  - EquivariantBlock: N x GCL layers + 1 x EquivariantUpdate
  - Output: x - x_input for coordinate noise, embedding_out(h) for feature noise

### NEW additions for multi-polyhedron:
  - center_emb: embedding for center vs neighbor/padding atoms
  - slot_emb: embedding for which polyhedron slot (0-4) each atom belongs to
  - comp_encoder: MLP encoding crystal composition vector
  - mask_head: predicts real vs padding per atom
  - type_head: predicts atom element per atom
  - gen_edges_per_slot: FC edges within each 13-atom slot, NO cross-slot edges
  - forward() accepts center_mask, slot_ids, comp_vec
  - Backward compatible: when conditioning args are None, falls back to old behavior
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
from torch_geometric.utils import dense_to_sparse

MAX_ATOMIC_NUM = 100

### NEW: import slot constants
from equicsp.common.data_utils import MAX_SLOTS, ATOMS_PER_SLOT, TOTAL_ATOMS
### END NEW


# ---------------------------------------------------------------------------
# Sinusoidal embedding for scalar distances (matches reference)
# ---------------------------------------------------------------------------
class SinusoidsEmbedding(nn.Module):
    """Sinusoidal positional embedding for scalar distances."""

    def __init__(self, max_res=15.0, min_res=15.0 / 2000.0, div_factor=4):
        super().__init__()
        self.n_frequencies = int(math.log(max_res / min_res, div_factor)) + 1
        self.frequencies = (
            2 * math.pi * div_factor ** torch.arange(self.n_frequencies) / max_res
        )
        self.dim = self.n_frequencies * 2  # sin + cos

    def forward(self, x):
        """x: (E, 1) squared distances."""
        x = torch.sqrt(x + 1e-8)
        emb = x * self.frequencies[None, :].to(x.device)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb.detach()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def coord2diff(x, edge_index, norm_constant=1):
    """Returns radial (squared dist) and normalized coord_diff."""
    src, dst = edge_index
    coord_diff = x[src] - x[dst]
    radial = (coord_diff**2).sum(dim=1, keepdim=True)
    norm = torch.sqrt(radial + 1e-8)
    coord_diff = coord_diff / (norm + norm_constant)
    return radial, coord_diff


def unsorted_segment_sum(
    data, segment_ids, num_segments, normalization_factor, aggregation_method
):
    """Scatter-add with configurable normalization."""
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)

    if aggregation_method == "sum":
        result = result / normalization_factor
    elif aggregation_method == "mean":
        norm = data.new_zeros(result.shape)
        norm.scatter_add_(0, segment_ids, data.new_ones(data.shape))
        norm[norm == 0] = 1
        result = result / norm

    return result


# ---------------------------------------------------------------------------
# GCL — Graph Convolution Layer (updates h ONLY, not x)
# ---------------------------------------------------------------------------
class GCL(nn.Module):
    """Invariant message-passing layer. Updates node features h only."""

    def __init__(
        self,
        input_nf,
        output_nf,
        hidden_nf,
        edges_in_d=0,
        act_fn=nn.SiLU(),
        attention=False,
        normalization_factor=1,
        aggregation_method="sum",
    ):
        super().__init__()
        input_edge = input_nf * 2
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        self.attention = attention

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf),
        )

        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid(),
            )

    def edge_model(self, source, target, edge_attr, edge_mask):
        if edge_attr is None:
            out = torch.cat([source, target], dim=1)
        else:
            out = torch.cat([source, target, edge_attr], dim=1)
        mij = self.edge_mlp(out)

        if self.attention:
            att_val = self.att_mlp(mij)
            out = mij * att_val
        else:
            out = mij

        if edge_mask is not None:
            out = out * edge_mask
        return out, mij

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(
            edge_attr,
            row,
            num_segments=x.size(0),
            normalization_factor=self.normalization_factor,
            aggregation_method=self.aggregation_method,
        )
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = x + self.node_mlp(agg)
        return out, agg

    def forward(
        self,
        h,
        edge_index,
        edge_attr=None,
        node_attr=None,
        node_mask=None,
        edge_mask=None,
    ):
        row, col = edge_index
        edge_feat, mij = self.edge_model(h[row], h[col], edge_attr, edge_mask)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)
        if node_mask is not None:
            h = h * node_mask
        return h, mij


# ---------------------------------------------------------------------------
# EquivariantUpdate — updates coordinates x ONLY (not h)
# ---------------------------------------------------------------------------
class EquivariantUpdate(nn.Module):
    """Equivariant coordinate update. Updates x only, using current h."""

    def __init__(
        self,
        hidden_nf,
        edges_in_d=1,
        act_fn=nn.SiLU(),
        tanh=False,
        coords_range=10.0,
        normalization_factor=1,
        aggregation_method="sum",
    ):
        super().__init__()
        self.tanh = tanh
        self.coords_range = coords_range
        input_edge = hidden_nf * 2 + edges_in_d
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method

        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        self.coord_mlp = nn.Sequential(
            nn.Linear(input_edge, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            layer,
        )

    def coord_model(self, h, coord, edge_index, coord_diff, edge_attr, edge_mask):
        row, col = edge_index
        input_tensor = torch.cat([h[row], h[col], edge_attr], dim=1)
        if self.tanh:
            trans = (
                coord_diff
                * torch.tanh(self.coord_mlp(input_tensor))
                * self.coords_range
            )
        else:
            trans = coord_diff * self.coord_mlp(input_tensor)
        if edge_mask is not None:
            trans = trans * edge_mask
        agg = unsorted_segment_sum(
            trans,
            row,
            num_segments=coord.size(0),
            normalization_factor=self.normalization_factor,
            aggregation_method=self.aggregation_method,
        )
        coord = coord + agg
        return coord

    def forward(
        self,
        h,
        coord,
        edge_index,
        coord_diff,
        edge_attr=None,
        node_mask=None,
        edge_mask=None,
    ):
        coord = self.coord_model(h, coord, edge_index, coord_diff, edge_attr, edge_mask)
        if node_mask is not None:
            coord = coord * node_mask
        return coord


# ---------------------------------------------------------------------------
# EquivariantBlock — N x GCL (h-only) + 1 x EquivariantUpdate (x-only)
# ---------------------------------------------------------------------------
class EquivariantBlock(nn.Module):
    def __init__(
        self,
        hidden_nf,
        edge_feat_nf=2,
        act_fn=nn.SiLU(),
        n_layers=2,
        attention=True,
        tanh=False,
        coords_range=15,
        norm_constant=1,
        sin_embedding=None,
        normalization_factor=1,
        aggregation_method="sum",
    ):
        super().__init__()
        self.hidden_nf = hidden_nf
        self.n_layers = n_layers
        self.coords_range_layer = float(coords_range)
        self.norm_constant = norm_constant
        self.sin_embedding = sin_embedding
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method

        for i in range(n_layers):
            self.add_module(
                "gcl_%d" % i,
                GCL(
                    hidden_nf,
                    hidden_nf,
                    hidden_nf,
                    edges_in_d=edge_feat_nf,
                    act_fn=act_fn,
                    attention=attention,
                    normalization_factor=normalization_factor,
                    aggregation_method=aggregation_method,
                ),
            )

        self.add_module(
            "gcl_equiv",
            EquivariantUpdate(
                hidden_nf,
                edges_in_d=edge_feat_nf,
                act_fn=nn.SiLU(),
                tanh=tanh,
                coords_range=self.coords_range_layer,
                normalization_factor=normalization_factor,
                aggregation_method=aggregation_method,
            ),
        )

    def forward(self, h, x, edge_index, node_mask=None, edge_mask=None, edge_attr=None):
        distances, coord_diff = coord2diff(x, edge_index, self.norm_constant)
        if self.sin_embedding is not None:
            distances = self.sin_embedding(distances)

        if edge_attr is not None:
            edge_attr_full = torch.cat([distances, edge_attr], dim=1)
        else:
            edge_attr_full = distances

        for i in range(self.n_layers):
            h, _ = self._modules["gcl_%d" % i](
                h,
                edge_index,
                edge_attr=edge_attr_full,
                node_mask=node_mask,
                edge_mask=edge_mask,
            )

        x = self._modules["gcl_equiv"](
            h,
            x,
            edge_index,
            coord_diff,
            edge_attr_full,
            node_mask,
            edge_mask,
        )

        if node_mask is not None:
            h = h * node_mask

        return h, x


# ---------------------------------------------------------------------------
# EGNNDenoiser — main model
# ---------------------------------------------------------------------------
class EGNNDenoiser(nn.Module):
    """
    EGNN-based noise predictor for Cartesian coordinate diffusion.

    ### NEW additions for multi-polyhedron mode:
    ###   - center_emb: embedding for center vs neighbor/padding
    ###   - slot_emb: embedding for which slot (0-4)
    ###   - comp_encoder: MLP encoding crystal composition vector (100,) → hidden_dim
    ###   - mask_head: predicts real vs padding per atom (sigmoid)
    ###   - type_head: predicts atom element per atom (softmax)
    ###   - gen_edges_per_slot: FC within each 13-atom slot, no cross-slot edges
    ###   - forward() accepts center_mask, slot_ids, comp_vec
    ###   - Backward compatible: when these are None, uses old edge generation
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        latent_dim: int = 256,
        num_layers: int = 4,
        max_atoms: int = 100,
        act_fn: str = "silu",
        edge_style: str = "fc",
        cutoff: float = 6.0,
        residual: bool = True,
        attention: bool = True,
        tanh: bool = False,
        coords_range: float = 15.0,
        norm_constant: float = 1.0,
        inv_sublayers: int = 2,
        sin_embedding: bool = False,
        normalization_factor: float = 1.0,
        aggregation_method: str = "sum",
        ln: bool = False,
        smooth: bool = False,
        pred_type: bool = False,
        pred_scalar: bool = False,
        dropout: float = 0.0,
        ### NEW parameters
        multi_poly: bool = False,       # enable multi-polyhedron mode
        comp_dim: int = 100,            # composition vector dimension
        ### END NEW
        # Accept and ignore extra kwargs from hydra config
        **kwargs,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.edge_style = edge_style
        self.cutoff = cutoff
        self.num_layers = num_layers
        self.smooth = smooth
        self.pred_type = pred_type
        self.pred_scalar = pred_scalar
        self.ln = ln
        self.coords_range_layer = float(coords_range / num_layers)
        self.norm_constant = norm_constant
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        ### NEW
        self.multi_poly = multi_poly
        ### END NEW

        activ_fn = nn.SiLU() if act_fn == "silu" else nn.ReLU()

        # --- Sinusoidal distance embedding ---
        if sin_embedding:
            self.sin_embedding = SinusoidsEmbedding()
            edge_feat_nf = self.sin_embedding.dim * 2
        else:
            self.sin_embedding = None
            edge_feat_nf = 2

        # --- Atom type embedding ---
        if self.smooth:
            self.node_embedding = nn.Linear(max_atoms, hidden_dim)
        else:
            ### NEW: +1 for PAD token (index 0) in multi-poly mode
            if self.multi_poly:
                self.node_embedding = nn.Embedding(max_atoms + 1, hidden_dim)  # 0=PAD
            else:
                self.node_embedding = nn.Embedding(max_atoms, hidden_dim)
            ### END NEW

        ### NEW: conditioning embeddings for multi-polyhedron mode
        if self.multi_poly:
            self.center_emb = nn.Embedding(2, hidden_dim)        # 0=neighbor/pad, 1=center
            self.slot_emb = nn.Embedding(MAX_SLOTS, hidden_dim)  # which polyhedron slot
            self.comp_encoder = nn.Sequential(
                nn.Linear(comp_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        ### END NEW

        # --- Input/output projections ---
        self.atom_latent_emb = nn.Linear(hidden_dim + latent_dim, hidden_dim)
        self.embedding_out = nn.Linear(hidden_dim, hidden_dim)

        # --- Equivariant blocks ---
        for i in range(num_layers):
            self.add_module(
                "e_block_%d" % i,
                EquivariantBlock(
                    hidden_nf=hidden_dim,
                    edge_feat_nf=edge_feat_nf,
                    act_fn=activ_fn,
                    n_layers=inv_sublayers,
                    attention=attention,
                    tanh=tanh,
                    coords_range=self.coords_range_layer,
                    norm_constant=norm_constant,
                    sin_embedding=self.sin_embedding,
                    normalization_factor=normalization_factor,
                    aggregation_method=aggregation_method,
                ),
            )

        # --- Output heads ---
        if self.ln:
            self.final_layer_norm = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if self.pred_type:
            self.type_out = nn.Linear(hidden_dim, MAX_ATOMIC_NUM)
        if self.pred_scalar:
            self.scalar_out = nn.Linear(hidden_dim, 1)

        ### NEW: multi-poly output heads
        if self.multi_poly:
            self.mask_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.SiLU(),
                nn.Linear(hidden_dim // 2, 1),
            )
            self.type_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, max_atoms + 1),  # 0=PAD, 1..100=elements
            )
        ### END NEW

    def gen_edges(self, num_atoms, cart_coords, node2graph):
        """Generate edges — fully connected or KNN. Original single-poly version."""
        if self.edge_style == "fc":
            lis = []
            for n in num_atoms:
                adj = torch.ones(n, n, device=num_atoms.device)
                adj.fill_diagonal_(0)
                lis.append(adj)
            fc_graph = torch.block_diag(*lis)
            fc_edges, _ = dense_to_sparse(fc_graph)
            return fc_edges, (cart_coords[fc_edges[1]] - cart_coords[fc_edges[0]])
        elif self.edge_style == "knn":
            edge_index_list, diff_list = [], []
            start = 0
            for n in num_atoms.tolist():
                n = int(n)
                if n > 0:
                    coords = cart_coords[start : start + n]
                    dists = torch.cdist(coords, coords)
                    mask = (dists <= self.cutoff) & (dists > 1e-5)
                    if mask.any():
                        i, j = torch.nonzero(mask, as_tuple=True)
                        edge_index_list.append(
                            torch.stack([i + start, j + start], dim=0)
                        )
                        diff_list.append(
                            cart_coords[j + start] - cart_coords[i + start]
                        )
                start += n
            if not edge_index_list:
                return (
                    torch.zeros((2, 0), dtype=torch.long, device=cart_coords.device),
                    torch.zeros((0, 3), device=cart_coords.device),
                )
            return torch.cat(edge_index_list, dim=1), torch.cat(diff_list, dim=0)
        else:
            raise ValueError(f"Unknown edge_style: {self.edge_style}")

    ### NEW: per-slot FC edge generation for multi-poly mode
    def gen_edges_per_slot(self, batch_size, device):
        """
        Build FC edges within each 13-atom slot. No edges between slots.
        
        For a batch of B crystals, each with MAX_SLOTS × ATOMS_PER_SLOT atoms:
          Slot 0: atoms 0-12, FC (12*13=156 directed edges)
          Slot 1: atoms 13-25, FC
          ...
          No edges between slots → polyhedra don't interact during denoising
        """
        # Pre-build adjacency for one slot (13×13 minus diagonal)
        slot_adj = torch.ones(ATOMS_PER_SLOT, ATOMS_PER_SLOT, device=device)
        slot_adj.fill_diagonal_(0)
        slot_edges_local = slot_adj.nonzero(as_tuple=False).t()  # (2, 156)
        
        all_src = []
        all_dst = []
        
        for b in range(batch_size):
            graph_offset = b * TOTAL_ATOMS
            for s in range(MAX_SLOTS):
                slot_offset = graph_offset + s * ATOMS_PER_SLOT
                all_src.append(slot_edges_local[0] + slot_offset)
                all_dst.append(slot_edges_local[1] + slot_offset)
        
        edge_index = torch.stack([
            torch.cat(all_src),
            torch.cat(all_dst),
        ], dim=0)
        
        return edge_index
    ### END NEW

    def forward(
        self,
        t: torch.Tensor,           # (B, time_dim)
        atom_types: torch.Tensor,   # (N_total,)
        cart_coords: torch.Tensor,  # (N_total, 3)
        num_atoms: torch.Tensor,    # (B,)
        node2graph: torch.Tensor,   # (N_total,)
        ### NEW: optional conditioning for multi-poly mode
        center_mask: torch.Tensor = None,   # (N_total,) 1=center atom
        slot_ids: torch.Tensor = None,      # (N_total,) slot index 0-4
        comp_vec: torch.Tensor = None,      # (B, 100) composition vector
        ### END NEW
        mean_operate: bool = False,
    ):
        # --- 1. Build node features ---
        if self.smooth:
            h = self.node_embedding(atom_types)
        else:
            ### NEW: multi-poly uses index directly (0=PAD has own embedding)
            if self.multi_poly:
                h = self.node_embedding(atom_types)
            else:
                h = self.node_embedding(atom_types - 1)
            ### END NEW

        ### NEW: add multi-poly conditioning
        if self.multi_poly and center_mask is not None:
            h = h + self.center_emb(center_mask)
        if self.multi_poly and slot_ids is not None:
            h = h + self.slot_emb(slot_ids)
        if self.multi_poly and comp_vec is not None:
            comp_vec = comp_vec.view(-1, 100)  # unflatten from PyG concat
            comp_emb = self.comp_encoder(comp_vec)        # (B, hidden_dim)
            comp_per_node = comp_emb[node2graph]           # (N_total, hidden_dim)
            h = h + comp_per_node
        ### END NEW

        t_per_atom = t.repeat_interleave(num_atoms, dim=0)
        h = torch.cat([h, t_per_atom], dim=1)
        h = self.atom_latent_emb(h)  # (N_total, hidden_dim)

        # --- 2. Build edges ---
        ### NEW: use per-slot edges in multi-poly mode
        if self.multi_poly and center_mask is not None:
            batch_size = len(num_atoms)
            edge_index = self.gen_edges_per_slot(batch_size, cart_coords.device)
        else:
            edge_index, _ = self.gen_edges(num_atoms, cart_coords, node2graph)
        ### END NEW

        # --- 3. Cache input coords ---
        x = cart_coords.clone()
        x_input = cart_coords

        # --- 4. Compute initial distances ---
        radial_init, _ = coord2diff(x, edge_index, self.norm_constant)
        if self.sin_embedding is not None:
            edge_attr = self.sin_embedding(radial_init)
        else:
            edge_attr = radial_init

        # --- 5. Run equivariant blocks ---
        for i in range(self.num_layers):
            h, x = self._modules["e_block_%d" % i](
                h,
                x,
                edge_index,
                edge_attr=edge_attr,
            )

        # --- 6. Output ---
        pred_noise = x - x_input  # equivariant displacement

        if self.ln:
            h = self.final_layer_norm(h)
        h = self.dropout(h)
        h = self.embedding_out(h)

        if mean_operate:
            context_mean = scatter(pred_noise, node2graph, dim=0, reduce="mean")
            pred_noise = pred_noise - context_mean[node2graph]

        if self.pred_scalar:
            graph_features = scatter(h, node2graph, dim=0, reduce="mean")
            return self.scalar_out(graph_features)

        ### NEW: multi-poly returns mask_logits and type_logits
        if self.multi_poly:
            mask_logits = self.mask_head(h).squeeze(-1)   # (N_total,)
            type_logits = self.type_head(h)               # (N_total, max_atoms+1)
            return pred_noise, mask_logits, type_logits
        ### END NEW

        log_out = None
        if self.pred_type:
            type_out = self.type_out(h)
            return pred_noise, log_out, type_out

        return pred_noise, log_out
