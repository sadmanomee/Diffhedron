import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_scatter import scatter
import numpy as np
import math
from scipy.stats import norm
from abc import ABC, abstractmethod
from pathlib import Path
import equicsp


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


### modfications
def cosine_beta_schedule_edm(timesteps, s=0.008, raise_to_power: float = 1):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = np.clip(betas, a_min=0, a_max=0.999)
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)

    if raise_to_power != 1:
        alphas_cumprod = np.power(alphas_cumprod, raise_to_power)

    return alphas_cumprod


### modfications


def linear_beta_schedule(timesteps, beta_start, beta_end):
    return torch.linspace(beta_start, beta_end, timesteps)


def quadratic_beta_schedule(timesteps, beta_start, beta_end):
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2


def sigmoid_beta_schedule(timesteps, beta_start, beta_end):
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


def p_wrapped_normal(x, sigma, N=10, T=1.0):
    p_ = 0
    for i in range(-N, N + 1):
        p_ += torch.exp(-((x + T * i) ** 2) / 2 / sigma**2)
    return p_


def d_log_p_wrapped_normal(x, sigma, N=10, T=1.0):
    p_ = 0
    for i in range(-N, N + 1):
        p_ += (x + T * i) / sigma**2 * torch.exp(-((x + T * i) ** 2) / 2 / sigma**2)
    return p_ / p_wrapped_normal(x, sigma, N, T)


def sigma_norm(sigma, T=1.0, sn=10000):
    sigmas = sigma[None, :].repeat(sn, 1)
    x_sample = sigma * torch.randn_like(sigmas)
    x_sample = x_sample % T
    normal_ = d_log_p_wrapped_normal(x_sample, sigmas, T=T)
    return (normal_**2).mean(dim=0)


# normalization operation for better training in practice like most diffution model, which not mentioned in our paper
def sample_norm(sigma, T=1.0, sn=10000, num_atoms=52):
    # created by 'equicsp/pl_modules/von_mises_norm.py'
    # for normalization

    # sample_norm = torch.load(
    #     Path(equicsp.__file__).parent / "normalization" / "sample_norm.pth"
    # ).to("cpu")

    ### cpu version
    sample_norm = torch.load(
        Path(equicsp.__file__).parent / "normalization" / "sample_norm.pth",
        map_location="cpu",
    )
    ### cpu version
    return sample_norm


# normalization operation for better training in practice like most diffution model, which not mentioned in our paper
def kappa_func():
    # created by 'equicsp/pl_modules/von_mises_norm.py'
    # for normalization
    kappa_matrix = torch.load(
        Path(equicsp.__file__).parent / "normalization" / "kappa_matrix.pth"
    )
    kappa_norm = torch.load(
        Path(equicsp.__file__).parent / "normalization" / "kappa_norm.pth"
    )
    kappa_matrix = torch.tensor(kappa_matrix).float()
    kappa_norm = torch.tensor(kappa_norm).float()
    kappa_norm[1] = kappa_norm[1] + 1e-10
    return kappa_matrix, kappa_norm


class BetaScheduler(nn.Module):

    def __init__(self, timesteps, scheduler_mode, beta_start=0.0001, beta_end=0.02):
        super(BetaScheduler, self).__init__()
        self.timesteps = timesteps
        if scheduler_mode == "cosine":
            betas = cosine_beta_schedule(timesteps)
        elif scheduler_mode == "linear":
            betas = linear_beta_schedule(timesteps, beta_start, beta_end)
        elif scheduler_mode == "quadratic":
            betas = quadratic_beta_schedule(timesteps, beta_start, beta_end)
        elif scheduler_mode == "sigmoid":
            betas = sigmoid_beta_schedule(timesteps, beta_start, beta_end)

        betas = torch.cat([torch.zeros([1]), betas], dim=0)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)

        sigmas = torch.zeros_like(betas)

        sigmas[1:] = (
            betas[1:] * (1.0 - alphas_cumprod[:-1]) / (1.0 - alphas_cumprod[1:])
        )

        sigmas = torch.sqrt(sigmas)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sigmas", sigmas)

    def uniform_sample_t(self, batch_size, device):
        ts = np.random.choice(np.arange(1, self.timesteps + 1), batch_size)
        return torch.from_numpy(ts).to(device)

    def uniform_sample_t_no_zero(self, batch_size, device):
        ts = np.random.choice(np.arange(2, self.timesteps + 1), batch_size)
        return torch.from_numpy(ts).to(device)


class SigmaScheduler(nn.Module):

    def __init__(self, timesteps, sigma_begin=0.01, sigma_end=1.0):
        super(SigmaScheduler, self).__init__()
        self.timesteps = timesteps
        self.sigma_begin = sigma_begin
        self.sigma_end = sigma_end
        sigmas = torch.FloatTensor(
            np.exp(np.linspace(np.log(sigma_begin), np.log(sigma_end), timesteps))
        )

        sample_norm_ = sample_norm(sigmas)
        self.register_buffer(
            "sample_norm", torch.cat([torch.ones([52, 1]), sample_norm_], dim=1)
        )

        kappa_matrix_, kappa_norm_ = kappa_func()
        self.register_buffer(
            "kappa_matrix", torch.cat([torch.ones([53, 1]), kappa_matrix_], dim=1)
        )
        self.register_buffer(
            "kappa_norm", torch.cat([torch.ones([53, 1]), kappa_norm_], dim=1)
        )

        self.register_buffer("sigmas", torch.cat([torch.zeros([1]), sigmas], dim=0))

    def uniform_sample_t(self, batch_size, device):
        ts = np.random.choice(np.arange(1, self.timesteps + 1), batch_size)
        return torch.from_numpy(ts).to(device)


class LogNormalSampler:
    def __init__(self, p_mean=-1.2, p_std=1.2, even=False):
        self.p_mean = p_mean
        self.p_std = p_std

    def sample(self, bs, device):
        log_sigmas = self.p_mean + self.p_std * torch.randn(bs, device=device)
        sigmas = torch.exp(log_sigmas)
        weights = torch.ones_like(sigmas)
        return sigmas, weights


class ScheduleSampler(ABC):
    """
    A distribution over timesteps in the diffusion process, intended to reduce
    variance of the objective.

    By default, samplers perform unbiased importance sampling, in which the
    objective's mean is unchanged.
    However, subclasses may override sample() to change how the resampled
    terms are reweighted, allowing for actual changes in the objective.
    """

    @abstractmethod
    def weights(self):
        """
        Get a numpy array of weights, one per diffusion step.

        The weights needn't be normalized, but must be positive.
        """

    def sample(self, batch_size, device):
        """
        Importance-sample timesteps for a batch.

        :param batch_size: the number of timesteps.
        :param device: the torch device to save to.
        :return: a tuple (timesteps, weights):
                 - timesteps: a tensor of timestep indices.
                 - weights: a tensor of weights to scale the resulting losses.
        """
        w = self.weights()
        p = w / np.sum(w)
        indices_np = np.random.choice(len(p), size=(batch_size,), p=p)
        indices = torch.from_numpy(indices_np).long().to(device)
        weights_np = 1 / (len(p) * p[indices_np])
        weights = torch.from_numpy(weights_np).float().to(device)
        return indices, weights


class UniformSampler(ScheduleSampler):
    def __init__(self, diffusion):
        self.diffusion = diffusion
        self._weights = np.ones([diffusion.num_timesteps])

    def weights(self):
        return self._weights


def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        # print(targ.device)
        # print(src.device)
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)


class DummyGenerator:
    def randn(self, *args, **kwargs):
        return torch.randn(*args, **kwargs)

    def randint(self, *args, **kwargs):
        return torch.randint(*args, **kwargs)

    def randn_like(self, *args, **kwargs):
        return torch.randn_like(*args, **kwargs)


### modfications
### EDM schedulers and necessary functions


def assert_mean_zero(x, batch, tol=1e-4):
    """
    x: (N_total, 3)
    batch: (N_total,) integer graph IDs
    """
    per_graph_mean = scatter(x, batch, dim=0, reduce="mean")  # (num_graphs, 3)
    max_abs = per_graph_mean.abs().max().item()
    assert max_abs < tol, f"Nonzero mean detected: {max_abs:.2e}"


def clip_noise_schedule(alphas2, clip_value=0.001):
    """
    For a noise schedule given by alpha^2, this clips alpha_t / alpha_t-1. This may help improve stability during
    sampling.
    """
    alphas2 = np.concatenate([np.ones(1), alphas2], axis=0)

    alphas_step = alphas2[1:] / alphas2[:-1]

    alphas_step = np.clip(alphas_step, a_min=clip_value, a_max=1.0)
    alphas2 = np.cumprod(alphas_step, axis=0)

    return alphas2


def polynomial_schedule(timesteps: int, s=1e-4, power=3.0):
    """
    A noise schedule based on a simple polynomial equation: 1 - x^power.
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas2 = (1 - np.power(x / steps, power)) ** 2

    alphas2 = clip_noise_schedule(alphas2, clip_value=0.001)

    precision = 1 - 2 * s

    alphas2 = precision * alphas2 + s

    return alphas2


class PredefinedNoiseSchedule(torch.nn.Module):
    """
    Predefined noise schedule. Essentially creates a lookup array for predefined (non-learned) noise schedules.
    """

    def __init__(self, noise_schedule, timesteps, precision):
        super(PredefinedNoiseSchedule, self).__init__()
        self.timesteps = timesteps

        if noise_schedule == "cosine":
            alphas2 = cosine_beta_schedule_edm(timesteps)
        elif "polynomial" in noise_schedule:
            splits = noise_schedule.split("_")
            assert len(splits) == 2
            power = float(splits[1])
            alphas2 = polynomial_schedule(timesteps, s=precision, power=power)
        else:
            raise ValueError(noise_schedule)

        print("alphas2", alphas2)

        sigmas2 = 1 - alphas2

        log_alphas2 = np.log(alphas2)
        log_sigmas2 = np.log(sigmas2)

        log_alphas2_to_sigmas2 = log_alphas2 - log_sigmas2

        print("gamma", -log_alphas2_to_sigmas2)

        self.gamma = torch.nn.Parameter(
            torch.from_numpy(-log_alphas2_to_sigmas2).float(), requires_grad=False
        )

    def forward(self, t):
        t_int = torch.round(t * self.timesteps).long()
        return self.gamma[t_int]


def softplus(x: torch.Tensor) -> torch.Tensor:
    return F.softplus(x)


class PositiveLinear(torch.nn.Module):
    """Linear layer with weights forced to be positive."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        weight_init_offset: int = -2,
    ):
        super(PositiveLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.empty((out_features, in_features)))
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self.weight_init_offset = weight_init_offset
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        with torch.no_grad():
            self.weight.add_(self.weight_init_offset)

        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        positive_weight = softplus(self.weight)
        return F.linear(input, positive_weight, self.bias)


class GammaNetwork(torch.nn.Module):
    """The gamma network models a monotonic increasing function. Construction as in the VDM paper."""

    def __init__(self):
        super().__init__()

        self.l1 = PositiveLinear(1, 1)
        self.l2 = PositiveLinear(1, 1024)
        self.l3 = PositiveLinear(1024, 1)

        self.gamma_0 = torch.nn.Parameter(torch.tensor([-5.0]))
        self.gamma_1 = torch.nn.Parameter(torch.tensor([10.0]))
        self.show_schedule()

    def show_schedule(self, num_steps=50):
        t = torch.linspace(0, 1, num_steps).view(num_steps, 1)
        gamma = self.forward(t)
        print("Gamma schedule:")
        print(gamma.detach().cpu().numpy().reshape(num_steps))

    def gamma_tilde(self, t):
        l1_t = self.l1(t)
        return l1_t + self.l3(torch.sigmoid(self.l2(l1_t)))

    def forward(self, t):
        zeros, ones = torch.zeros_like(t), torch.ones_like(t)
        # Not super efficient.
        gamma_tilde_0 = self.gamma_tilde(zeros)
        gamma_tilde_1 = self.gamma_tilde(ones)
        gamma_tilde_t = self.gamma_tilde(t)

        # Normalize to [0, 1]
        normalized_gamma = (gamma_tilde_t - gamma_tilde_0) / (
            gamma_tilde_1 - gamma_tilde_0
        )

        # Rescale to [gamma_0, gamma_1]
        gamma = self.gamma_0 + (self.gamma_1 - self.gamma_0) * normalized_gamma

        return gamma


### modfications
