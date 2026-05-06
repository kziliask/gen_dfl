"""Lightweight conditional normalizing flow used by the official scripts."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class ConditionalCouplingLayer(nn.Module):
    """Affine coupling layer conditioned on feature tensor ``x``."""

    def __init__(self, c_dim: int, x_dim: int, hidden_dim: int = 128, mask_type: str = "alternate"):
        super().__init__()
        self.c_dim = c_dim
        if mask_type == "alternate":
            mask = torch.arange(c_dim) % 2
        else:
            mask = torch.cat([torch.ones(c_dim // 2), torch.zeros(c_dim - c_dim // 2)])
        self.register_buffer("mask", mask.bool())

        input_dim = x_dim + (c_dim - int(self.mask.sum().item()))
        self.scale_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, int(self.mask.sum().item())),
            nn.Tanh(),
        )
        self.translation_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, int(self.mask.sum().item())),
        )

    def forward(
        self,
        c: Tensor,
        x: Tensor,
        log_det: Tensor | None = None,
        reverse: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        c1 = c[:, self.mask]
        c2 = c[:, ~self.mask]
        h = torch.cat([x, c2], dim=1)

        scale = self.scale_net(h)
        shift = self.translation_net(h)

        if reverse:
            c1_transformed = (c1 - shift) * torch.exp(-scale)
            if log_det is not None:
                log_det = log_det - scale.sum(dim=1)
        else:
            c1_transformed = c1 * torch.exp(scale) + shift
            if log_det is not None:
                log_det = log_det + scale.sum(dim=1)

        c_new = c.clone()
        c_new[:, self.mask] = c1_transformed
        c_new[:, ~self.mask] = c2
        if log_det is not None:
            return c_new, log_det
        return c_new


class ConditionalFlow(nn.Module):
    """Conditional affine-coupling flow with ``sample`` and ``nll`` methods."""

    def __init__(self, c_dim: int, x_dim: int, n_layers: int = 4, hidden_dim: int = 128):
        super().__init__()
        self.c_dim = c_dim
        self.x_dim = x_dim
        self.layers = nn.ModuleList(
            [
                ConditionalCouplingLayer(
                    c_dim=c_dim,
                    x_dim=x_dim,
                    hidden_dim=hidden_dim,
                    mask_type="alternate" if i % 2 == 0 else "half",
                )
                for i in range(n_layers)
            ]
        )

    def forward(self, c: Tensor, x: Tensor, reverse: bool = False) -> tuple[Tensor, Tensor]:
        log_det = torch.zeros(c.size(0), device=c.device, dtype=c.dtype)
        layers = reversed(self.layers) if reverse else self.layers
        for layer in layers:
            c, log_det = layer(c, x, log_det, reverse=reverse)
        return c, log_det

    def log_prob(self, c: Tensor, x: Tensor) -> Tensor:
        z, log_det = self.forward(c, x)
        base_log_prob = -0.5 * torch.sum(z**2, dim=1)
        base_log_prob = base_log_prob - 0.5 * z.size(1) * math.log(2.0 * math.pi)
        return base_log_prob + log_det

    def nll(self, c: Tensor, x: Tensor, reduction: str = "none") -> Tensor:
        loss = -self.log_prob(c, x)
        if reduction == "mean":
            return loss.mean()
        if reduction == "sum":
            return loss.sum()
        if reduction != "none":
            raise ValueError(f"Unknown reduction: {reduction}")
        return loss

    def sample(self, num_samples: int, x: Tensor, temperature: float = 1.0) -> Tensor:
        batch_size = x.size(0)
        x_expanded = x.unsqueeze(1).expand(-1, num_samples, -1)
        x_reshaped = x_expanded.reshape(-1, self.x_dim)

        z = torch.randn(
            batch_size,
            num_samples,
            self.c_dim,
            device=x.device,
            dtype=x.dtype,
        ) * temperature
        z_reshaped = z.reshape(-1, self.c_dim)
        samples, _ = self.forward(z_reshaped, x_reshaped, reverse=True)
        return samples.reshape(batch_size, num_samples, self.c_dim)

