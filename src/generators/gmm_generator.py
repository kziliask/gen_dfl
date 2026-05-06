"""Conditional diagonal Gaussian mixture / MDN generator."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class MDNGenerator(nn.Module):
    """Mixture-density network with diagonal Gaussian components.

    The class implements the generator contract used in the experiment plan:
    ``sample(num_samples, x) -> [batch, num_samples, c_dim]`` and
    ``nll(c, x) -> [batch]`` by default.
    """

    def __init__(
        self,
        x_dim: int,
        c_dim: int,
        num_components: int = 5,
        hidden_dim: int = 128,
        num_hidden_layers: int = 2,
        min_log_scale: float = -7.0,
        max_log_scale: float = 5.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        if num_components < 1:
            raise ValueError("num_components must be >= 1")
        self.x_dim = x_dim
        self.c_dim = c_dim
        self.num_components = num_components
        self.min_log_scale = min_log_scale
        self.max_log_scale = max_log_scale
        self.eps = eps

        layers: list[nn.Module] = []
        in_dim = x_dim
        for _ in range(num_hidden_layers):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.ReLU()])
            in_dim = hidden_dim
        self.backbone = nn.Sequential(*layers) if layers else nn.Identity()
        self.logit_head = nn.Linear(in_dim, num_components)
        self.mean_head = nn.Linear(in_dim, num_components * c_dim)
        self.log_scale_head = nn.Linear(in_dim, num_components * c_dim)
        nn.init.constant_(self.log_scale_head.bias, -0.5)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        h = self.backbone(x)
        logits = self.logit_head(h)
        means = self.mean_head(h).view(-1, self.num_components, self.c_dim)
        log_scales = self.log_scale_head(h).view(-1, self.num_components, self.c_dim)
        log_scales = log_scales.clamp(self.min_log_scale, self.max_log_scale)
        return logits, means, log_scales

    def log_prob_components(self, c: Tensor, x: Tensor) -> Tensor:
        logits, means, log_scales = self.forward(x)
        c_expanded = c.unsqueeze(1)
        scales = log_scales.exp().clamp_min(self.eps)
        standardized = (c_expanded - means) / scales
        log_prob = -0.5 * standardized.pow(2) - log_scales - 0.5 * math.log(2.0 * math.pi)
        log_prob = log_prob.sum(dim=-1)
        return F.log_softmax(logits, dim=-1) + log_prob

    def log_prob(self, c: Tensor, x: Tensor) -> Tensor:
        return torch.logsumexp(self.log_prob_components(c, x), dim=-1)

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
        logits, means, log_scales = self.forward(x)
        batch_size = x.size(0)
        components = torch.distributions.Categorical(logits=logits).sample((num_samples,))
        components = components.transpose(0, 1).contiguous()
        gather_index = components.unsqueeze(-1).expand(batch_size, num_samples, self.c_dim)
        selected_means = torch.gather(means, dim=1, index=gather_index)
        selected_log_scales = torch.gather(log_scales, dim=1, index=gather_index)
        scales = selected_log_scales.exp().clamp_min(self.eps)
        noise = torch.randn_like(selected_means)
        return selected_means + noise * scales * temperature


GMMGenerator = MDNGenerator

