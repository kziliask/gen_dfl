"""CNF adapter matching the generator interface used by MDN/GMM models."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from src.generators.cnf import ConditionalFlow


class CNFGeneratorWrapper(nn.Module):
    """Wrap a ``ConditionalFlow`` behind ``sample`` and ``nll`` methods."""

    def __init__(self, flow: ConditionalFlow):
        super().__init__()
        self.flow = flow
        self.c_dim = flow.c_dim
        self.x_dim = flow.x_dim

    def forward(self, c: Tensor, x: Tensor, reverse: bool = False) -> tuple[Tensor, Tensor]:
        return self.flow(c, x, reverse=reverse)

    def sample(self, num_samples: int, x: Tensor, temperature: float = 1.0) -> Tensor:
        return self.flow.sample(num_samples, x, temperature=temperature)

    def nll(self, c: Tensor, x: Tensor, reduction: str = "none") -> Tensor:
        return self.flow.nll(c, x, reduction=reduction)

    def log_prob(self, c: Tensor, x: Tensor) -> Tensor:
        return self.flow.log_prob(c, x)


def build_cnf_generator(c_dim: int, x_dim: int, **kwargs) -> ConditionalFlow:
    return ConditionalFlow(c_dim=c_dim, x_dim=x_dim, **kwargs)

