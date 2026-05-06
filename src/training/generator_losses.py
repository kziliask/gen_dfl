"""Shared training helpers for generator experiments."""

from __future__ import annotations

import math
import random
from typing import Iterable

import numpy as np
import torch
from torch import Tensor, nn


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generator_nll(generator: nn.Module, c: Tensor, x: Tensor, reduction: str = "mean") -> Tensor:
    """Compute NLL for either the new generator interface or legacy flows."""
    if hasattr(generator, "nll"):
        return generator.nll(c, x, reduction=reduction)

    z, log_det = generator(c, x)
    base_log_prob = -0.5 * torch.sum(z**2, dim=1)
    base_log_prob = base_log_prob - 0.5 * z.size(1) * math.log(2.0 * math.pi)
    loss = -(base_log_prob + log_det)
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    if reduction != "none":
        raise ValueError(f"Unknown reduction: {reduction}")
    return loss


def grad_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    squared_norm = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            squared_norm += float(parameter.grad.detach().pow(2).sum().cpu())
    return squared_norm ** 0.5

