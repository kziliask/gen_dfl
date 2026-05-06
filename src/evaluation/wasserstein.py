"""Distribution distance diagnostics used by the reproducibility plan."""

from __future__ import annotations

import torch
from torch import Tensor


def _to_double_tensor(value: Tensor | object) -> Tensor:
    return torch.as_tensor(value, dtype=torch.double)


def _symmetrize(matrix: Tensor) -> Tensor:
    return 0.5 * (matrix + matrix.transpose(-2, -1))


def _sqrtm_psd(matrix: Tensor) -> Tensor:
    matrix = _symmetrize(matrix)
    eigvals, eigvecs = torch.linalg.eigh(matrix)
    eigvals = eigvals.clamp_min(0.0)
    return eigvecs @ torch.diag_embed(torch.sqrt(eigvals)) @ eigvecs.transpose(-2, -1)


def gaussian_w2_squared(mu_p: Tensor | object, sigma_p: Tensor | object, mu_q: Tensor | object, sigma_q: Tensor | object) -> Tensor:
    """Return squared 2-Wasserstein distance between Gaussians."""
    mu_p_t = _to_double_tensor(mu_p)
    sigma_p_t = _to_double_tensor(sigma_p)
    mu_q_t = _to_double_tensor(mu_q)
    sigma_q_t = _to_double_tensor(sigma_q)

    diff_sq = (mu_p_t - mu_q_t).pow(2).sum(dim=-1)
    sigma_q_sqrt = _sqrtm_psd(sigma_q_t)
    middle = sigma_q_sqrt @ sigma_p_t @ sigma_q_sqrt
    middle_sqrt = _sqrtm_psd(middle)
    trace_term = torch.diagonal(
        sigma_p_t + sigma_q_t - 2.0 * middle_sqrt,
        dim1=-2,
        dim2=-1,
    ).sum(dim=-1)
    return (diff_sq + trace_term).clamp_min(0.0)


def gaussian_w2(mu_p: Tensor | object, sigma_p: Tensor | object, mu_q: Tensor | object, sigma_q: Tensor | object) -> Tensor:
    """Return 2-Wasserstein distance between Gaussians."""
    return torch.sqrt(gaussian_w2_squared(mu_p, sigma_p, mu_q, sigma_q))


def random_projection_directions(
    dim: int,
    num_projections: int,
    seed: int | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.double,
) -> Tensor:
    generator = torch.Generator(device="cpu")
    if seed is not None:
        generator.manual_seed(seed)
    directions = torch.randn(num_projections, dim, generator=generator, dtype=dtype)
    directions = directions / directions.norm(dim=1, keepdim=True).clamp_min(1e-12)
    if device is not None:
        directions = directions.to(device)
    return directions


def _flatten_samples(samples: Tensor | object) -> Tensor:
    tensor = _to_double_tensor(samples)
    if tensor.ndim == 3:
        return tensor.reshape(-1, tensor.shape[-1])
    if tensor.ndim == 2:
        return tensor
    raise ValueError("samples must have shape [n, d] or [batch, n, d]")


def _resample_sorted(sorted_values: Tensor, target_size: int) -> Tensor:
    n = sorted_values.shape[0]
    if n == target_size:
        return sorted_values
    positions = torch.linspace(0, n - 1, target_size, device=sorted_values.device, dtype=sorted_values.dtype)
    lower = torch.floor(positions).long()
    upper = torch.ceil(positions).long()
    weight = (positions - lower.to(positions.dtype)).unsqueeze(-1)
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def sliced_wasserstein_distance(
    samples_p: Tensor | object,
    samples_q: Tensor | object,
    num_projections: int = 64,
    seed: int | None = 0,
    directions: Tensor | None = None,
    squared: bool = False,
) -> Tensor:
    """Estimate sliced W2 from empirical samples."""
    p = _flatten_samples(samples_p)
    q = _flatten_samples(samples_q)
    if p.shape[-1] != q.shape[-1]:
        raise ValueError("sample dimensions must match")

    if directions is None:
        directions = random_projection_directions(
            p.shape[-1],
            num_projections,
            seed=seed,
            device=p.device,
            dtype=p.dtype,
        )
    else:
        directions = directions.to(device=p.device, dtype=p.dtype)

    p_projected = torch.sort(p @ directions.T, dim=0).values
    q_projected = torch.sort(q @ directions.T, dim=0).values
    target_size = max(p_projected.shape[0], q_projected.shape[0])
    p_projected = _resample_sorted(p_projected, target_size)
    q_projected = _resample_sorted(q_projected, target_size)
    value = (p_projected - q_projected).pow(2).mean()
    return value if squared else torch.sqrt(value)

