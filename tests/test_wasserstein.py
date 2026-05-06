import torch

from src.evaluation.wasserstein import gaussian_w2, sliced_wasserstein_distance


def test_gaussian_w2_zero_for_identical_gaussians():
    mu = torch.zeros(3)
    sigma = torch.eye(3)
    value = gaussian_w2(mu, sigma, mu, sigma)
    assert torch.allclose(value, torch.tensor(0.0, dtype=torch.double), atol=1e-6)


def test_gaussian_w2_equals_shift_norm_for_same_covariance():
    mu = torch.zeros(4)
    shift = torch.tensor([0.5, -0.25, 0.0, 1.0])
    sigma = torch.eye(4) * 2.0
    value = gaussian_w2(mu, sigma, mu + shift, sigma)
    assert torch.allclose(value, shift.double().norm(), atol=1e-6)


def test_sliced_wasserstein_near_zero_for_identical_samples():
    torch.manual_seed(0)
    samples = torch.randn(128, 3)
    value = sliced_wasserstein_distance(samples, samples.clone(), num_projections=32, seed=1)
    assert value.item() < 1e-8


def test_sliced_wasserstein_increases_under_mean_shift():
    torch.manual_seed(0)
    samples = torch.randn(256, 2)
    shifted = samples + torch.tensor([1.0, 0.0])
    same = sliced_wasserstein_distance(samples, samples.clone(), num_projections=32, seed=2)
    moved = sliced_wasserstein_distance(samples, shifted, num_projections=32, seed=2)
    assert moved > same

