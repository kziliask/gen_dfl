import torch

from src.generators.gmm_generator import GMMGenerator


def test_gmm_outputs_and_sample_shape():
    torch.manual_seed(0)
    model = GMMGenerator(x_dim=3, c_dim=4, num_components=5, hidden_dim=16)
    x = torch.randn(7, 3)

    logits, means, log_scales = model(x)
    assert logits.shape == (7, 5)
    assert means.shape == (7, 5, 4)
    assert log_scales.shape == (7, 5, 4)

    samples = model.sample(200, x)
    assert samples.shape == (7, 200, 4)


def test_gmm_nll_is_finite_and_backpropagates():
    torch.manual_seed(1)
    model = GMMGenerator(x_dim=2, c_dim=3, num_components=2, hidden_dim=12)
    x = torch.randn(8, 2)
    c = torch.randn(8, 3)

    loss = model.nll(c, x, reduction="mean")
    assert torch.isfinite(loss)
    loss.backward()

    total_grad = 0.0
    for parameter in model.parameters():
        assert parameter.grad is not None
        total_grad += float(parameter.grad.abs().sum())
    assert total_grad > 0.0


def test_extreme_log_scales_are_clamped():
    model = GMMGenerator(x_dim=2, c_dim=2, num_components=3, hidden_dim=8)
    with torch.no_grad():
        model.log_scale_head.weight.fill_(100.0)
        model.log_scale_head.bias.fill_(100.0)
    x = torch.randn(5, 2)
    c = torch.randn(5, 2)
    assert torch.isfinite(model.nll(c, x)).all()


def test_k1_behaves_as_diagonal_gaussian_shape_contract():
    model = GMMGenerator(x_dim=2, c_dim=3, num_components=1, hidden_dim=8)
    x = torch.randn(4, 2)
    c = torch.randn(4, 3)

    logits, means, log_scales = model(x)
    assert logits.shape == (4, 1)
    assert means.shape == (4, 1, 3)
    assert log_scales.shape == (4, 1, 3)
    assert model.sample(10, x).shape == (4, 10, 3)
    assert torch.isfinite(model.nll(c, x)).all()

