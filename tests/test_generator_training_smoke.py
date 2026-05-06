import torch

from src.generators.cnf import ConditionalFlow
from src.generators.gmm_generator import GMMGenerator
from src.training.generator_losses import generator_nll, set_global_seed


def test_gmm_nll_overfit_smoke():
    set_global_seed(7)
    x = torch.randn(16, 3)
    weights = torch.tensor(
        [
            [0.5, -0.2],
            [0.1, 0.3],
            [-0.4, 0.2],
        ]
    )
    c = x @ weights + 0.05 * torch.randn(16, 2)
    model = GMMGenerator(x_dim=3, c_dim=2, num_components=1, hidden_dim=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    initial = model.nll(c, x, reduction="mean").item()
    for _ in range(50):
        loss = model.nll(c, x, reduction="mean")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    final = model.nll(c, x, reduction="mean").item()

    assert torch.isfinite(torch.tensor(final))
    assert final < initial


def test_generator_nll_handles_cnf_and_gmm():
    set_global_seed(3)
    x = torch.randn(5, 2)
    c = torch.randn(5, 3)

    cnf = ConditionalFlow(c_dim=3, x_dim=2, hidden_dim=8, n_layers=2)
    gmm = GMMGenerator(x_dim=2, c_dim=3, num_components=2, hidden_dim=8)

    assert torch.isfinite(generator_nll(cnf, c, x, reduction="mean"))
    assert torch.isfinite(generator_nll(gmm, c, x, reduction="mean"))

