"""
To launch all the tasks, create tmux sessions (separately for each of the following) 
and run (for instance):

python canvi_sbibm.py --task two_moons --cuda_idx 0
python canvi_sbibm.py --task slcp --cuda_idx 1
python canvi_sbibm.py --task gaussian_linear_uniform --cuda_idx 2
python canvi_sbibm.py --task bernoulli_glm --cuda_idx 3
python canvi_sbibm.py --task gaussian_mixture --cuda_idx 4
python canvi_sbibm.py --task gaussian_linear --cuda_idx 5
python canvi_sbibm.py --task slcp_distractors --cuda_idx 6
python canvi_sbibm.py --task bernoulli_glm_raw --cuda_idx 7
"""

import pandas as pd
import numpy as np
import torch
import math
import torch.distributions as D
import matplotlib.pyplot as plt

from functools import partial
from typing import Optional
from warnings import warn
from torch import Tensor, nn, relu, tanh, tensor, uint8

import matplotlib as mpl
import matplotlib.pyplot as plt

try:
    import seaborn as sns
except ImportError:
    sns = None

mpl.rcParams['text.usetex'] = True
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsfonts}'

if sns is not None:
    sns.set_theme()

try:
    import sbibm
    from pyknos.nflows import distributions as distributions_
    from pyknos.nflows import flows, transforms
    from pyknos.nflows.nn import nets
    from pyknos.nflows.transforms.splines import rational_quadratic
    from sbi.utils.sbiutils import (
        standardizing_net,
        standardizing_transform,
        z_score_parser,
    )
    from sbi.utils.torchutils import create_alternating_binary_mask
    from sbi.utils.user_input_checks import check_data_device, check_embedding_net_device
    _SBI_IMPORT_ERROR = None
except ImportError as exc:
    sbibm = None
    distributions_ = flows = transforms = nets = rational_quadratic = None
    standardizing_net = standardizing_transform = z_score_parser = None
    create_alternating_binary_mask = None
    check_data_device = check_embedding_net_device = None
    _SBI_IMPORT_ERROR = exc

import os
import pickle
import argparse


class ContextSplineMap(nn.Module):
    """
    Neural network from `context` to the spline parameters.
    We cannot use the resnet as conditioner to learn each dimension conditioned
    on the other dimensions (because there is only one). Instead, we learn the
    spline parameters directly. In the case of conditinal density estimation,
    we make the spline parameters conditional on the context. This is
    implemented in this class.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: int,
        context_features: int,
        hidden_layers: int,
    ):
        """
        Initialize neural network that learns to predict spline parameters.
        Args:
            in_features: Unused since there is no `conditioner` in 1D.
            out_features: Number of spline parameters.
            hidden_features: Number of hidden units.
            context_features: Number of context features.
        """
        super().__init__()
        # `self.hidden_features` is only defined such that nflows can infer
        # a scaling factor for initializations.
        self.hidden_features = hidden_features

        # Use a non-linearity because otherwise, there will be a linear
        # mapping from context features onto distribution parameters.

        # Initialize with input layer.
        layer_list = [nn.Linear(context_features, hidden_features), nn.ReLU()]
        # Add hidden layers.
        layer_list += [
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(),
        ] * hidden_layers
        # Add output layer.
        layer_list += [nn.Linear(hidden_features, out_features)]
        self.spline_predictor = nn.Sequential(*layer_list)

    def __call__(self, inputs: Tensor, context: Tensor, *args, **kwargs) -> Tensor:
        """
        Return parameters of the spline given the context.
        Args:
            inputs: Unused. It would usually be the other dimensions, but in
                1D, there are no other dimensions.
            context: Context features.
        Returns:
            Spline parameters.
        """
        return self.spline_predictor(context)

# Declan: this code from SBI library
def build_nsf(
    batch_x: Tensor,
    batch_y: Tensor,
    z_score_x: Optional[str] = "independent",
    z_score_y: Optional[str] = "independent",
    hidden_features: int = 50,
    num_transforms: int = 5,
    num_bins: int = 10,
    embedding_net: nn.Module = nn.Identity(),
    tail_bound: float = 3.0,
    hidden_layers_spline_context: int = 1,
    num_blocks: int = 2,
    dropout_probability: float = 0.0,
    use_batch_norm: bool = False,
    **kwargs,
) -> nn.Module:
    """Builds NSF p(x|y).
    Args:
        batch_x: Batch of xs, used to infer dimensionality and (optional) z-scoring.
        batch_y: Batch of ys, used to infer dimensionality and (optional) z-scoring.
        z_score_x: Whether to z-score xs passing into the network, can be one of:
            - `none`, or None: do not z-score.
            - `independent`: z-score each dimension independently.
            - `structured`: treat dimensions as related, therefore compute mean and std
            over the entire batch, instead of per-dimension. Should be used when each
            sample is, for example, a time series or an image.
        z_score_y: Whether to z-score ys passing into the network, same options as
            z_score_x.
        hidden_features: Number of hidden features.
        num_transforms: Number of transforms.
        num_bins: Number of bins used for the splines.
        embedding_net: Optional embedding network for y.
        tail_bound: tail bound for each spline.
        hidden_layers_spline_context: number of hidden layers of the spline context net
            for one-dimensional x.
        num_blocks: number of blocks used for residual net for context embedding.
        dropout_probability: dropout probability for regularization in residual net.
        use_batch_norm: whether to use batch norm in residual net.
        kwargs: Additional arguments that are passed by the build function but are not
            relevant for maf and are therefore ignored.
    Returns:
        Neural network.
    """
    if _SBI_IMPORT_ERROR is not None:
        raise ImportError(
            "build_nsf requires optional SBI dependencies: sbi, sbibm, and pyknos. "
            "The main Gen-DFL synthetic scripts use src.generators.cnf.ConditionalFlow "
            "and do not require these packages."
        ) from _SBI_IMPORT_ERROR
    x_numel = batch_x[0].numel()
    # Infer the output dimensionality of the embedding_net by making a forward pass.
    check_data_device(batch_x, batch_y)
    check_embedding_net_device(embedding_net=embedding_net, datum=batch_y)
    y_numel = embedding_net(batch_y[:1]).numel()

    # Define mask function to alternate between predicted x-dimensions.
    def mask_in_layer(i):
        return create_alternating_binary_mask(features=x_numel, even=(i % 2 == 0))

    # If x is just a scalar then use a dummy mask and learn spline parameters using the
    # conditioning variables only.
    if x_numel == 1:
        # Conditioner ignores the data and uses the conditioning variables only.
        conditioner = partial(
            ContextSplineMap,
            hidden_features=hidden_features,
            context_features=y_numel,
            hidden_layers=hidden_layers_spline_context,
        )
    else:
        # Use conditional resnet as spline conditioner.
        conditioner = partial(
            nets.ResidualNet,
            hidden_features=hidden_features,
            context_features=y_numel,
            num_blocks=num_blocks,
            activation=relu,
            dropout_probability=dropout_probability,
            use_batch_norm=use_batch_norm,
        )

    # Stack spline transforms.
    transform_list = []
    for i in range(num_transforms):
        block = [
            transforms.PiecewiseRationalQuadraticCouplingTransform(
                mask=mask_in_layer(i) if x_numel > 1 else tensor([1], dtype=uint8),
                transform_net_create_fn=conditioner,
                num_bins=num_bins,
                tails="linear",
                tail_bound=tail_bound,
                apply_unconditional_transform=False,
            )
        ]
        # Add LU transform only for high D x. Permutation makes sense only for more than
        # one feature.
        if x_numel > 1:
            block.append(
                transforms.LULinear(x_numel, identity_init=True),
            )
        transform_list += block

    z_score_x_bool, structured_x = z_score_parser(z_score_x)
    if z_score_x_bool:
        # Prepend standardizing transform to nsf transforms.
        transform_list = [
            standardizing_transform(batch_x, structured_x)
        ] + transform_list

    z_score_y_bool, structured_y = z_score_parser(z_score_y)
    if z_score_y_bool:
        # Prepend standardizing transform to y-embedding.
        embedding_net = nn.Sequential(
            standardizing_net(batch_y, structured_y), embedding_net
        )

    distribution = distributions_.StandardNormal((x_numel,))

    # Combine transforms.
    transform = transforms.CompositeTransform(transform_list)
    neural_net = flows.Flow(transform, distribution, embedding_net)

    return neural_net


class EmbeddingNet(nn.Module):
    def __init__(self, dim):
        super(EmbeddingNet, self).__init__()
        self.context_dim = dim
        self.dense = nn.Sequential(
            nn.Linear(dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
        )


    def forward(self, x):
        '''
        Assumes context x is of shape (batch_size, self.context_dim)
        '''
        return self.dense(x)

def generate_data(prior, simulator, n_pts, return_theta=False):
    theta = prior(num_samples=n_pts)
    x = simulator(theta)

    if return_theta: 
        return theta, x
    else:
        return x

def ci_len(encoder, q_hat, theta_grid, test_X_grid, test_sims, discretization):
    grid_scores = 1 / encoder.log_prob(theta_grid, test_X_grid).detach().cpu().exp().numpy()
    grid_scores = grid_scores.reshape(test_sims, -1) # reshape back to 2D grid per-trial

    # hacky solution to vectorize this computation, but hey, I like it
    confidence_mask = np.zeros(grid_scores.shape)
    confidence_mask[grid_scores < q_hat] = discretization
    interval_lengths = np.sum(confidence_mask, axis=1)
    return np.mean(interval_lengths)

class ConditionalCouplingLayer(nn.Module):
    def __init__(self, c_dim, x_dim, hidden_dim=128, mask_type='alternate'):
        super().__init__()
        self.c_dim = c_dim
        
        # Create binary mask for splitting channels
        if mask_type == 'alternate':
            self.mask = torch.arange(c_dim) % 2
        else:  # 'half'
            self.mask = torch.cat([torch.ones(c_dim//2), torch.zeros(c_dim - c_dim//2)])
        self.mask = self.mask.bool()
        
        # Scale and translation networks
        # Input: concatenation of x and masked c
        input_dim = x_dim + (c_dim - self.mask.sum().item())
        self.scale_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.mask.sum().item()),
            nn.Tanh()  # bound the scale
        )
        
        self.translation_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.mask.sum().item())
        )
        
    def forward(self, c, x, log_det=None, reverse=False):
        # Split into unchanged and changed parts using mask
        c1 = c[:, self.mask]
        c2 = c[:, ~self.mask]
        
        # Concatenate x with unchanged part for conditioning
        h = torch.cat([x, c2], dim=1)
        
        # Calculate scale and translation factors
        s = self.scale_net(h)  # Scale is bounded by tanh
        t = self.translation_net(h)
        
        if not reverse:
            # Forward transformation
            c1_transformed = c1 * torch.exp(s) + t
            if log_det is not None:
                log_det = log_det + s.sum(dim=1)
        else:
            # Inverse transformation
            c1_transformed = (c1 - t) * torch.exp(-s)
            if log_det is not None:
                log_det = log_det - s.sum(dim=1)
        
        # Merge back transformed and unchanged parts
        c_new = c.clone()
        c_new[:, self.mask] = c1_transformed
        c_new[:, ~self.mask] = c2
        
        if log_det is not None:
            return c_new, log_det
        return c_new

class ConditionalFlow(nn.Module):
    def __init__(self, c_dim, x_dim, n_layers=4, hidden_dim=128):
        super().__init__()
        self.c_dim = c_dim
        self.x_dim = x_dim
        
        # Stack of coupling layers
        self.layers = nn.ModuleList([
            ConditionalCouplingLayer(
                c_dim=c_dim,
                x_dim=x_dim,
                hidden_dim=hidden_dim,
                mask_type='alternate' if i % 2 == 0 else 'half'
            ) for i in range(n_layers)
        ])
        
    def forward(self, c, x, reverse=False):
        log_det = torch.zeros(c.size(0)).to(c.device)
        
        if not reverse:
            # Forward transformation: data space to latent space
            for layer in self.layers:
                c, log_det = layer(c, x, log_det)
        else:
            # Inverse transformation: latent space to data space
            for layer in reversed(self.layers):
                c, log_det = layer(c, x, log_det, reverse=True)
                
        return c, log_det
    
    def sample(self, num_samples, x, temperature=1.0):
        """Generate samples given x by sampling from base distribution and transforming"""
        #self.eval()
        batch_size = x.size(0)
        
        #with torch.no_grad():
        # Expand x to match number of samples: [batch_size, num_samples, x_dim]
        x_expanded = x.unsqueeze(1).expand(-1, num_samples, -1)
        # Reshape to [batch_size * num_samples, x_dim] for processing
        x_reshaped = x_expanded.reshape(-1, self.x_dim)
        
        # Sample from base distribution: [batch_size, num_samples, c_dim]
        z = torch.randn(batch_size, num_samples, self.c_dim).to(x.device) * temperature
        # Reshape to [batch_size * num_samples, c_dim] for processing
        z_reshaped = z.reshape(-1, self.c_dim)
        
        # Transform to data space
        samples, _ = self.forward(z_reshaped, x_reshaped, reverse=True)
        
        # Reshape back to [batch_size, num_samples, c_dim]
        samples = samples.reshape(batch_size, num_samples, self.c_dim)
            
        return samples


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--task")
    # parser.add_argument("--cuda_idx")
    # args = parser.parse_args()

    # task = sbibm.get_task('two_moons')
    # prior = task.get_prior()
    # simulator = task.get_simulator()

    # proj_dim = 2 # to consider a projected, lower-dimensional version of the problem
    # setup_theta, setup_x = generate_data(prior, simulator, 100, return_theta=True) 
    # setup_theta = setup_theta[:,:proj_dim]

    # mb_size = 50
    # device = f"cuda:0"

    # # EXAMPLE BATCH FOR SHAPES
    # z_dim = setup_theta.shape[-1]
    # x_dim = setup_x.shape[-1]
    # num_obs_flow = mb_size
    # fake_zs = torch.randn((mb_size, z_dim))
    # fake_xs = torch.randn((mb_size, x_dim))
    # encoder = build_nsf(fake_zs, fake_xs, z_score_x='none', z_score_y='none')

    # encoder.to(device)
    # optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)
    
    # save_iterate = 1_000
    # for j in range(5_001):
    #     theta, x = generate_data(prior, simulator, mb_size, return_theta=True)
    #     theta = theta[:,:proj_dim]
    #     optimizer.zero_grad()
    #     loss = -1 * encoder.log_prob(theta.to(device), x.to(device)).mean()
    #     loss.backward()
    #     optimizer.step()


    #     if j % save_iterate == 0:    
    #         cached_fn = os.path.join("trained", f"{args.task}.nf")
    #         # cached_fn = os.path.join("projected_results", f"{args.task}.nf")
    #         with open(cached_fn, "wb") as f:
    #             pickle.dump(encoder, f)
    
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    x_train, x_test = np.load('./data/x_train_energy.npy'), np.load('./data/x_test_energy.npy')
    c_train, c_test = np.load('./data/c_train_energy.npy'), np.load('./data/c_test_energy.npy')
    # load plo_t, psch_t, pup_t from npy
    plo_t, psch_t, pup_t = np.load('./data/plo_t_energy.npy'), np.load('./data/psch_t_energy.npy'), np.load('./data/pup_t_energy.npy')
    x = np.concatenate((x_train, x_test), axis=0)
    c = np.concatenate((c_train, c_test), axis=0)    
    
    mb_size = 50
    fake_zs = torch.randn((mb_size, c.shape[1]))
    fake_xs = torch.randn((mb_size, x.shape[1]))
    encoder = build_nsf(fake_zs, fake_xs, z_score_x='none', z_score_y='none')    
    
    flow = ConditionalFlow(c.shape[1], x.shape[1])
    
    dataset = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(c).float())
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    for x_t, c_t in loader:
        
        c_samples = encoder.sample(100, x_t)
        c_flow_samples = flow.sample(100, x_t)
        print(c_samples.shape)
        print(c_flow_samples.shape)
        break
