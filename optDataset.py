import time

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from pyepo.model.opt import optModel

import random
from scipy.spatial import distance

class optDataset(Dataset):
    """
    This class is Torch Dataset for optimization problems.

    Attributes:
        model (optModel): Optimization models
        feats (np.ndarray): Data features
        costs (np.ndarray): Cost vectors
        sols (np.ndarray): Optimal solutions
        objs (np.ndarray): Optimal objective values
    """

    def __init__(self, model, feats, costs, contextual, num_samples=200):
        """
        A method to create a optDataset from optModel

        Args:
            model (optModel): an instance of optModel
            feats (np.ndarray): data features
            costs (np.ndarray): costs of objective function
            contextual: conditional distribution p(c|x) or p(costs|feats)
        """
        if not isinstance(model, optModel):
            raise TypeError("arg model is not an optModel")
        self.model = model
        # data
        self.feats = feats
        self.costs = costs
        self.contextual = contextual
        self.num_samples = num_samples
        # find optimal solutions
        self.sols, self.objs, self.costs = self._getSols()

    def _getSols(self):
        """
        A method to get optimal solutions for all cost vectors
        """
        sols = []
        objs = []
        costs = []  # each cost is of size 
        print("Optimizing for optDataset...")
        time.sleep(1)
        for i in tqdm(range(len(self.feats))):
            # print(self.feats[i:i+1].shape)
            
            c_samples = self.contextual.sample(self.num_samples, torch.tensor(self.feats[i:i+1]).float()).detach()
            costs.append(c_samples.mean(dim=1).squeeze())
            # try:
            sol, obj = self._solve(c_samples[0])
            # except:
            #     raise ValueError(
            #         "For optModel, the method 'solve' should return solution vector and objective value."
            #     )
            sols.append(sol)
            objs.append([obj])
            # for c in c_samples:
            #     try:
            #         sol, obj = self._solve(c)
            #     except:
            #         raise ValueError(
        #             "For optModel, the method 'solve' should return solution vector and objective value."
        #         )
        #     sols.append(sol)
        #     objs.append([obj])
        costs = torch.stack(costs)
        return np.array(sols), np.array(objs), costs.numpy()

    def _solve(self, cost):
        """
        A method to solve optimization problem to get an optimal solution with given cost

        Args:
            cost (np.ndarray): cost of objective function

        Returns:
            tuple: optimal solution (np.ndarray) and objective value (float)
        """
        self.model.setObj(cost)
        sol, obj = self.model.solve()
        return sol, obj

    def __len__(self):
        """
        A method to get data size

        Returns:
            int: the number of optimization problems
        """
        return len(self.costs)

    def __getitem__(self, index):
        """
        A method to retrieve data

        Args:
            index (int): data index

        Returns:
            tuple: data features (torch.tensor), costs (torch.tensor), optimal solutions (torch.tensor) and objective values (torch.tensor)
        """
        return (
            torch.FloatTensor(self.feats[index]),
            torch.FloatTensor(self.costs[index]),
            torch.FloatTensor(self.sols[index]),
            torch.FloatTensor(self.objs[index]),
        )

def portfolio_genData(num_data, num_features, num_assets, deg=1, noise_level=1, rank: int = None, seed=135):
    """
    A function to generate synthetic data and features for travelling salesman

    Args:
        num_data (int): number of data points
        num_features (int): dimension of features
        num_assets (int): number of assets
        deg (int): data polynomial degree
        noise_level (float): level of data random noise
        seed (int): random seed

    Returns:
        tuple: data features (np.ndarray), costs (np.ndarray)
    """
    # positive integer parameter
    if type(deg) is not int:
        raise ValueError("deg = {} should be int.".format(deg))
    if deg <= 0:
        raise ValueError("deg = {} should be positive.".format(deg))
    # set seed
    rnd = np.random.RandomState(seed)
    # number of data points
    n = num_data
    # dimension of features
    p = num_features
    # number of assets
    m = num_assets
    # random matrix parameter B
    B = rnd.binomial(1, 0.5, (m, p))
    # random matrix parameter L
    L = rnd.uniform(-2.5e-3*noise_level, 2.5e-3*noise_level, (num_assets, num_features))
    if rank is not None:
        # Perform SVD decomposition
        U, s, Vh = np.linalg.svd(L, full_matrices=False)
        # Truncate to rank k by zeroing out singular values beyond k
        s_trunc = np.zeros_like(s)
        s_trunc[:rank] = s[:rank]
        # Reconstruct L with rank k
        L = U @ np.diag(s_trunc) @ Vh
    # feature vectors
    x = rnd.normal(0, 1, (n, p))
    # value of items
    r = np.zeros((n, m))
    for i in range(n):
        # mean return of assets
        r[i] = (0.05 * np.dot(B, x[i].reshape(p, 1)).T / np.sqrt(p) + \
                0.1 ** (1 / deg)) ** deg
        # random noise
        f = rnd.randn(num_features)
        eps = rnd.randn(num_assets)
        r[i] += L @ f + 0.01 * noise_level * eps
    # covariance matrix of the returns
    cov = L @ L.T + (1e-2 * noise_level) ** 2 * np.eye(num_assets)
    return cov, x, r
# for testing
class MoonSampler:
    def __init__(self, noise_std=0.1, line_centers=[-1, 1]):
        self.noise_std = noise_std
        self.line_centers = line_centers
        
    def project_to_moon(self, points, is_upper_moon):
        """
        Project points to the nearest point on the corresponding moon
        """
        t = np.linspace(0, np.pi, 1000)
        
        if is_upper_moon:
            moon_points = np.stack([np.cos(t), np.sin(t)], axis=1)
        else:
            moon_points = np.stack([1 - np.cos(t), 0.5 - np.sin(t)], axis=1)
            
        projected_points = np.zeros_like(points)
        
        for i in range(len(points)):
            distances = np.sum((moon_points - points[i])**2, axis=1)
            closest_idx = np.argmin(distances)
            projected_points[i] = moon_points[closest_idx]
            
        return projected_points
    
    def sample(self, num_samples, x_query, random_state=None):
        """
        Sample from p(c|x) for given x points, ensuring samples stay on the moon
        """
        if random_state is None:
            random_state = np.random.RandomState()
            
        batch_size = len(x_query)
        samples = np.zeros((batch_size, num_samples, 2))
        
        # Determine which line/moon based on x coordinate
        is_upper_line = (np.abs(x_query[:, 1] - self.line_centers[0]) < 
                        np.abs(x_query[:, 1] - self.line_centers[1]))
        
        for i in range(batch_size):
            # Generate random t values with some spread
            t = random_state.normal(0, self.noise_std, num_samples) + np.pi/2
            t = np.clip(t, 0, np.pi)
            
            if is_upper_line[i]:
                # Upper moon
                samples[i, :, 0] = np.cos(t)
                samples[i, :, 1] = np.sin(t)
            else:
                # Lower moon
                samples[i, :, 0] = 1 - np.cos(t)
                samples[i, :, 1] = 0.5 - np.sin(t)
            
            # Add noise
            noise = random_state.normal(0, self.noise_std, (num_samples, 2))
            noisy_samples = samples[i] + noise
            
            # Project noisy samples back to moon
            samples[i] = self.project_to_moon(noisy_samples, is_upper_line[i])
            
        return torch.tensor(samples)
    
def twomoon_data_with_sampler(num_data, num_features, num_items, noise_std=0.1, seed=135):
    """
    Generate synthetic data with deterministic x on parallel lines and returns a MoonSampler
    """
    if num_items != 2:
        raise ValueError("This generation process only works for num_items = 2")
    
    rnd = np.random.RandomState(seed)
    n = num_data
    p = num_features
    n_per_line = n // 2
    
    # Line parameters
    line_centers = [-1, 1]  # y-coordinates of the two lines
    
    # Generate x on two perfectly parallel lines
    x = np.zeros((n, p))
    
    # First line - evenly spaced points
    x[:n_per_line, 0] = np.linspace(-2, 2, n_per_line)
    x[:n_per_line, 1] = line_centers[0]
    
    # Second line - evenly spaced points
    x[n_per_line:, 0] = np.linspace(-2, 2, n - n_per_line)
    x[n_per_line:, 1] = line_centers[1]
    
    if p > 2:
        x[:, 2:] = 0
    
    # Generate deterministic moon points
    c = np.zeros((n, 2))
    t1 = np.linspace(0, np.pi, n_per_line)
    t2 = np.linspace(0, np.pi, n - n_per_line)
    
    # First moon (upper)
    c[:n_per_line, 0] = np.cos(t1)
    c[:n_per_line, 1] = np.sin(t1)
    
    # Second moon (lower)
    c[n_per_line:, 0] = 1 - np.cos(t2)
    c[n_per_line:, 1] = 0.5 - np.sin(t2)
    
    # Generate random weights for the knapsack
    weights = rnd.uniform(3, 8, (1, 2))
    
    # Create sampler instance
    sampler = MoonSampler(noise_std=noise_std, line_centers=line_centers)
    
    return weights, x, c, sampler
# for testing

if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from optModel import ExpectedKnapsackModel
    from train import build_nsf
    weights, x, c, contextual = twomoon_data_with_sampler(100, 2, 2)
    x_train, x_test, c_train, c_test = train_test_split(x, c, test_size=0.2, random_state=42)
    optmodel = ExpectedKnapsackModel(weights, 10)
    print(weights.shape, x_train.shape, c_train.shape)
    dataset_train = optDataset(optmodel, x_train, c_train, contextual)
    dataset_test = optDataset(optmodel, x_test, c_test, contextual)
    loader_train = DataLoader(dataset_train, batch_size=32, shuffle=True)
    # loader_test = DataLoader(dataset_test, batch_size=32, shuffle=False)
    mb_size = 50
    fake_zs = torch.randn((mb_size, 2))
    fake_xs = torch.randn((mb_size, 2))
    gen_model   = build_nsf(fake_zs, fake_xs, z_score_x='none', z_score_y='none').float()
    

    optimizer = torch.optim.Adam(gen_model.parameters(), lr=0.001)
    for data in loader_train:
        x, c, _, _ = data
        print(x.shape)
        # alphas = [0.1, 0.5, 0.9]
        optimizer.zero_grad()
        loss = optmodel.regret_loss(x, gen_model, alpha=0.5, num_samples=200)
        loss.backward()
        optimizer.step()
        print(f"loss: {loss}")
        break
