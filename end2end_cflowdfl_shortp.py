import torch
import numpy as np
import pyepo
from pyepo.model.grb import optGrbModel
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from pyepo import EPO
from src.generators.cnf import ConditionalFlow
import json
import os
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional, Union

from optModel import ExpectedShortestPathModel
from optDataset import optDataset
from torch.utils.data import TensorDataset
import torch.distributions as dist

from func.contrastive import NCE, contrastiveMAP

# include argparse
import argparse

class Contextual_wrapper():
    def __init__(self, task):
        self.task = task
    
    def sample(self, num_samples, x):
        return self.task._sample_reference_posterior(num_samples, num_observation=x.shape[0], observation=x)
    
class Contextual:
    """
    A class representing conditional probability distributions p(c|x).
    Supports various distribution types and multi-dimensional outputs.
    """
    def __init__(
        self,
        x_dim: int,
        c_dim: int,
        hidden_dim: int = 64,
        dist_type: str = "normal",
        num_layers: int = 2
    ):
        """
        Args:
            x_dim: Dimension of input features x
            c_dim: Dimension of output variable c
            hidden_dim: Hidden dimension for parameter network
            dist_type: Type of distribution ("normal", "beta", "categorical")
            num_layers: Number of hidden layers in parameter network
        """
        self.x_dim = x_dim
        self.c_dim = c_dim
        self.dist_type = dist_type
        
        # Determine number of parameters needed for each distribution type
        params_per_dim = {
            "normal": 2,  # mean, log_scale
            "beta": 2,    # alpha, beta
            "categorical": 1  # logits
        }
        self.num_params = params_per_dim[dist_type] * c_dim
        
        # Build parameter network
        layers = []
        prev_dim = x_dim
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                # nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, self.num_params))
        
        self.param_net = nn.Sequential(*layers)
    
    def get_distribution(
        self,
        x: torch.Tensor,
        temp: float = 1.0
    ) -> Union[dist.Distribution, dist.Independent]:
        """
        Get the conditional distribution p(c|x) for given x.
        
        Args:
            x: Input tensor of shape (batch_size, x_dim)
            temp: Temperature parameter for sampling
            
        Returns:
            PyTorch distribution object
        """
        
        params = self.param_net(x)
        batch_size = x.shape[0]
        
        if self.dist_type == "normal":
            # Split params into mean and scale for each dimension
            means = params[:, :self.c_dim]
            log_scales = params[:, self.c_dim:]
            scales = torch.exp(log_scales) * temp
            
            # Create multi-dimensional normal distribution
            return dist.Independent(
                dist.Normal(means, scales),
                reinterpreted_batch_ndims=1
            )
            
        elif self.dist_type == "beta":
            # Split params into alpha and beta
            alpha = torch.exp(params[:, :self.c_dim])
            beta = torch.exp(params[:, self.c_dim:])
            
            return dist.Independent(
                dist.Beta(alpha, beta),
                reinterpreted_batch_ndims=1
            )
            
        elif self.dist_type == "categorical":
            # Reshape logits for categorical distribution
            logits = params.view(batch_size, self.c_dim, -1)
            return dist.Independent(
                dist.Categorical(logits=logits / temp),
                reinterpreted_batch_ndims=1
            )
    
    def sample(
        self,
        num_samples: int,
        x: torch.Tensor,
        temp: float = 1.0
    ) -> torch.Tensor:
        """Sample from p(c|x) with specified number of samples per x

        Args:
            x: Input tensor of shape [batch_size, x_dim]
            num_samples: Number of samples to draw for each x
            temp: Temperature parameter for sampling
            
        Returns:
            Samples of shape [num_samples, batch_size, c_dim]
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        distribution = self.get_distribution(x, temp)
        samples = distribution.sample((num_samples,))
        return samples.permute(1, 0, 2)
    
    def log_prob(
        self,
        x: torch.Tensor,
        c: torch.Tensor
    ) -> torch.Tensor:
        """Compute log p(c|x)"""
        distribution = self.get_distribution(x)
        return distribution.log_prob(c)
    

def generate_data(n, p, deg, e, grid):
    '''
    n: number of samples
    p: number of features
    deg: degree of the polynomial
    grid: grid size
    e: edge density
    '''
    x,c = pyepo.data.shortestpath.genData(n, p, grid, deg, e, seed=42)
    contextual = ConditionalFlow(c.shape[1], x.shape[1])
    optimizer = torch.optim.Adam(contextual.parameters(), lr=0.001)
    
    dataset = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(c).float())
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    losses = []
    for epoch in tqdm(range(30)):
        for x_t, c_t in loader:
            optimizer.zero_grad()
            z, log_det = contextual(c_t, x_t)
            
            # Compute negative log likelihood
            # Prior is standard normal, so log p(z) = -0.5 * z^2 - 0.5 * log(2π)
            log_prob = -0.5 * torch.sum(z**2, dim=1) - 0.5 * z.size(1) * np.log(2 * np.pi)
            
            # Total loss is negative log likelihood minus log determinant
            loss = -(log_prob + log_det).mean()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
    
    # Plot loss curve
    plt.rcParams['text.usetex'] = False 
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Contextual Loss Curve')
    plt.legend()
    os.makedirs("eval/shortp", exist_ok=True)
    plt.savefig('eval/shortp/contextual_loss_curve_shortp_cflow.png')
    plt.close()
    
    
    return x, c, contextual

def create_datasets(x, c, grid, batch_size, contextual):
    x_train, x_test, c_train, c_test = train_test_split(x, c, test_size=int(x.shape[0]*0.2), random_state=246)
    
    optmodel = ExpectedShortestPathModel(grid)
    dataset_train = optDataset(optmodel, x_train, c_train, contextual)
    dataset_test = optDataset(optmodel, x_test, c_test, contextual)
    
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False)
    
    return loader_train, loader_test, optmodel

def train_model(gen_model, contextual, optmodel, optimizer, loader_train, num_epochs, batch_size, device, alpha=0.5, beta=10):
    
    dfl_losses = []
    nll_losses = []
    
    criterion = contrastiveMAP(optmodel=optmodel, processes=1, solve_ratio=1, reduction="mean", dataset=loader_train.dataset)
    
    # Early stopping parameters
    patience = 5
    best_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    # Initialize lists to track validation metrics
    val_dfl_losses = []
    val_nll_losses = []
    
    # Create validation loader with 10% of training data
    train_size = int(0.9 * len(loader_train.dataset))
    val_size = len(loader_train.dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(loader_train.dataset, [train_size, val_size])
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    for epoch in range(num_epochs):
        # Training loop
        for data in tqdm(loader_train, desc=f"Epoch {epoch+1}/{num_epochs}"):
            x, c, w, z = data
            x, c, w, z = x.to(device), c.to(device), w.to(device), z.to(device)
            
            dfl_loss = 0
            for i in range(x.shape[0]):
                c_trues = contextual.sample(200, x[i].unsqueeze(0))
                optmodel.setObj(c_trues.squeeze(0).detach())
                sol_true, _ = optmodel.solve()
                
                c_gens = gen_model.sample(200, x[i].unsqueeze(0))
                
                dfl_loss += criterion(c_gens, torch.tensor(sol_true).float(), alpha=alpha)
            
            dfl_loss = dfl_loss / x.shape[0]
            # Forward pass: transform to latent space
            z, log_det = gen_model(c, x)
            # Compute negative log likelihood
            # Prior is standard normal, so log p(z) = -0.5 * z^2 - 0.5 * log(2π)
            log_prob = -0.5 * torch.sum(z**2, dim=1) - 0.5 * z.size(1) * np.log(2 * np.pi)
            # Total loss is negative log likelihood minus log determinant
            nll_loss = -(log_prob + log_det).mean()
            loss = dfl_loss * beta + nll_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            dfl_losses.append(dfl_loss.item())
            nll_losses.append(nll_loss.item())
        
        print(f"DFL Loss on Training Set at epoch {epoch}: {dfl_loss.item()}")
        print(f"NLL Loss on Training Set at epoch {epoch}: {nll_loss.item()}")

        # Validation loop
        val_dfl_loss = 0
        val_nll_loss = 0
        gen_model.eval()
        with torch.no_grad():
            for data in val_loader:
                x, c, w, z = data
                x, c, w, z = x.to(device), c.to(device), w.to(device), z.to(device)
                
                val_dfl = 0
                for i in range(x.shape[0]):
                    c_trues = contextual.sample(200, x[i].unsqueeze(0))
                    optmodel.setObj(c_trues.squeeze(0).detach())
                    sol_true, _ = optmodel.solve()
                    
                    c_gens = gen_model.sample(200, x[i].unsqueeze(0))
                    
                    val_dfl += criterion(c_gens, torch.tensor(sol_true).float(), alpha=alpha)
                
                val_dfl = val_dfl / x.shape[0]
                # Forward pass: transform to latent space
                z, log_det = gen_model(c, x)
                # Compute negative log likelihood
                # Prior is standard normal, so log p(z) = -0.5 * z^2 - 0.5 * log(2π)
                log_prob = -0.5 * torch.sum(z**2, dim=1) - 0.5 * z.size(1) * np.log(2 * np.pi)
                # Total loss is negative log likelihood minus log determinant
                val_nll = -(log_prob + log_det).mean()
            
                val_dfl_loss += val_dfl.item()
                val_nll_loss += val_nll.item()
        
        gen_model.train()
        val_dfl_loss /= len(val_loader)
        val_nll_loss /= len(val_loader)
        val_total_loss = val_dfl_loss * beta + val_nll_loss
        
        val_dfl_losses.append(val_dfl_loss)
        val_nll_losses.append(val_nll_loss)
        
        print(f"Validation DFL Loss at epoch {epoch}: {val_dfl_loss}")
        print(f"Validation NLL Loss at epoch {epoch}: {val_nll_loss}")

        # Early stopping check
        if val_dfl_loss < best_loss:
            best_loss = val_dfl_loss
            patience_counter = 0
            best_model_state = gen_model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                gen_model.load_state_dict(best_model_state)
                break
    return gen_model, dfl_losses, nll_losses

def evaluate_model(gen_model, contextual, optmodel, loader_test, batch_size, device, alpha=1):
    
    average_objectives = []
    average_regrets = []
    cvar_objectives = []
    cvar_regrets = []
    cvar_01_objectives = []
    cvar_01_regrets = []
    
    for data in tqdm(loader_test, desc="Evaluating"):
        x, c, w, z = data
        x, c, w, z = x.to(device), c.to(device), w.to(device), z.to(device)
        
        cp_samples = gen_model.sample(200, x).detach()
        # cp_samples = cp_samples.view(1 * 200, -1)
        c_trues = contextual.sample(200, x).detach()
        
        # Average Objective
        obj = optmodel.obj_eval(x, cp_samples, c_trues, alpha=1)
        average_objectives.append(obj)# / (z.abs().sum().item() + 1e-7))
        
        # Average Regret
        regret = optmodel.regret_loss_batch(x, cp_samples, c_trues, alpha=1)
        average_regrets.append(regret / (z.abs().sum().item() + 1e-7))
        
        # CVaR Objective
        obj = optmodel.obj_eval(x, cp_samples, c_trues, alpha=0.5)
        cvar_objectives.append(obj)# / (z.abs().sum().item() + 1e-7))
        
        # CVaR Regret
        regret = optmodel.regret_loss_batch(x, cp_samples, c_trues, alpha=0.5)
        cvar_regrets.append(regret / (z.abs().sum().item() + 1e-7))

        # CVaR 0.1 objective
        obj = optmodel.obj_eval(x, cp_samples, c_trues, alpha=0.1)
        cvar_01_objectives.append(obj)# / (z.abs().sum().item() + 1e-7))
        
        # CVaR 0.1 regret
        regret = optmodel.regret_loss_batch(x, cp_samples, c_trues, alpha=0.1)
        cvar_01_regrets.append(regret / (z.abs().sum().item() + 1e-7))
    
    print(f"Average Objective: {np.mean(average_objectives)}")
    print(f"Average Regret: {np.mean(average_regrets)}")
    print(f"CVaR Objective: {np.mean(cvar_objectives)}")
    print(f"CVaR Regret: {np.mean(cvar_regrets)}")
    print(f"CVaR 0.1 Objective: {np.mean(cvar_01_objectives)}")
    print(f"CVaR 0.1 Regret: {np.mean(cvar_01_regrets)}")
    return np.mean(average_objectives), np.mean(average_regrets), np.mean(cvar_objectives), np.mean(cvar_regrets), np.mean(cvar_01_objectives), np.mean(cvar_01_regrets)

def pretrain_model(gen_model, optimizer, loader_train, num_epochs, device):
    nll_losses = []
    
    for epoch in range(num_epochs):
        epoch_losses = []
        for data in tqdm(loader_train, desc=f"Pretraining Epoch {epoch+1}/{num_epochs}"):
            x, c, w, z = data
            x, c = x.to(device), c.to(device)
            
            z, log_det = gen_model(c, x)
            
            # Compute negative log likelihood
            # Prior is standard normal, so log p(z) = -0.5 * z^2 - 0.5 * log(2π)
            log_prob = -0.5 * torch.sum(z**2, dim=1) - 0.5 * z.size(1) * np.log(2 * np.pi)
            
            # Total loss is negative log likelihood minus log determinant
            nll_loss = -(log_prob + log_det).mean()
            
            optimizer.zero_grad()
            nll_loss.backward()
            optimizer.step()

            epoch_losses.append(nll_loss.item())
        
        avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
        nll_losses.append(avg_epoch_loss)
        print(f"Average NLL Loss on Training Set at epoch {epoch}: {avg_epoch_loss}")
    
    return gen_model, nll_losses

def run_experiment(n, p, deg, grid, noise_width, batch_size, num_epochs, device, beta, alpha):
    x, c, contextual = generate_data(n, p, deg, noise_width, grid)
    loader_train, loader_test, optmodel = create_datasets(x, c, grid, batch_size, contextual)
    
    gen_model = ConditionalFlow(c.shape[1], x.shape[1])
    gen_model.to(device)
    
    optimizer = torch.optim.Adam(gen_model.parameters(), lr=0.001)
    
    # Pretrain the model
    pretrain_epochs = 50  # You can adjust this value
    gen_model, pretrain_losses = pretrain_model(gen_model, optimizer, loader_train, pretrain_epochs, device)
    
    optimizer = torch.optim.Adam(gen_model.parameters(), lr=0.0001)
    trained_model, dfl_losses, nll_losses = train_model(gen_model, contextual, optmodel, optimizer, loader_train, num_epochs, batch_size, device, beta=beta, alpha=alpha)
    average_objective, average_regret, cvar_objective, cvar_regret, cvar_01_objective, cvar_01_regret = evaluate_model(trained_model, contextual, optmodel, loader_test, batch_size, device)
    
    return trained_model, average_objective, average_regret, cvar_objective, cvar_regret, cvar_01_objective, cvar_01_regret, dfl_losses, nll_losses

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--betas", nargs="+", type=float, default=[10])
    parser.add_argument("--alpha", type=float, default=1)
    parser.add_argument("--n", nargs="+", type=int, default=[400])
    parser.add_argument("--deg", nargs="+", type=int, default=[6])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--noise_width", type=float, default=0.5)
    parser.add_argument("--num_experiments", type=int, default=5)
    args = parser.parse_args()
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    n = args.n
    p = 5 # 5
    grid = (5, 5)
    noise_width = args.noise_width
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    num_experiments = args.num_experiments
    
    # betas = [0, 0.1, 1, 10, 50]
    betas = args.betas
    results = []
    
    for n in args.n:
        for deg in args.deg:
            for beta in betas:
                for exp_idx in range(num_experiments):
                    print(f"Running experiment for n={n}, beta={beta}, deg={deg}, experiment {exp_idx+1}/{num_experiments}")
                
                    model, avg_objective, avg_regret, cvar_objective, \
                        cvar_regret, cvar_01_objective, cvar_01_regret, dfl_losses, nll_losses \
                            = run_experiment(
                        n, p, deg, grid, noise_width, batch_size, num_epochs, device, beta, args.alpha
                    )
                    
                    result = {
                        "n": n,
                        "p": p,
                        "beta": beta,
                        "deg": deg,
                        "grid": grid[0],
                        "noise_width": noise_width,
                        "batch_size": batch_size,
                        "num_epochs": num_epochs,
                        "experiment_index": exp_idx,
                        "average_objective": float(avg_objective),
                        "average_regret": float(avg_regret),
                        "cvar_objective": float(cvar_objective),
                        "cvar_regret": float(cvar_regret),
                        "cvar_01_objective": float(cvar_01_objective),
                        "cvar_01_regret": float(cvar_01_regret),
                        "final_dfl_loss": float(dfl_losses[-1]),
                        "final_nll_loss": float(nll_losses[-1]),
                    }
                    results.append(result)
                    
                    # Save loss curves as numpy arrays
                    os.makedirs(f"eval/shortp/deg_{deg}_alpha_{args.alpha}/losses", exist_ok=True)
                    np.save(f"eval/shortp/deg_{deg}_alpha_{args.alpha}/losses/end2end_dfl_losses_exp_{exp_idx}_beta_{beta}_n{n}.npy", np.array(dfl_losses))
                    np.save(f"eval/shortp/deg_{deg}_alpha_{args.alpha}/losses/end2end_nll_losses_exp_{exp_idx}_beta_{beta}_n{n}.npy", np.array(nll_losses))

                    # Plot and save DFL loss curve
                    plt.rcParams['text.usetex'] = False 
                    plt.figure(figsize=(10, 5))
                    plt.plot(dfl_losses, label='DFL Loss')
                    plt.xlabel('Epoch')
                    plt.ylabel('Loss')
                    plt.title(f'DFL Loss Curve (deg={deg}, exp={exp_idx})')
                    plt.legend()
                    plt.savefig(f"eval/shortp/deg_{deg}_alpha_{args.alpha}/losses/end2end_dfl_loss_exp_{exp_idx}_beta_{beta}_n{n}.png")
                    plt.close()

                    # Plot and save NLL loss curve
                    plt.figure(figsize=(10, 5))
                    plt.plot(nll_losses, label='NLL Loss')
                    plt.xlabel('Epoch')
                    plt.ylabel('Loss')
                    plt.title(f'NLL Loss Curve (beta={beta}, exp={exp_idx})')
                    plt.legend()
                    plt.savefig(f"eval/shortp/deg_{deg}_alpha_{args.alpha}/losses/end2end_nll_loss_exp_{exp_idx}_beta_{beta}_n{n}.png")
                    plt.close()
                    
                    # Save model weights
                    os.makedirs(f"eval/shortp/deg_{deg}_alpha_{args.alpha}", exist_ok=True)
                    torch.save(model.state_dict(), f"eval/shortp/deg_{deg}_alpha_{args.alpha}/end2end_model_exp_{exp_idx}_beta_{beta}_n{n}.pth")
                    
                    # Save results
                    with open(f"eval/shortp/deg_{deg}_alpha_{args.alpha}/end2end_results_exp_{exp_idx}_beta_{beta}_n{n}.json", "w") as f:
                        json.dump(result, f, indent=2)
        
    # Save all results in a single file
    with open(f"eval/shortp/all_results_end2end_alpha_{args.alpha}_deg{deg}.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
