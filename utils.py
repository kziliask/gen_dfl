import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pyepo
from pyepo.model.grb import optGrbModel
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import TensorDataset
import torch.distributions as dist
from typing import Dict, List, Optional, Union


from pyepo import EPO

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

class ContextualTrainer:
    def __init__(
        self,
        x_dim: int,
        c_dim: int,
        hidden_dim: int = 64,
        dist_type: str = "normal",
        lr: float = 1e-3,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.contextual = Contextual(
            x_dim=x_dim,
            c_dim=c_dim,
            hidden_dim=hidden_dim,
            dist_type=dist_type
        ).to(device)
        
        self.optimizer = optim.Adam(self.contextual.parameters(), lr=lr)
        self.history: Dict[str, List[float]] = {"loss": [], "val_loss": []}
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None
    ) -> Dict[str, float]:
        """Train for one epoch and return metrics"""
        self.contextual.train()
        total_loss = 0.0
        num_batches = 0
        
        for x, c in train_loader:
            x, c = x.to(self.device), c.to(self.device)
            
            self.optimizer.zero_grad()
            log_prob = self.contextual.log_prob(x, c)
            loss = -log_prob.mean()  # Negative log likelihood
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        metrics = {"loss": total_loss / num_batches}
        
        # Validation
        if val_loader is not None:
            self.contextual.eval()
            val_loss = 0.0
            num_val_batches = 0
            
            with torch.no_grad():
                for x, c in val_loader:
                    x, c = x.to(self.device), c.to(self.device)
                    log_prob = self.contextual.log_prob(x, c)
                    val_loss -= log_prob.mean().item()
                    num_val_batches += 1
            
            metrics["val_loss"] = val_loss / num_val_batches
        
        # Update history
        for k, v in metrics.items():
            self.history[k].append(v)
        
        return metrics
    
    def fit(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        batch_size: int = 32,
        epochs: int = 100,
        val_split: float = 0.1,
        verbose: bool = True,
        patience: int = 10
    ):
        """Train the model on (x, c) pairs"""
        # Split into train and validation
        n_samples = len(x)
        n_val = int(n_samples * val_split)
        indices = torch.randperm(n_samples)
        
        train_idx = indices[n_val:]
        val_idx = indices[:n_val]
        
        train_loader = DataLoader(
            TensorDataset(x[train_idx], c[train_idx]),
            batch_size=batch_size,
            shuffle=True
        )
        
        val_loader = DataLoader(
            TensorDataset(x[val_idx], c[val_idx]),
            batch_size=batch_size
        )
        
        # Training loop with early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            metrics = self.train_epoch(train_loader, val_loader)
            
            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}")
                for k, v in metrics.items():
                    print(f"{k}: {v:.4f}")
                print()
            
            # Early stopping
            val_loss = metrics.get('val_loss', float('inf'))
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch}")
                break
    
    def plot_history(self):
        """Plot training history"""
        plt.figure(figsize=(10, 5))
        for metric in self.history:
            plt.plot(self.history[metric], label=metric)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def get_distribution(self, x: torch.Tensor):
        """Get the learned distribution for given x"""
        self.contextual.eval()
        with torch.no_grad():
            return self.contextual.get_distribution(x.to(self.device))

# class myModel(optGrbModel):
class AverageRegretModel(pyepo.model.grb.knapsackModel):
    """
    This class is optimization model for knapsack problem

    Attributes:
        _model (GurobiPy model): Gurobi model
        weights (np.ndarray / list): Weights of items
        capacity (np.ndarray / listy): Total capacity
        items (list): List of item index
    """

    def __init__(self, weights, capacity):
        """
        Args:
            weights (np.ndarray / list): weights of items
            capacity (np.ndarray / list): total capacity
        """
        self.weights = np.array(weights)
        self.capacity = np.array(capacity)
        self.items = self.weights.shape[1]
        super().__init__(weights, capacity)

    def _getModel(self):
        """
        A method to build Gurobi model

        Returns:
            tuple: optimization model and variables
        """
        # ceate a model
        m = gp.Model("knapsack")
        # varibles
        x = m.addMVar(self.items, name="x", vtype=GRB.BINARY)
        # sense
        m.modelSense = GRB.MAXIMIZE
        # constraints
        m.addConstr(self.weights @ x <= self.capacity)
        return m, x
    
    def setObj(self, c_samples):
        obj = gp.quicksum(
            (1/c_samples.shape[0]) * np.array(c_samples[i]) @ self.x
            for i in range(c_samples.shape[0])
        )
        self._model.setObjective(obj)
        
class CVaRRegretModel(pyepo.model.grb.knapsackModel):
    def __init__(self, weights, capacity, alpha=0.95):
        """
        Args:
            weights (np.ndarray / list): weights of items
            capacity (np.ndarray / list): total capacity
            alpha (float): confidence level for CVaR
        """
        self.alpha = alpha
        self.weights = np.array(weights)
        self.capacity = np.array(capacity)
        self.items = self.weights.shape[1]
        super().__init__(weights, capacity)

    def _getModel(self):
        """
        A method to build Gurobi model

        Returns:
            tuple: optimization model and variables
        """
        # create a model
        m = gp.Model("knapsack")
        # variables
        x = m.addMVar(self.items, name="x", vtype=GRB.BINARY)
        # sense
        m.modelSense = GRB.MAXIMIZE
        # constraints
        m.addConstr(self.weights @ x <= self.capacity)
        return m, x

    def setObj(self, c_samples):
        m = self._model
        x = self.x
        
        # Add auxiliary variables
        t = m.addVar(name="t")
        z = m.addMVar(c_samples.shape[0], name="z")
        
        # Set objective
        m.setObjective(t - (1 / (1 - self.alpha)) * gp.quicksum(z) / c_samples.shape[0], GRB.MAXIMIZE)
        
        # Add CVaR constraints
        for i in range(c_samples.shape[0]):
            m.addConstr(z[i] >= t - gp.quicksum(c_samples[i, j] * x[j] for j in range(self.items)))
            m.addConstr(z[i] >= 0)