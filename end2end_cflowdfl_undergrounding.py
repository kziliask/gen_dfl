#!/usr/bin/env python3
"""
train_undergrounding.py

Train a decision-focused model for the "undergrounding" (knapsack) problem
using synthetic SIR + weather data. Each "city" is one data point:
  x = flattened weather time-series
  c = SAIDI (sum of infected / population)
  
We'll build a PyTorch dataset from the train file, and another from the test file,
and train a ConditionalFlow to learn p(c|x). Then we optimize w.r.t. our
ExpectedUnderGroundingModel (knapsack style) using a DFL objective.

Example usage:
    python train_undergrounding.py \
        --train_file synthetic_power_data/synthetic_data_seed42_train.pt \
        --test_file synthetic_power_data/synthetic_data_seed42_test.pt \
        --capacity 2 \
        --batch_size 2 \
        --num_epochs 30
"""

import os
import json
import argparse
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

# 1) Import your SIR data utility code
def load_synthetic_data(file_path):
    """
    Loads .pt file from synthetic_data.py.
    Returns a dict with [time_points, weather_data, sir_data, total_households].
    """
    data_dict = torch.load(file_path)
    return data_dict

def calc_SAIDI(sir_data, total_households):
    """
    sir_data: shape (T, C, 3).
    total_households: shape (C,). population of each city.
    Return SAIDI for each city => sum(Infected(t))/total_households[i].
    Output shape: (C,)
    """
    # infected = sir_data[:,:,1] => shape (T,C)
    infected_sum = sir_data[:, :, 1].sum(dim=0)    # shape(C,)
    saidi = infected_sum / total_households
    return saidi

def flatten_weather(weather_data):
    """
    Flatten entire time series for each city’s weather:
      weather_data: shape (T, C, 3).
    Return x_tensor: shape (C, T*3).
    For each city i, we gather weather_data[:, i, :].reshape(-1).
    """
    T, C, _ = weather_data.shape
    x_list = []
    for i in range(C):
        # Flatten the (T,3) slice into (T*3,)
        w_i = weather_data[:, i, :].reshape(-1)
        x_list.append(w_i)
    x_tensor = torch.stack(x_list, dim=0)  # shape(C, T*3)
    return x_tensor

def build_dataset_from_synthetic(sir_data, weather_data, total_households):
    """
    Build features (x) and labels (c) for all cities in the dataset.
      x: shape (C, T*3)
      c: shape (C, 1), each city’s SAIDI
    """
    # c => SAIDI
    saidi = calc_SAIDI(sir_data, total_households)  # shape(C,)
    c_tensor = saidi.unsqueeze(1)                   # shape(C,1)

    # x => flattened weather
    x_tensor = flatten_weather(weather_data)         # shape(C, T*3)
    return x_tensor, c_tensor


# 2) The undergrounding model (knapsack style)
import gurobipy as gp
from gurobipy import GRB

class ExpectedUnderGroundingModel(nn.Module):
    """
    A 'knapsack-style' model for undergrounding/hardening a set of cities.

    Attributes:
        customer (np.ndarray): For each city, e.g. # of households
        capacity (float or int): max # of cities to select
        items (int): number of cities
        _model, self.x: the Gurobi model and its decision variables
    """
    def __init__(self, customer, capacity):
        super().__init__()
        self.customer = np.array(customer, dtype=float)  # shape (num_cities,)
        self.capacity = capacity
        self.items = len(self.customer)

        # Build the Gurobi model once
        m = gp.Model("undergrounding")
        m.Params.OutputFlag = 0  # silence Gurobi

        # x: binary or continuous choice?
        x = m.addMVar(self.items, name="x", ub=1.0, vtype=GRB.BINARY)
        m.modelSense = GRB.MAXIMIZE
        m.addConstr(x.sum() <= self.capacity)

        self._model = m
        self.x = x

    def setObj(self, c_samples):
        """
        Sets Gurobi objective = average of (c[i] * self.customer[i]) dot x over scenarios in c_samples.
        
        c_samples: shape (num_scenarios, num_items).
        We interpret c[i] as the "benefit" or "cost" of picking city i.
        Here we multiply by self.customer[i] to scale city i’s importance.
        """
        num_scen = c_samples.shape[0]
        obj_expr = 0.0
        for i in range(num_scen):
            # scenario i => c_samples[i], shape (num_items,)
            cost_vec = c_samples[i] * self.customer  # shape (items,)
            obj_expr += (1.0 / num_scen) * (cost_vec @ self.x)
        self._model.setObjective(obj_expr, GRB.MAXIMIZE)

    def solve(self):
        self._model.optimize()
        sol = self.x.X
        return sol, self._model.ObjVal

    @torch.no_grad()
    def regret_loss_batch(self, x_batch, c_samples, c_trues, alpha=1.0):
        """
        Evaluate "regret" = difference in the alpha-CVaR objective when using the solution
        from c_samples vs. the solution from c_trues.

        x_batch is unused here, but kept for API consistency.
        c_samples, c_trues: shape (batch_size, num_scenarios, num_items).
        alpha in (0,1].
        Returns average regret across batch.
        """
        batch_size = x_batch.shape[0]
        cust_t = torch.from_numpy(self.customer).float().to(c_samples.device)

        total_regret = 0.0
        for b in range(batch_size):
            c_samp = c_samples[b]  # shape(num_scenarios, num_items)
            c_true = c_trues[b]

            # Solve with c_samp
            self.setObj(c_samp.detach().cpu().numpy())
            sol_samp, _ = self.solve()
            sol_samp_t = torch.tensor(sol_samp, dtype=c_samp.dtype, device=c_samp.device)

            # Evaluate solution under c_true => get alpha-CVaR
            # For each scenario s => c_true[s] * self.customer => dot sol_samp
            objs_samp = torch.matmul(c_true * cust_t, sol_samp_t)  # shape(num_scenarios,)
            objs_samp_sorted, _ = torch.sort(objs_samp)
            m = int(alpha * len(objs_samp))
            cvar_samp = torch.mean(objs_samp_sorted[:m])  # lower-tail or upper-tail? 
            # If you want lower-tail risk measure, you might pick the bottom fraction,
            # or if you want upper-tail measure, you'd pick the top fraction. 
            # You can adapt as needed.

            # Solve with c_true
            self.setObj(c_true.detach().cpu().numpy())
            sol_true, _ = self.solve()
            sol_true_t = torch.tensor(sol_true, dtype=c_samp.dtype, device=c_samp.device)

            objs_true = torch.matmul(c_true * cust_t, sol_true_t)
            objs_true_sorted, _ = torch.sort(objs_true)
            cvar_true = torch.mean(objs_true_sorted[:m])

            regret = torch.abs(cvar_samp - cvar_true)
            total_regret += regret

        return total_regret / batch_size

    @torch.no_grad()
    def obj_eval(self, x_batch, c_samples, c_trues, alpha=1.0):
        """
        Evaluate the alpha-CVaR objective from solutions that are *fit* to c_samples,
        but measured under c_trues.

        Returns the *averaged* alpha-CVaR across the batch.
        """
        batch_size = x_batch.shape[0]
        cust_t = torch.from_numpy(self.customer).float().to(c_samples.device)

        total_obj = 0.0
        for b in range(batch_size):
            c_samp = c_samples[b]
            c_true = c_trues[b]

            # Solve with c_samp
            self.setObj(c_samp.detach().cpu().numpy())
            sol_samp, _ = self.solve()
            sol_samp_t = torch.tensor(sol_samp, dtype=c_samp.dtype, device=c_samp.device)

            # Evaluate under c_true => alpha-CVaR
            objs_samp = torch.matmul(c_true * cust_t, sol_samp_t)
            objs_samp_sorted, _ = torch.sort(objs_samp)
            m = int(alpha * len(objs_samp))
            cvar_samp = torch.mean(objs_samp_sorted[:m])
            total_obj += cvar_samp

        return total_obj / batch_size


# 3) A small dataset class for (x, c) used by the decision model
class UndergroundingDataset(Dataset):
    """
    Each example is:
      x[i,:] => weather features (flattened)
      c[i,:] => SAIDI label
    """
    def __init__(self, x_tensor, c_tensor):
        """
        x_tensor: shape (N, x_dim)
        c_tensor: shape (N, c_dim=1)
        """
        super().__init__()
        self.x = x_tensor
        self.c = c_tensor

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.c[idx]


# 4) A simple conditional normalizing flow or any parametric model
#    For brevity, here's a toy "ConditionalFlow" placeholder
#    You can replace with your actual normalizing flow code.
class ConditionalFlow(nn.Module):
    def __init__(self, c_dim, x_dim):
        """
        c_dim: dimension of label c (here, 1)
        x_dim: dimension of features x
        This model learns p(c | x).
        """
        super().__init__()
        hidden = 32
        self.net = nn.Sequential(
            nn.Linear(x_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2)  # We'll interpret output as (mean, log-variance) for c
        )

    def forward(self, c, x):
        """
        Forward pass: we interpret c as [batch_size, 1],
        x as [batch_size, x_dim].
        We'll produce "z = (c - mu)/sigma" + log_det, for NLL training.
        """
        out = self.net(x)
        mu = out[:, 0:1]
        log_var = out[:, 1:2]
        sigma = torch.exp(0.5*log_var)

        z = (c - mu) / (sigma + 1e-8)      # shape (B,1)
        # log_det = - log(sigma). sum across dimension => shape (B,)
        log_det = -0.5 * log_var.squeeze(-1)  # shape(B,)
        return z, log_det

    def sample(self, num_samples, x):
        """
        Sample c from the learned distribution p(c|x).
        x: shape (B, x_dim).
        We'll replicate x for each sample? Usually you want 1 example at a time.
        For simplicity, assume B=1 if you do large sample. Or handle broadcast.
        """
        with torch.no_grad():
            out = self.net(x)  # shape (B,2)
            mu = out[:,0:1]
            log_var = out[:,1:2]
            sigma = torch.exp(0.5*log_var)

            # We'll draw z from N(0,1)
            # If B>1, we replicate each row for num_samples, etc.
            B = x.shape[0]
            z = torch.randn(B, num_samples, 1, device=x.device)
            # c = mu + sigma * z
            # But we must broadcast carefully
            mu_expand     = mu.unsqueeze(1)     # (B,1,1)
            sigma_expand  = sigma.unsqueeze(1)  # (B,1,1)
            c = mu_expand + sigma_expand * z     # shape (B, num_samples, 1)
        return c.squeeze(-1)  # shape (B, num_samples)

# 5) Decision-focused losses
#    We'll define a simple "contrastive" style objective with your decision model:

class ContrastiveMAPLoss(nn.Module):
    def __init__(self, optmodel, alpha=1.0):
        """
        alpha: for alpha-CVaR portion
        optmodel: the undergrounding model
        """
        super().__init__()
        self.optmodel = optmodel
        self.alpha = alpha

    def forward(self, c_pred_samples, c_true_samples):
        """
        c_pred_samples: shape (B, num_samples, num_items) or (B, num_samples) if 1D label
        c_true_samples: shape (B, num_samples, num_items) or (B, num_samples) if 1D label
        We compute the regret or difference in alpha-CVaR.
        """
        # Evaluate regret in a batch
        # We'll rely on the model's built-in regret computations:
        regret = self.optmodel.regret_loss_batch(
            x_batch=torch.zeros_like(c_pred_samples),  # x is not used inside the method
            c_samples=c_pred_samples,
            c_trues=c_true_samples,
            alpha=self.alpha
        )
        return regret

# 6) Putting it all together in the training loop
def pretrain_flow(flow_model, loader, num_epochs=10, lr=1e-3, device="cpu"):
    """
    Pretrain the flow model (negative log-likelihood) ignoring the decision model.
    """
    optimizer = torch.optim.Adam(flow_model.parameters(), lr=lr)
    flow_model.to(device)

    nll_losses = []
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        n_batches = 0
        for x_batch, c_batch in loader:
            x_batch = x_batch.to(device)
            c_batch = c_batch.to(device)
            z, log_det = flow_model(c_batch, x_batch)
            # log_prob ~ -0.5 * z^2 - 0.5 log(2π)  (per dimension)
            log_prob = -0.5*torch.sum(z**2, dim=1) - 0.5*z.size(1)*np.log(2*np.pi)
            loss = -(log_prob + log_det).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
        avg_loss = epoch_loss / max(n_batches,1)
        nll_losses.append(avg_loss)
        print(f"[Pretrain] Epoch {epoch+1}/{num_epochs}, NLL: {avg_loss:.4f}")
    return nll_losses

def train_dfl(flow_model, loader, contextual_model, optmodel, alpha=1.0, beta=10.0, num_epochs=10, lr=1e-3, device="cpu"):
    """
    Jointly train with DFL. We combine negative log-likelihood + regret.
    flow_model: p(c|x)
    contextual_model: can be a second flow or the same model if you want c_true...
                     but typically you'd have 'ground truth' or you sample again.
                     For synthetic experiments, we might treat 'contextual_model'
                     as the *true* distribution. Or you can just use c_batch itself.
    alpha: for alpha-CVaR
    beta: weighting on the regret term
    """
    optimizer = torch.optim.Adam(flow_model.parameters(), lr=lr)
    flow_model.to(device)
    dfl_losses = []
    nll_losses = []

    # We'll define a simple contrastive-style object
    criterion = ContrastiveMAPLoss(optmodel, alpha=alpha)

    for epoch in range(num_epochs):
        flow_model.train()
        total_dfl = 0.0
        total_nll = 0.0
        n_batches = 0

        for x_batch, c_batch in loader:
            x_batch = x_batch.to(device)
            c_batch = c_batch.to(device)
            B = x_batch.shape[0]

            # 1) DFL part:
            #    For each example, we sample "true" costs from the *ground-truth distribution*.
            #    In a real system, you'd only have c_batch as the single observed label.
            #    For synthetic, we might replicate c_batch or use some "contextual_model.sample(...)"
            #    If you do have a separate "true distribution" model or more samples, do that.
            #    Here, let's pretend we just replicate c_batch as the "true sample" 200 times.
            c_true_samples = c_batch.unsqueeze(1).expand(-1, 200, -1)  # (B,200,1)

            # 2) Sample from the flow as the "pred distribution"
            c_pred_samples = flow_model.sample(200, x_batch) # shape(B,200)
            # Expand to shape (B,200,1) if needed:
            c_pred_samples = c_pred_samples.unsqueeze(-1)

            dfl_loss = criterion(c_pred_samples, c_true_samples)
            total_dfl += dfl_loss.item()

            # 3) NLL part:
            z, log_det = flow_model(c_batch, x_batch)
            log_prob = -0.5*torch.sum(z**2, dim=1) - 0.5*z.size(1)*np.log(2*np.pi)
            nll_loss = -(log_prob + log_det).mean()
            total_nll += nll_loss.item()

            loss = beta*dfl_loss + nll_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            n_batches += 1

        avg_dfl = total_dfl / max(n_batches,1)
        avg_nll = total_nll / max(n_batches,1)
        dfl_losses.append(avg_dfl)
        nll_losses.append(avg_nll)
        print(f"[DFL] Epoch {epoch+1}/{num_epochs} => DFL: {avg_dfl:.4f}, NLL: {avg_nll:.4f}")

    return dfl_losses, nll_losses

def evaluate_model(flow_model, loader, contextual_model, optmodel, device="cpu"):
    """
    Evaluate some metrics on test set, e.g. average alpha-CVaR objective & regret.
    """
    flow_model.eval()
    alpha_list = [1.0, 0.5, 0.1]
    results = {}
    with torch.no_grad():
        for alpha in alpha_list:
            all_objs = []
            all_regrets = []
            for x_batch, c_batch in loader:
                x_batch = x_batch.to(device)
                c_batch = c_batch.to(device)

                # sample from flow
                c_pred_samples = flow_model.sample(200, x_batch)  # shape(B,200)
                c_pred_samples = c_pred_samples.unsqueeze(-1)

                # "true" distribution samples
                c_true_samples = c_batch.unsqueeze(1).expand(-1, 200, -1)

                # Evaluate
                obj_val = optmodel.obj_eval(x_batch, c_pred_samples, c_true_samples, alpha=alpha)
                reg_val = optmodel.regret_loss_batch(x_batch, c_pred_samples, c_true_samples, alpha=alpha)
                all_objs.append(obj_val.item())
                all_regrets.append(reg_val.item())

            mean_obj = np.mean(all_objs)
            mean_reg = np.mean(all_regrets)
            results[f"alpha={alpha}"] = {
                "obj": mean_obj,
                "reg": mean_reg
            }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, required=True,
                        help="Path to .pt train file from synthetic_data.py")
    parser.add_argument("--test_file", type=str, required=True,
                        help="Path to .pt test file from synthetic_data.py")
    parser.add_argument("--capacity", type=int, default=1,
                        help="Knapsack capacity (# cities you can pick)")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--pretrain_epochs", type=int, default=10)
    parser.add_argument("--beta", type=float, default=10.0, help="weight of DFL term")
    parser.add_argument("--alpha", type=float, default=1.0, help="CVaR alpha")
    parser.add_argument("--outdir", type=str, default="undergrounding_results")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device("cpu")  # or "cuda"

    # -- Load train + test from synthetic_data.py
    train_data = load_synthetic_data(args.train_file)
    test_data  = load_synthetic_data(args.test_file)

    sir_train = train_data["sir_data"]             # shape(T, C, 3)
    weather_train = train_data["weather_data"]     # shape(T, C, 3)
    pops_train = train_data["total_households"]    # shape(C,)

    sir_test = test_data["sir_data"]
    weather_test = test_data["weather_data"]
    pops_test = test_data["total_households"]

    # Build (x_train, c_train), (x_test, c_test)
    x_train, c_train = build_dataset_from_synthetic(sir_train, weather_train, pops_train)
    x_test,  c_test  = build_dataset_from_synthetic(sir_test,  weather_test,  pops_test)

    print(f"Train set: {x_train.shape[0]} cities, each x dim={x_train.shape[1]}, c dim={c_train.shape[1]}")
    print(f"Test  set: {x_test.shape[0]} cities, each x dim={x_test.shape[1]}, c dim={c_test.shape[1]}")

    # Make PyTorch datasets
    train_dataset = UndergroundingDataset(x_train.float(), c_train.float())
    test_dataset  = UndergroundingDataset(x_test.float(),  c_test.float())

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=1,               shuffle=False)

    # Create the undergrounding model
    # We'll interpret "customer = pops_train" for each city
    # capacity = args.capacity
    # Even if test set has a different number of cities, for the model to be consistent,
    # we usually fix the dimension. So let's assume we train with # of cities in the train set,
    # but you might unify them if they differ. Typically they'd be the same # of cities or more.
    capacity = args.capacity
    print("Creating ExpectedUnderGroundingModel with capacity =", capacity)
    ug_model = ExpectedUnderGroundingModel(customer=pops_train.numpy(), capacity=capacity)

    # Define the flow model
    c_dim = 1   # SAIDI is 1D
    x_dim = x_train.shape[1]
    flow_model = ConditionalFlow(c_dim, x_dim)

    # (Optional) Pretrain with MLE
    print("=== Pretraining Flow ===")
    pretrain_flow(flow_model, train_loader, num_epochs=args.pretrain_epochs, lr=1e-3, device=device)

    # If you had a "contextual_model" to represent the true distribution, you could define it here.
    # In many synthetic experiments, we just use c_batch as the truth. So let's pass None or re-use the same flow.
    contextual_model = None  # placeholder

    # Train with Decision-Focused Learning
    print("=== DFL Training ===")
    dfl_losses, nll_losses = train_dfl(flow_model, train_loader,
                                       contextual_model, ug_model,
                                       alpha=args.alpha, beta=args.beta,
                                       num_epochs=args.num_epochs, lr=1e-3,
                                       device=device)

    # Plot the training losses
    plt.figure(figsize=(8,6))
    plt.plot(dfl_losses, label="DFL Loss")
    plt.plot(nll_losses, label="NLL Loss")
    plt.legend()
    plt.title("Training Losses")
    plt.xlabel("Epoch")
    plt.savefig(os.path.join(args.outdir, "training_losses.png"), dpi=150)
    plt.close()

    # Evaluate
    print("=== Evaluating on Test Set ===")
    metrics = evaluate_model(flow_model, test_loader, contextual_model, ug_model, device=device)
    print("Evaluation metrics:")
    for k,v in metrics.items():
        print(k, "=", v)

    # Save final results
    out_json = {
        "capacity": args.capacity,
        "alpha": args.alpha,
        "beta": args.beta,
        "metrics": metrics,
        "DFL_losses": dfl_losses,
        "NLL_losses": nll_losses
    }
    with open(os.path.join(args.outdir, "final_results.json"), "w") as f:
        json.dump(out_json, f, indent=2)

    # Optionally save the model
    torch.save(flow_model.state_dict(), os.path.join(args.outdir, "flow_model.pth"))

    print("Done.")

if __name__ == "__main__":
    main()
