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
from optDataset import optDataset, portfolio_genData
from torch.utils.data import TensorDataset
import torch.distributions as dist

from func.contrastive import NCE, contrastiveMAP, NCEPred, contrastiveMAPPred
from func.spo import SPOPlus
from func.rank import listwiseLTR, pairwiseLTR, pointwiseLTR
# include argparse
import argparse

# Prediction DFL
class NN(nn.Module):

    def __init__(self, num_feat, num_item, hidden_size=16):
        super(NN, self).__init__()
        self.linear = nn.Linear(num_feat, hidden_size)
        self.linear2 = nn.Linear(hidden_size, num_item)

    def forward(self, x):
        x = torch.relu(self.linear(x))
        out = self.linear2(x)
        return out

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
    plt.savefig('eval/shortp/contextual_loss_curve_shortp_pred.png')
    plt.close()
    
    
    return x, c, contextual

def create_datasets(x, c, grid, batch_size, contextual):
    x_train, x_test, c_train, c_test = train_test_split(x, c, test_size=int(x.shape[0]*0.2), random_state=246)
    
    optmodel = ExpectedShortestPathModel(grid)
    dataset_train = optDataset(optmodel, x_train, c_train, contextual)
    dataset_test = optDataset(optmodel, x_test, c_test, contextual)
    
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False)
    
    optmodel = pyepo.model.grb.shortestPathModel(grid)

    return loader_train, loader_test, optmodel

def train_model(predmodel, criterion, contextual, optmodel, optimizer, loader_train, num_epochs, batch_size, device, loss_func):
    
    # criterion = pyepo.func.SPOPlus(optmodel, processes=1)
    
    losses = []
    for epoch in range(num_epochs):
        # Training loop
        for data in tqdm(loader_train, desc=f"Epoch {epoch+1}/{num_epochs}"):
            x, c, w, z = data
            x, c, w, z = x.to(device), c.to(device), w.to(device), z.to(device)
            
            cp = predmodel(x)
            c_trues = contextual.sample(200, x).detach().squeeze()
            c = c_trues.mean(1).squeeze()
            # print(cp.shape, c.shape, w.shape, z.shape)    [bs, d], [bs, d], [bs, d], [bs, 1]
            if loss_func == 'spo+':
                loss = criterion(cp, c, w, z)
            elif loss_func in ['imle', 'aimle']:
                # print(cp.shape, w.shape)
                loss = criterion(cp, w)
            elif loss_func in ['nce', 'map']:
                loss = criterion(cp, w)     # [bs (1), num_samples, d], [d]
            elif loss_func in ['lltr', 'pltr', 'ptltr']:
                # print(cp.shape, c.shape)  [bs, d], [bs, d]
                loss = criterion(cp, c)
            elif loss_func == 'two_stage':
                loss = criterion(cp, c)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(f"DFL Loss on Training Set at epoch {epoch}: {loss.item()}")
      
    return predmodel, losses

def evaluate_model(predmodel, contextual, optmodel, loader_test, batch_size, device):
    
    average_objectives = []
    average_regrets = []
    mse_losses = []
    
    for data in tqdm(loader_test, desc="Evaluating"):
        x, c, w, z = data
        x, c, w, z = x.to(device), c.to(device), w.to(device), z.to(device)
        
        cp = predmodel(x)
        optmodel.setObj(cp.detach())
        sol, _ = optmodel.solve()

        # mse between cp and c
        mse = (cp - c).pow(2).mean().item()
        mse_losses.append(mse)
        # Average Objective
        sol_tensor = torch.tensor(sol, dtype=cp.dtype, device=cp.device)
        # c_trues = contextual.sample(200, x).detach().squeeze()
        # obj = torch.matmul(c_trues, sol_tensor)
        obj = torch.matmul(c, sol_tensor)
        average_objectives.append(obj.mean().item())# / (z.abs().sum().item() + 1e-7))
        # Average Regret
        average_regrets.append(torch.abs(obj - z).mean().item() / (z.abs().sum().item() + 1e-7))

        
    print(f"Average Objective: {np.mean(average_objectives)}")
    print(f"Average Regret: {np.mean(average_regrets)}")
    print(f"Average MSE: {np.mean(mse_losses)}")
    print(f"Real Objective: {z.mean().item()}")
    return np.mean(average_objectives), np.mean(average_regrets), np.mean(mse_losses)

def run_experiment(n, p, deg, grid, noise_width, batch_size, num_epochs, device, loss_func):
    x, c, contextual = generate_data(n, p, deg, noise_width, grid)
    loader_train, loader_test, optmodel = create_datasets(x, c, grid, batch_size, contextual)
    
    predmodel = NN(p, 40).to(device)
    
    optimizer = torch.optim.Adam(predmodel.parameters(), lr=0.1)
    
    if loss_func == "spo+":
        criterion = SPOPlus(optmodel, processes=1)
    elif loss_func == "nce":
        criterion = NCEPred(optmodel=optmodel, processes=1, solve_ratio=0.1, reduction="mean", dataset=loader_train.dataset)
    elif loss_func == 'map':
        # criterion = contrastiveMAP(optmodel=optmodel, processes=1, solve_ratio=0.1, reduction="mean", dataset=loader_train.dataset)
        # criterion = pyepo.func.contrastiveMAP(optmodel=optmodel, processes=1, solve_ratio=0.1, reduction="mean", dataset=loader_train.dataset)
        criterion = contrastiveMAPPred(optmodel=optmodel, processes=1, solve_ratio=0.1, reduction="mean", dataset=loader_train.dataset)
    elif loss_func == 'lltr':
        criterion = listwiseLTR(optmodel, processes=2, solve_ratio=0.1, dataset=loader_train.dataset)
    elif loss_func == 'pltr':
        criterion = pairwiseLTR(optmodel, processes=2, solve_ratio=0.1, dataset=loader_train.dataset)
    elif loss_func == 'ptltr':
        criterion = pointwiseLTR(optmodel, processes=2, solve_ratio=0.1, dataset=loader_train.dataset)
    elif loss_func == 'two_stage':
        criterion = torch.nn.MSELoss()

    trained_model, dfl_losses = train_model(predmodel, criterion, contextual, optmodel, optimizer, loader_train, num_epochs, batch_size, device, loss_func)
    optmodel = ExpectedShortestPathModel(grid)
    average_objective, average_regret, average_mse = evaluate_model(trained_model, contextual, optmodel, loader_test, batch_size, device)
    
    return trained_model, average_objective, average_regret, average_mse, dfl_losses

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=1)
    parser.add_argument("--n", nargs="+", type=int, default=[400])
    parser.add_argument("--deg", nargs="+", type=int, default=[6])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--noise_width", type=float, default=0.5)
    parser.add_argument("--num_experiments", type=int, default=5)
    parser.add_argument("--loss_func", type=str, default="spo+", 
                        choices=["spo+", "nce", "dbb", "lltr", 'pltr', 'ptltr', 'map', 'two_stage'])
    args = parser.parse_args()
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    # m_values = [2, 10, 20, 50] #, 80]
    n = args.n
    p = 5 # 5
    grid = (5, 5)
    noise_width = args.noise_width
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    num_experiments = args.num_experiments
    
    results = []
    for n in args.n:
        for deg in args.deg:
            for exp_idx in range(num_experiments):
                print(f"Running experiment for n={n}, deg={deg}, experiment {exp_idx+1}/{num_experiments}")
                    
                model, avg_objective, avg_regret, avg_mse, dfl_losses = run_experiment(
                    n, p, deg, grid, noise_width, batch_size, num_epochs, device, args.loss_func
                )
                    
                result = {
                    "n": n,
                    "p": p,
                    "deg": deg,
                    "grid": grid[0],
                    "noise_width": noise_width,
                    "batch_size": batch_size,
                    "num_epochs": num_epochs,
                    "experiment_index": exp_idx,
                    "average_objective": float(avg_objective),
                    "average_regret": float(avg_regret),
                    "average_mse": float(avg_mse),
                    "final_dfl_loss": float(dfl_losses[-1]),
                    }
                results.append(result)
                    
                # Save loss curves as numpy arrays
                os.makedirs(f"eval/shortp/pred_{args.loss_func}/losses", exist_ok=True)
                np.save(f"eval/shortp/pred_{args.loss_func}/losses/pred_dfl_losses_exp_{exp_idx}_deg{deg}_n{n}_noise_{noise_width}.npy", np.array(dfl_losses))

                # Plot and save DFL loss curve
                plt.rcParams['text.usetex'] = False 
                plt.figure(figsize=(10, 5))
                plt.plot(dfl_losses, label='DFL Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title(f'DFL Loss Curve (deg={deg}, exp={exp_idx})')
                plt.legend()
                plt.savefig(f"eval/shortp/pred_{args.loss_func}/losses/pred_dfl_loss_exp_{exp_idx}_deg{deg}_n{n}_noise_{noise_width}.png")
                plt.close()

                # Plot and save NLL loss curve
                
                # Save model weights
                os.makedirs(f"eval/shortp/pred_{args.loss_func}", exist_ok=True)
                torch.save(model.state_dict(), f"eval/shortp/pred_{args.loss_func}/pred_model_exp_{exp_idx}_deg{deg}_n{n}_noise_{noise_width}.pth")
                
                # Save results
                with open(f"eval/shortp/pred_{args.loss_func}/pred_results_exp_{exp_idx}_deg{deg}_n{n}_noise_{noise_width}.json", "w") as f:
                    json.dump(result, f, indent=2)
            
        # Save all results in a single file
        with open(f"eval/shortp/pred_{args.loss_func}/pred_results_deg{deg}_n{n}_noise_{noise_width}.json", "w") as f:
            json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
