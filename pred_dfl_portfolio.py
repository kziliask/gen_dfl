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

from optModel import ExpectedPortfolioModel
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

def generate_data(m, n, p, deg, dim, noise_width, caps, rank=None):
    # covariance, x, c = pyepo.data.portfolio.genData(num_data=n, num_features=p, num_assets=m,
    #                                                               deg=deg, noise_level=noise_width, seed=42)
    covariance, x, c = portfolio_genData(num_data=n, num_features=p, num_assets=m,
                                                                  deg=deg, noise_level=noise_width, rank=rank, seed=42)
    # mb_size = 50
    # fake_zs = torch.randn((mb_size, m))
    # fake_xs = torch.randn((mb_size, p))
    # contextual = build_nsf(fake_zs, fake_xs, z_score_x='none', z_score_y='none').float()   


    contextual = ConditionalFlow(c.shape[1], x.shape[1])
    optimizer = torch.optim.Adam(contextual.parameters(), lr=0.001)
    
    dataset = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(c).float())
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    losses = []
    for epoch in tqdm(range(30)):
        for x_t, c_t in loader:
            optimizer.zero_grad()
            # loss = -1 * contextual.log_prob(c_t, x_t).mean()
            
            # Forward pass: transform to latent space
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
    plt.savefig('contextual_loss_curve.png')
    plt.close()
    
    
    return covariance, x, c, contextual

def create_datasets(x, c, batch_size, covariance, contextual):
    x_train, x_test, c_train, c_test = train_test_split(x, c, test_size=int(x.shape[0]*0.2), random_state=246)
    
    optmodel = ExpectedPortfolioModel(c_test.shape[1], covariance)
    dataset_train = optDataset(optmodel, x_train, c_train, contextual)
    dataset_test = optDataset(optmodel, x_test, c_test, contextual)
    
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False)
    
    optmodel = pyepo.model.grb.portfolioModel(c_test.shape[1], covariance)

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
        c_trues = contextual.sample(200, x).detach().squeeze()
        obj = torch.matmul(c_trues, sol_tensor)
        obj = torch.matmul(c, sol_tensor)
        average_objectives.append(obj.mean().item())# / (z.abs().sum().item() + 1e-7))
        # Average Regret
        average_regrets.append(torch.abs(obj - z).mean().item() / (z.abs().sum().item() + 1e-7))

        
    print(f"Average Objective: {np.mean(average_objectives)}")
    print(f"Average Regret: {np.mean(average_regrets)}")
    print(f"Average MSE: {np.mean(mse_losses)}")
    return np.mean(average_objectives), np.mean(average_regrets), np.mean(mse_losses)

def run_experiment(m, n, p, deg, dim, noise_width, caps, batch_size, num_epochs, device, loss_func, rank=None):
    covariance, x, c, contextual = generate_data(m, n, p, deg, dim, noise_width, caps, rank=rank)
    loader_train, loader_test, optmodel = create_datasets(x, c, batch_size, covariance, contextual)
    
    predmodel = NN(p, m).to(device)
    
    optimizer = torch.optim.Adam(predmodel.parameters(), lr=0.001)
    
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
    optmodel = ExpectedPortfolioModel(c.shape[-1], covariance)
    average_objective, average_regret, average_mse = evaluate_model(trained_model, contextual, optmodel, loader_test, batch_size, device)
    
    return trained_model, average_objective, average_regret, average_mse, dfl_losses

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--m", nargs="+", type=int, default=[10])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--noise_widths", nargs="+", type=int, default=[100])
    parser.add_argument("--loss_func", type=str, default="spo+", 
                        choices=["spo+", "nce", "dbb", "lltr", 'pltr', 'ptltr', 'map', 'two_stage'])
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--deg", nargs="+", type=int, default=[6])
    parser.add_argument("--num_experiments", type=int, default=10)
    args = parser.parse_args()
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    # m_values = [2, 10, 20, 50] #, 80]
    m_values = args.m
    n = args.n
    p = 5 # 5
    dim = 2
    noise_widths = args.noise_widths
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    num_experiments = args.num_experiments
    
    results = []
    for deg in args.deg:
        for noise_width in noise_widths:
            for m in m_values:
                caps = [20] * dim
                for exp_idx in range(num_experiments):
                    print(f"Running experiment for m={m}, noise_width={noise_width}, deg={deg}, experiment {exp_idx+1}/{num_experiments}")
                        
                    model, avg_objective, avg_regret, avg_mse, dfl_losses = run_experiment(
                        m, n, p, deg, dim, noise_width, caps, batch_size, num_epochs, device, args.loss_func, rank=args.rank
                    )
                        
                    result = {
                        "m": m,
                        "n": n,
                        "p": p,
                        "deg": deg,
                        "dim": dim,
                        "noise_width": noise_width,
                        "caps": caps,
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
                    os.makedirs(f"eval/portfolio/pred_{args.loss_func}/losses", exist_ok=True)
                    np.save(f"eval/portfolio/pred_{args.loss_func}/losses/pred_dfl_losses_deg{deg}_exp_{exp_idx}_m{m}_n{n}_noise_{noise_width}.npy", np.array(dfl_losses))

                    # Plot and save DFL loss curve
                    plt.rcParams['text.usetex'] = False 
                    plt.figure(figsize=(10, 5))
                    plt.plot(dfl_losses, label='DFL Loss')
                    plt.xlabel('Epoch')
                    plt.ylabel('Loss')
                    plt.title(f'DFL Loss Curve (m={m}, deg={deg}, exp={exp_idx})')
                    plt.legend()
                    plt.savefig(f"eval/portfolio/pred_{args.loss_func}/losses/pred_dfl_loss_deg{deg}_exp_{exp_idx}_m{m}_n{n}_noise_{noise_width}.png")
                    plt.close()

                    # Plot and save NLL loss curve
                    
                    # Save model weights
                    os.makedirs(f"eval/portfolio/pred_{args.loss_func}", exist_ok=True)
                    torch.save(model.state_dict(), f"eval/portfolio/pred_{args.loss_func}/pred_model_deg{deg}_exp_{exp_idx}_m{m}_n{n}_noise_{noise_width}.pth")
                    
                    # Save results
                    with open(f"eval/portfolio/pred_{args.loss_func}/pred_results_deg{deg}_exp_{exp_idx}_m{m}_n{n}_noise_{noise_width}.json", "w") as f:
                        json.dump(result, f, indent=2)
                
            # Save all results in a single file
            with open(f"eval/portfolio/pred_{args.loss_func}/pred_results_deg{deg}_m{m}_n{n}_noise_{noise_width}.json", "w") as f:
                json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
