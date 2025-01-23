import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from pyepo import EPO
import json
import os
from sklearn.model_selection import train_test_split

from optModel import ExpectedPortfolioModel
from optDataset import optDataset, portfolio_genData
from train import ConditionalFlow
from diffusion import ConditionalDiffusionModel
from func.contrastive import contrastiveMAP

import argparse

def generate_data(m, n, p, deg, dim, noise_width, caps, rank=None):
    weights, x, c = portfolio_genData(num_data=n, num_features=p, num_assets=m,
                                    deg=deg, noise_level=noise_width, rank=rank, seed=42)
    
    # Initialize and train contextual flow model (unchanged)
    contextual = ConditionalFlow(c.shape[1], x.shape[1])
    optimizer = torch.optim.Adam(contextual.parameters(), lr=0.001)
    
    dataset = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(c).float())
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    losses = []
    for epoch in tqdm(range(30)):
        for x_t, c_t in loader:
            optimizer.zero_grad()
            z, log_det = contextual(c_t, x_t)
            
            log_prob = -0.5 * torch.sum(z**2, dim=1) - 0.5 * z.size(1) * np.log(2 * np.pi)
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
    
    return weights, x, c, contextual

def create_datasets(x, c, batch_size, weights, caps, contextual):
    x_train, x_test, c_train, c_test = train_test_split(x, c, test_size=0.2, random_state=246)
    
    optmodel = ExpectedPortfolioModel(c_test.shape[1], weights)
    dataset_train = optDataset(optmodel, x_train, c_train, contextual)
    dataset_test = optDataset(optmodel, x_test, c_test, contextual)
    
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False)
    
    return loader_train, loader_test, optmodel

def train_model(gen_model, contextual, optmodel, optimizer, loader_train, num_epochs, 
                batch_size, device, guidance_scale=3.0, null_prob=0.1, alpha=0.5, beta=10):
    
    dfl_losses = []
    diffusion_losses = []
    
    criterion = contrastiveMAP(optmodel=optmodel, processes=1, solve_ratio=1, reduction="mean", 
                             dataset=loader_train.dataset)
    
    # Early stopping parameters
    patience = 5
    best_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    train_size = int(0.9 * len(loader_train.dataset))
    val_size = len(loader_train.dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(loader_train.dataset, [train_size, val_size])
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    for epoch in range(num_epochs):
        for data in tqdm(loader_train, desc=f"Epoch {epoch+1}/{num_epochs}"):
            x, c, w, z = data
            x, c, w, z = x.to(device), c.to(device), w.to(device), z.to(device)
            
            # DFL loss computation
            dfl_loss = 0
            for i in range(x.shape[0]):
                c_trues = contextual.sample(200, x[i].unsqueeze(0))
                optmodel.setObj(c_trues.squeeze(0).detach())
                sol_true, _ = optmodel.solve()
                
                c_gens = gen_model.sample(200, x[i].unsqueeze(0))
                dfl_loss += criterion(c_gens, torch.tensor(sol_true).float(), alpha=alpha)
            
            dfl_loss = dfl_loss / x.shape[0]
            
            # Diffusion loss with classifier-free guidance
            diffusion_loss = gen_model.get_loss(c, x, guidance_scale, null_prob)
            
            # Total loss
            loss = dfl_loss * beta + diffusion_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            dfl_losses.append(dfl_loss.item())
            diffusion_losses.append(diffusion_loss.item())
        
        print(f"DFL Loss on Training Set at epoch {epoch}: {dfl_loss.item()}")
        print(f"Diffusion Loss on Training Set at epoch {epoch}: {diffusion_loss.item()}")
        
        # Validation
        val_dfl_loss = 0
        val_diff_loss = 0
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
                val_diff = gen_model.get_loss(c, x, guidance_scale, null_prob)
                
                val_dfl_loss += val_dfl.item()
                val_diff_loss += val_diff.item()
        
        gen_model.train()
        val_dfl_loss /= len(val_loader)
        val_diff_loss /= len(val_loader)
        val_total_loss = val_dfl_loss * beta + val_diff_loss
        
        if val_total_loss < best_loss:
            best_loss = val_total_loss
            patience_counter = 0
            best_model_state = gen_model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                gen_model.load_state_dict(best_model_state)
                break
    
    return gen_model, dfl_losses, diffusion_losses

def evaluate_model(gen_model, contextual, optmodel, loader_test, batch_size, device, alpha=1):
    average_objectives = []
    average_regrets = []
    cvar_objectives = []
    cvar_regrets = []
    cvar_01_objectives = []
    cvar_01_regrets = []
    cvar_025_objectives = []
    cvar_025_regrets = []
    cvar_075_objectives = []
    cvar_075_regrets = []
    
    for data in tqdm(loader_test, desc="Evaluating"):
        x, c, w, z = data
        x, c, w, z = x.to(device), c.to(device), w.to(device), z.to(device)
        
        cp_samples = gen_model.sample(200, x).squeeze(1)  # Remove channel dim from diffusion output
        c_trues = contextual.sample(200, x)
        
        # Average Objective
        obj = optmodel.obj_eval(x, cp_samples, c_trues, alpha=1)
        average_objectives.append(obj)
        
        # Average Regret
        obj = optmodel.regret_loss_batch(x, cp_samples, c_trues, alpha=1)
        average_regrets.append(obj / (z.abs().sum().item() + 1e-7))
        
        # Various CVaR computations
        for alpha_val, objectives, regrets in [
            (0.5, cvar_objectives, cvar_regrets),
            (0.1, cvar_01_objectives, cvar_01_regrets),
            (0.25, cvar_025_objectives, cvar_025_regrets),
            (0.75, cvar_075_objectives, cvar_075_regrets)
        ]:
            obj = optmodel.obj_eval(x, cp_samples, c_trues, alpha=alpha_val)
            objectives.append(obj)
            obj = optmodel.regret_loss_batch(x, cp_samples, c_trues, alpha=alpha_val)
            regrets.append(obj / (z.abs().sum().item() + 1e-7))
    
    cvars = {
        "cvar_objective": np.mean(cvar_objectives),
        "cvar_regrets": np.mean(cvar_regrets),
        "cvar_01_objective": np.mean(cvar_01_objectives),
        "cvar_01_regrets": np.mean(cvar_01_regrets),
        "cvar_025_objective": np.mean(cvar_025_objectives),
        "cvar_025_regrets": np.mean(cvar_025_regrets),
        "cvar_075_objective": np.mean(cvar_075_objectives),
        "cvar_075_regrets": np.mean(cvar_075_regrets)
    }
    
    return (np.mean(average_objectives), np.mean(average_regrets), 
            cvars, np.mean(cvar_objectives), np.mean(cvar_regrets),
            np.mean(cvar_01_objectives), np.mean(cvar_01_regrets))

def run_experiment(m, n, p, deg, dim, noise_width, caps, batch_size, num_epochs, device, 
                  beta, alpha, guidance_scale=3.0, null_prob=0.1, rank=None):
    weights, x, c, contextual = generate_data(m, n, p, deg, dim, noise_width, caps, rank=rank)
    loader_train, loader_test, optmodel = create_datasets(x, c, batch_size, weights, caps, contextual)
    
    # Initialize diffusion model instead of flow
    gen_model = ConditionalDiffusionModel(x_dim=x.shape[1], c_dim=c.shape[1])
    gen_model.to(device)
    
    optimizer = torch.optim.Adam(gen_model.parameters(), lr=0.001)
    
    trained_model, dfl_losses, diff_losses = train_model(
        gen_model, contextual, optmodel, optimizer, loader_train, num_epochs, batch_size,
        device, guidance_scale, null_prob, alpha=alpha, beta=beta
    )
    
    results = evaluate_model(trained_model, contextual, optmodel, loader_test, batch_size, device)
    return (trained_model, *results, dfl_losses, diff_losses)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--betas", type=float, default=10)
    parser.add_argument("--alpha", nargs="+", type=float, default=[1])
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--m", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--noise_width", type=int, default=100)
    parser.add_argument("--num_experiments", type=int, default=5)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--deg", nargs="+", type=int, default=[6])
    parser.add_argument("--guidance_scale", type=float, default=3.0)
    parser.add_argument("--null_prob", type=float, default=0.1)
    args = parser.parse_args()
    
    device = "cpu"  # or "cuda" if available
    m_values = [args.m]
    n = args.n
    p = 5
    dim = 2
    noise_width = args.noise_width
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    num_experiments = args.num_experiments
    beta = args.betas
    alphas = args.alpha
    
    results = []
    
    for deg in args.deg:
        for m in m_values:
            caps = [20] * dim
            for alpha in alphas:
                for exp_idx in range(num_experiments):
                    print(f"Running experiment for m={m}, alpha={alpha}, deg={deg}, experiment {exp_idx+1}/{num_experiments}")
                    
                    model, avg_obj, avg_reg, cvars, cvar_obj, cvar_reg, cvar_01_obj, cvar_01_reg, dfl_losses, diff_losses = run_experiment(
                        m, n, p, deg, dim, noise_width, caps, batch_size, num_epochs, device,
                        beta, alpha, args.guidance_scale, args.null_prob, rank=args.rank
                    )
                    
                    result = {
                        "m": m, "n": n, "p": p, "beta": beta, "deg": deg,
                        "dim": dim, "noise_width": noise_width, "caps": caps,
                        "batch_size": batch_size, "num_epochs": num_epochs,
                        "guidance_scale": args.guidance_scale,
                        "null_prob": args.null_prob,
                        "experiment_index": exp_idx,
                        "average_objective": float(avg_obj),
                        "average_regret": float(avg_reg),
                        "cvar_objective": float(cvar_obj),
                        "cvar_regret": float(cvar_reg),
                        "cvar_01_objective": float(cvar_01_obj),
                        "cvar_01_regret": float(cvar_01_reg),
                        **{f"{k}": float(v) for k, v in cvars.items()},
                        "final_dfl_loss": float(dfl_losses[-1]),
                        "final_diffusion_loss": float(diff_losses[-1])
                    }
                    results.append(result)
                    
                    # Save results and visualizations
                    save_dir = f"eval/portfolio/diffusion_beta_{beta}_alpha_{alpha}"
                    os.makedirs(f"{save_dir}/losses", exist_ok=True)
                    
                    # Save loss curves as numpy arrays
                    np.save(f"{save_dir}/losses/dfl_losses_exp_{exp_idx}_m{m}_n{n}_noise_{noise_width}.npy",
                           np.array(dfl_losses))
                    np.save(f"{save_dir}/losses/diffusion_losses_exp_{exp_idx}_m{m}_n{n}_noise_{noise_width}.npy",
                           np.array(diff_losses))
                    
                    # Plot and save DFL loss curve
                    plt.figure(figsize=(10, 5))
                    plt.plot(dfl_losses, label='DFL Loss')
                    plt.xlabel('Iteration')
                    plt.ylabel('Loss')
                    plt.title(f'DFL Loss Curve (m={m}, exp={exp_idx})')
                    plt.legend()
                    plt.savefig(f"{save_dir}/losses/dfl_loss_exp_{exp_idx}_m{m}_n{n}_noise_{noise_width}.png")
                    plt.close()
                    
                    # Plot and save Diffusion loss curve
                    plt.figure(figsize=(10, 5))
                    plt.plot(diff_losses, label='Diffusion Loss')
                    plt.xlabel('Iteration')
                    plt.ylabel('Loss')
                    plt.title(f'Diffusion Loss Curve (m={m}, exp={exp_idx})')
                    plt.legend()
                    plt.savefig(f"{save_dir}/losses/diffusion_loss_exp_{exp_idx}_m{m}_n{n}_noise_{noise_width}.png")
                    plt.close()
                    
                    # Save model weights
                    torch.save(model.state_dict(), 
                             f"{save_dir}/model_exp_{exp_idx}_m{m}_n{n}_noise_{noise_width}.pth")
                    
                    # Save individual experiment results
                    with open(f"{save_dir}/results_exp_{exp_idx}_m{m}_n{n}_noise_{noise_width}.json", "w") as f:
                        json.dump(result, f, indent=2)
    
    # Save all results in a single file
    with open(f"eval/portfolio/all_results_diffusion_beta{beta}_alpha_{alpha}_m{m}_noise_{noise_width}_n{n}.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()