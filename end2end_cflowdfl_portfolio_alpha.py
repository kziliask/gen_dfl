import torch
import time
import copy
import numpy as np
import pyepo
from pyepo.model.grb import optGrbModel
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from pyepo import EPO
from src.generators.cnf import ConditionalFlow
from src.generators.gmm_generator import GMMGenerator
from src.training.generator_losses import generator_nll, grad_norm, set_global_seed
import json
import os
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional, Union

from optModel import ExpectedPortfolioModel
from optDataset import optDataset, portfolio_genData
from torch.utils.data import TensorDataset
import torch.distributions as dist

from func.contrastive import NCE, contrastiveMAP

# include argparse
import argparse

def build_generator(generator, c_dim, x_dim, mixture_components):
    if generator == "cnf":
        return ConditionalFlow(c_dim, x_dim)
    if generator == "gmm":
        return GMMGenerator(x_dim=x_dim, c_dim=c_dim, num_components=mixture_components)
    raise ValueError(f"Unknown generator: {generator}")


def generate_data(m, n, p, deg, dim, noise_width, caps, rank=None, seed=42, contextual_pretrain_epochs=30):
    # weights, x, c = pyepo.data.portfolio.genData(num_data=n, num_features=p, num_assets=m,
    #                                                               deg=deg, noise_level=noise_width, seed=42)
    weights, x, c = portfolio_genData(num_data=n, num_features=p, num_assets=m,
                                                                  deg=deg, noise_level=noise_width, rank=rank, seed=seed)
    contextual = ConditionalFlow(c.shape[1], x.shape[1])
    optimizer = torch.optim.Adam(contextual.parameters(), lr=0.001)
    
    dataset = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(c).float())
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    losses = []
    for epoch in tqdm(range(contextual_pretrain_epochs)):
        for x_t, c_t in loader:
            optimizer.zero_grad()
            loss = generator_nll(contextual, c_t, x_t, reduction="mean")
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
    os.makedirs("eval/portfolio", exist_ok=True)
    plt.savefig('eval/portfolio/contextual_loss_curve.png')
    plt.close()
    
    
    return weights, x, c, contextual

def create_datasets(x, c, batch_size, weights, caps, contextual, seed=246):
    x_train, x_test, c_train, c_test = train_test_split(x, c, test_size=int(x.shape[0]*0.2), random_state=seed)
    
    optmodel = ExpectedPortfolioModel(c_test.shape[1], weights)
    print(weights.shape, x_train.shape, c_train.shape)
    dataset_train = optDataset(optmodel, x_train, c_train, contextual)
    dataset_test = optDataset(optmodel, x_test, c_test, contextual)
    
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False)
    
    return loader_train, loader_test, optmodel

def train_model(gen_model, contextual, optmodel, optimizer, loader_train, num_epochs, batch_size, device,
                alpha=0.5, beta=10, num_generated_samples=200, grad_clip_norm=10.0, seed=42):
    
    dfl_losses = []
    nll_losses = []
    grad_norms = []
    
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
    split_generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = torch.utils.data.random_split(loader_train.dataset, [train_size, val_size], generator=split_generator)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    for epoch in range(num_epochs):
        # Training loop
        for data in tqdm(loader_train, desc=f"Epoch {epoch+1}/{num_epochs}"):
            x, c, w, z = data
            x, c, w, z = x.to(device), c.to(device), w.to(device), z.to(device)
            
            dfl_loss = 0
            for i in range(x.shape[0]):
                c_trues = contextual.sample(num_generated_samples, x[i].unsqueeze(0))
                optmodel.setObj(c_trues.squeeze(0).detach())
                sol_true, _ = optmodel.solve()
                
                c_gens = gen_model.sample(num_generated_samples, x[i].unsqueeze(0))
                
                dfl_loss += criterion(c_gens, torch.tensor(sol_true).float(), alpha=alpha)
            
            dfl_loss = dfl_loss / x.shape[0]
            nll_loss = generator_nll(gen_model, c, x, reduction="mean")
            loss = dfl_loss * beta + nll_loss
            
            optimizer.zero_grad()
            loss.backward()
            grad_norms.append(grad_norm(gen_model.parameters()))
            torch.nn.utils.clip_grad_norm_(gen_model.parameters(), grad_clip_norm)
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
                    c_trues = contextual.sample(num_generated_samples, x[i].unsqueeze(0))
                    optmodel.setObj(c_trues.squeeze(0).detach())
                    sol_true, _ = optmodel.solve()
                    
                    c_gens = gen_model.sample(num_generated_samples, x[i].unsqueeze(0))
                    
                    val_dfl += criterion(c_gens, torch.tensor(sol_true).float(), alpha=alpha)
                
                val_dfl = val_dfl / x.shape[0]
                val_nll = generator_nll(gen_model, c, x, reduction="mean")
            
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
            best_model_state = copy.deepcopy(gen_model.state_dict())
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                gen_model.load_state_dict(best_model_state)
                break
    return gen_model, dfl_losses, nll_losses, grad_norms

def evaluate_model(gen_model, contextual, optmodel, loader_test, batch_size, device, alpha=1, num_generated_samples=200):
    
    average_objectives = []
    average_regrets = []
    cvar_objectives = []
    cvar_regrets = []
    cvar_01_objectives = []
    cvar_01_regrets = []
    
    for data in tqdm(loader_test, desc="Evaluating"):
        x, c, w, z = data
        x, c, w, z = x.to(device), c.to(device), w.to(device), z.to(device)
        
        cp_samples = gen_model.sample(num_generated_samples, x).detach()
        # cp_samples = cp_samples.view(1 * 200, -1)
        c_trues = contextual.sample(num_generated_samples, x).detach()
        
        # Average Objective
        obj = optmodel.obj_eval(x, cp_samples, c_trues, alpha=1)
        average_objectives.append(obj)# / (z.abs().sum().item() + 1e-7))
        # optmodel.setObj(cp_samples.squeeze(0).detach())
        # sol, _ = optmodel.solve()
        # sol_tensor = torch.tensor(sol, dtype=c.dtype, device=c.device)
        # obj = torch.matmul(c, sol_tensor)
        # average_objectives.append(obj.item())

        # Average Regret
        obj = optmodel.regret_loss_batch(x, cp_samples, c_trues, alpha=1)
        average_regrets.append(obj / (z.abs().sum().item() + 1e-7))
        # average_regrets.append(torch.abs(obj - z).mean().item())


        # CVaR Objective
        obj = optmodel.obj_eval(x, cp_samples, c_trues, alpha=0.5)
        cvar_objectives.append(obj)# / (z.abs().sum().item() + 1e-7))
        
        # CVaR Regret
        obj = optmodel.regret_loss_batch(x, cp_samples, c_trues, alpha=0.5)
        cvar_regrets.append(obj / (z.abs().sum().item() + 1e-7))

        # CVaR 0.1 objective
        obj = optmodel.obj_eval(x, cp_samples, c_trues, alpha=0.1)
        cvar_01_objectives.append(obj)# / (z.abs().sum().item() + 1e-7))
        
        # CVaR 0.1 regret
        obj = optmodel.regret_loss_batch(x, cp_samples, c_trues, alpha=0.1)
        cvar_01_regrets.append(obj / (z.abs().sum().item() + 1e-7))
    
    print(f"Average Objective: {np.mean(average_objectives)}")
    print(f"Average Regret: {np.mean(average_regrets)}")
    print(f"CVaR Objective: {np.mean(cvar_objectives)}")
    print(f"CVaR Regret: {np.mean(cvar_regrets)}")
    print(f"CVaR 0.1 Objective: {np.mean(cvar_01_objectives)}")
    print(f"CVaR 0.1 Regret: {np.mean(cvar_01_regrets)}")
    return np.mean(average_objectives), np.mean(average_regrets), np.mean(cvar_objectives), np.mean(cvar_regrets), np.mean(cvar_01_objectives), np.mean(cvar_01_regrets)

def pretrain_model(gen_model, optimizer, loader_train, num_epochs, device, grad_clip_norm=10.0):
    nll_losses = []
    
    for epoch in range(num_epochs):
        epoch_losses = []
        for data in tqdm(loader_train, desc=f"Pretraining Epoch {epoch+1}/{num_epochs}"):
            x, c, w, z = data
            x, c = x.to(device), c.to(device)
            
            nll_loss = generator_nll(gen_model, c, x, reduction="mean")
            
            optimizer.zero_grad()
            nll_loss.backward()
            torch.nn.utils.clip_grad_norm_(gen_model.parameters(), grad_clip_norm)
            optimizer.step()

            epoch_losses.append(nll_loss.item())
        
        avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
        nll_losses.append(avg_epoch_loss)
        print(f"Average NLL Loss on Training Set at epoch {epoch}: {avg_epoch_loss}")
    
    return gen_model, nll_losses

def run_experiment(m, n, p, deg, dim, noise_width, caps, batch_size, num_epochs, device, beta, alpha,
                   rank=None, seed=42, generator="cnf", mixture_components=1, num_generated_samples=200,
                   contextual_pretrain_epochs=30, p_theta_pretrain_epochs=50, learning_rate=0.001,
                   grad_clip_norm=10.0):
    set_global_seed(seed)
    data_start = time.perf_counter()
    weights, x, c, contextual = generate_data(
        m, n, p, deg, dim, noise_width, caps, rank=rank, seed=seed,
        contextual_pretrain_epochs=contextual_pretrain_epochs
    )
    contextual_pretrain_time = time.perf_counter() - data_start
    loader_train, loader_test, optmodel = create_datasets(x, c, batch_size, weights, caps, contextual, seed=seed)
    
    gen_model = build_generator(generator, c.shape[1], x.shape[1], mixture_components)
    gen_model.to(device)
    
    optimizer = torch.optim.Adam(gen_model.parameters(), lr=learning_rate)
    
    pretrain_start = time.perf_counter()
    gen_model, pretrain_losses = pretrain_model(
        gen_model, optimizer, loader_train, p_theta_pretrain_epochs, device, grad_clip_norm=grad_clip_norm
    )
    p_theta_pretrain_time = time.perf_counter() - pretrain_start
    
    # change the learning rate
    optimizer = torch.optim.Adam(gen_model.parameters(), lr=learning_rate)
    train_start = time.perf_counter()
    trained_model, dfl_losses, nll_losses, grad_norms = train_model(
        gen_model, contextual, optmodel, optimizer, loader_train, num_epochs, batch_size, device,
        beta=beta, alpha=alpha, num_generated_samples=num_generated_samples,
        grad_clip_norm=grad_clip_norm, seed=seed
    )
    dfl_train_time = time.perf_counter() - train_start
    eval_start = time.perf_counter()
    average_objective, average_regret, cvar_objective, cvar_regret, cvar_01_objective, cvar_01_regret = evaluate_model(
        trained_model, contextual, optmodel, loader_test, batch_size, device,
        num_generated_samples=num_generated_samples
    )
    eval_time = time.perf_counter() - eval_start
    timing = {
        "contextual_pretrain_time_seconds": contextual_pretrain_time,
        "p_theta_pretrain_time_seconds": p_theta_pretrain_time,
        "dfl_train_time_seconds": dfl_train_time,
        "train_time_seconds": contextual_pretrain_time + p_theta_pretrain_time + dfl_train_time,
        "eval_time_seconds": eval_time,
        "max_gradient_norm": max(grad_norms) if grad_norms else None,
        "final_gradient_norm": grad_norms[-1] if grad_norms else None,
        "pretrain_final_nll_loss": pretrain_losses[-1] if pretrain_losses else None,
    }
    
    return trained_model, average_objective, average_regret, cvar_objective, cvar_regret, cvar_01_objective, cvar_01_regret, dfl_losses, nll_losses, timing

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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--generator", choices=["cnf", "gmm"], default="cnf")
    parser.add_argument("--mixture_components", type=int, default=1)
    parser.add_argument("--num_generated_samples", type=int, default=200)
    parser.add_argument("--contextual_pretrain_epochs", type=int, default=30)
    parser.add_argument("--p_theta_pretrain_epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--grad_clip_norm", type=float, default=10.0)
    parser.add_argument("--raw_results_path", type=str, default="results/raw/portfolio_generator_comparison.jsonl")
    args = parser.parse_args()
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    # m_values = [2, 10, 20, 50] #, 80]
    m_values = [args.m]
    n = args.n
    p = 5 # 5
    dim = 2
    noise_width = args.noise_width
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    num_experiments = args.num_experiments
    
    # betas = [0, 0.1, 1, 10, 50]
    beta = args.betas
    alphas = args.alpha
    results = []
    
    for deg in args.deg:
        for m in m_values:
            caps = [20] * dim
            for alpha in alphas:
                for exp_idx in range(num_experiments):
                    run_seed = args.seed + exp_idx
                    print(f"Running experiment for m={m}, alpha={alpha}, deg={deg}, seed={run_seed}, experiment {exp_idx+1}/{num_experiments}")
                    
                    model, avg_objective, avg_regret, cvar_objective, cvar_regret, cvar_01_objective, cvar_01_regret, dfl_losses, nll_losses, timing = run_experiment(
                        m, n, p, deg, dim, noise_width, caps, batch_size, num_epochs, device, beta, alpha,
                        rank=args.rank, seed=run_seed, generator=args.generator,
                        mixture_components=args.mixture_components,
                        num_generated_samples=args.num_generated_samples,
                        contextual_pretrain_epochs=args.contextual_pretrain_epochs,
                        p_theta_pretrain_epochs=args.p_theta_pretrain_epochs,
                        learning_rate=args.learning_rate,
                        grad_clip_norm=args.grad_clip_norm
                    )
                    
                    result = {
                        "task": "portfolio",
                        "model": "gen-dfl",
                        "generator": args.generator,
                        "seed": run_seed,
                        "m": m,
                        "n": n,
                        "p": p,
                        "beta": beta,
                        "alpha": alpha,
                        "deg": deg,
                        "dim": dim,
                        "noise_width": noise_width,
                        "caps": caps,
                        "batch_size": batch_size,
                        "num_epochs": num_epochs,
                        "epochs": num_epochs,
                        "experiment_index": exp_idx,
                        "num_generated_samples": args.num_generated_samples,
                        "mixture_components": args.mixture_components if args.generator == "gmm" else "",
                        "covariance_type": "diagonal" if args.generator == "gmm" else "",
                        "learning_rate": args.learning_rate,
                        "q_architecture": "cnf",
                        "p_theta_architecture": args.generator,
                        "q_pretrain_epochs": args.contextual_pretrain_epochs,
                        "p_theta_pretrain_epochs": args.p_theta_pretrain_epochs,
                        "average_objective": float(avg_objective),
                        "average_regret": float(avg_regret),
                        "cvar_objective": float(cvar_objective),
                        "cvar_regret": float(cvar_regret),
                        "cvar_01_objective": float(cvar_01_objective),
                        "cvar_01_regret": float(cvar_01_regret),
                        "metric_objective": float(avg_objective),
                        "metric_regret": float(avg_regret),
                        "metric_true_regret": float(avg_regret),
                        "metric_proxy_regret": float(avg_regret),
                        "metric_cvar_regret": float(cvar_regret),
                        "metric_cvar_01_regret": float(cvar_01_regret),
                        "metric_nll": float(nll_losses[-1]),
                        "final_dfl_loss": float(dfl_losses[-1]),
                        "final_nll_loss": float(nll_losses[-1]),
                        "status": "success",
                        **timing,
                    }
                    results.append(result)
                    if args.raw_results_path:
                        raw_dir = os.path.dirname(args.raw_results_path)
                        if raw_dir:
                            os.makedirs(raw_dir, exist_ok=True)
                        with open(args.raw_results_path, "a") as f:
                            f.write(json.dumps(result, default=float) + "\n")
                    
                    eval_label = f"gmm_k{args.mixture_components}" if args.generator == "gmm" else "cnf"
                    eval_dir = f"eval/portfolio/beta_{beta}_alpha_{alpha}"
                    if args.generator != "cnf":
                        eval_dir = f"eval/portfolio/{eval_label}_beta_{beta}_alpha_{alpha}"
                    loss_dir = f"{eval_dir}/losses"
                    file_suffix = f"exp_{exp_idx}_m{m}_n{n}_noise_{noise_width}"

                    # Save loss curves as numpy arrays
                    os.makedirs(loss_dir, exist_ok=True)
                    np.save(f"{loss_dir}/end2end_cflowdfl_dfl_losses_{file_suffix}.npy", np.array(dfl_losses))
                    np.save(f"{loss_dir}/end2end_cflowdfl_nll_losses_{file_suffix}.npy", np.array(nll_losses))

                    # Plot and save DFL loss curve
                    plt.rcParams['text.usetex'] = False 
                    plt.figure(figsize=(10, 5))
                    plt.plot(dfl_losses, label='DFL Loss')
                    plt.xlabel('Epoch')
                    plt.ylabel('Loss')
                    plt.title(f'DFL Loss Curve (m={m}, exp={exp_idx})')
                    plt.legend()
                    plt.savefig(f"{loss_dir}/end2end_cflowdfl_dfl_loss_{file_suffix}.png")
                    plt.close()

                    # Plot and save NLL loss curve
                    plt.figure(figsize=(10, 5))
                    plt.plot(nll_losses, label='NLL Loss')
                    plt.xlabel('Epoch')
                    plt.ylabel('Loss')
                    plt.title(f'NLL Loss Curve (beta={beta}, exp={exp_idx})')
                    plt.legend()
                    plt.savefig(f"{loss_dir}/end2end_cflowdfl_nll_loss_{file_suffix}.png")
                    plt.close()
                    
                    # Save model weights
                    os.makedirs(eval_dir, exist_ok=True)
                    torch.save(model.state_dict(), f"{eval_dir}/end2end_cflowdfl_model_{file_suffix}.pth")
                    
                    # Save results
                    with open(f"{eval_dir}/end2end_cflowdfl_results_{file_suffix}.json", "w") as f:
                        json.dump(result, f, indent=2)
            
    # Save all results in a single file
    with open(f"eval/portfolio/all_results_end2end_betas{beta}_alpha_{alpha}_m{m}_noise_{noise_width}.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
