#!/usr/bin/env python3

"""
synthetic_data.py

Generates TWO sets of synthetic power data (train + test) with different SIR parameters,
then saves each set (and associated plots) to a specified folder.

Train parameters:
  beta=[0.5, 0.4, 0.2], gamma=[0.2, 0.15, 0.1]
Test parameters:
  beta=[0.45, 0.35, 0.25], gamma=[0.18, 0.14, 0.12]

Usage example:
    python synthetic_data.py --seed 42 --outfolder synthetic_power_data --num_cities 3 \
        --t_end 50 --t_steps 51
"""

import os
import argparse
import math
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchdiffeq import odeint

##############################################################################
# A) Weather generation
##############################################################################

def build_correlation_matrix(dim, rho=0.5, variance=1.0):
    """Simple correlation matrix with diagonal=variance, offdiag=rho*variance."""
    corr_mat = (1 - rho) * np.eye(dim) + rho * np.ones((dim, dim))
    cov_mat = variance * corr_mat
    return cov_mat

def generate_correlated_weather(time_points, num_cities=3, seed=0, rho=0.6, variance=2.0):
    """
    Returns a tensor of shape (T, num_cities, 3) for (wind, pressure, temperature),
    correlated across cities at each time step.
    """
    np.random.seed(seed)
    dim = num_cities * 3
    cov_mat = build_correlation_matrix(dim, rho=rho, variance=variance)
    mean_vec = np.zeros(dim, dtype=np.float32)

    def base_pattern(city_idx, factor_idx, t):
        if factor_idx == 0:  # wind
            base, amp, freq = 10.0, 5.0, 0.05
        elif factor_idx == 1:  # pressure
            base, amp, freq = 1000.0, 30.0, 0.02
        else:  # temperature
            base, amp, freq = 15.0, 10.0, 0.01
        
        phase_shift = city_idx * math.pi / num_cities
        return base + amp * math.sin(freq * t + phase_shift)

    weather_list = []
    for t in time_points:
        base_vec = []
        for city in range(num_cities):
            for factor in range(3):
                val = base_pattern(city, factor, float(t))
                base_vec.append(val)
        base_vec = np.array(base_vec, dtype=np.float32)

        eps = np.random.multivariate_normal(mean_vec, cov_mat).astype(np.float32)
        final_vec = base_vec + eps
        final_reshaped = final_vec.reshape(num_cities, 3)
        weather_list.append(final_reshaped)

    w_array = np.stack(weather_list, axis=0)
    return torch.tensor(w_array, dtype=torch.float32)

##############################################################################
# B) WeatherDependentSIR
##############################################################################

class WeatherDependentSIR(torch.nn.Module):
    """
    SIR model where city i's beta depends on wind, pressure, temperature.
    """
    def __init__(self, beta_base, gamma_base,
                 alpha_wind, alpha_press, alpha_temp,
                 weather_data, total_households):
        super().__init__()
        self.beta_base = beta_base
        self.gamma_base = gamma_base
        self.alpha_wind = alpha_wind
        self.alpha_press= alpha_press
        self.alpha_temp = alpha_temp

        self.weather_data = weather_data
        self.T = weather_data.shape[0]
        self.num_cities = weather_data.shape[1]
        self.total_households = total_households

    def forward(self, t, y):
        # y: (num_cities,3) => S,I,R
        S = y[:, 0]
        I = y[:, 1]
        R = y[:, 2]

        t_idx = int(torch.round(t).item())
        t_idx = max(0, min(self.T-1, t_idx))

        w_t = self.weather_data[t_idx, :, 0]
        p_t = self.weather_data[t_idx, :, 1]
        tmp_t= self.weather_data[t_idx, :, 2]

        beta_t = self.beta_base + self.alpha_wind*w_t + self.alpha_press*p_t + self.alpha_temp*tmp_t
        gamma_t= self.gamma_base
        N = self.total_households
        dSdt = -beta_t * S * I / N
        dIdt =  beta_t * S * I / N - gamma_t * I
        dRdt =  gamma_t * I
        return torch.stack([dSdt, dIdt, dRdt], dim=1)

##############################################################################
# C) Generation function with custom Beta/Gamma
##############################################################################

def generate_sir_data(num_cities, seed, t_start, t_end, t_steps,
                      beta_base, gamma_base):
    """
    1) Generate correlated weather
    2) Build WeatherDependentSIR with given beta_base, gamma_base
    3) Solve ODE => (S,I,R)
    4) Add noise => sir_noisy
    """
    time_points = torch.linspace(t_start, t_end, t_steps)
    total_households = torch.tensor([1000.0, 1200.0, 2000.0, 2500.0, 3000.0])[:num_cities]
    # weather
    weather_data = generate_correlated_weather(
        time_points, num_cities=num_cities, seed=seed,
        rho=0.6, variance=2.0
    )
    alpha_wind  = torch.tensor([0.01, 0.02,  0.015, 0.018, 0.025])[:num_cities]
    alpha_press = torch.tensor([0.0002, 0.0001, 0.00015,0.00012, 0.00013])[:num_cities]
    alpha_temp  = torch.tensor([0.005,  0.007,  0.006,  0.0065, 0.0075])[:num_cities]

    # Build SIR
    sir_model = WeatherDependentSIR(
        beta_base, gamma_base,
        alpha_wind, alpha_press, alpha_temp,
        weather_data, total_households
    )

    init_infected_frac = torch.tensor([0.01, 0.02, 0.01, 0.015, 0.02])[:num_cities]
    init_infected  = total_households * init_infected_frac
    init_recovered = torch.zeros_like(init_infected)
    init_suscept   = total_households - init_infected
    y0 = torch.stack([init_suscept, init_infected, init_recovered], dim=1)

    from torchdiffeq import odeint
    with torch.no_grad():
        sir_trajectory = odeint(sir_model, y0, time_points)

    # add noise
    noise_std = 0.05
    sir_noisy = sir_trajectory.clone()
    for t_idx in range(sir_trajectory.shape[0]):
        for city_idx in range(num_cities):
            S_ = sir_trajectory[t_idx, city_idx, 0]
            I_ = sir_trajectory[t_idx, city_idx, 1]
            N_ = total_households[city_idx]

            noiseS = torch.randn(1) * noise_std * S_
            noiseI = torch.randn(1) * noise_std * I_

            S_noisy = torch.clamp(S_ + noiseS, 0, N_.item())
            I_noisy = torch.clamp(I_ + noiseI, 0, N_.item())
            R_noisy = torch.clamp(N_ - S_noisy - I_noisy, 0, N_.item())
            sir_noisy[t_idx, city_idx, 0] = S_noisy
            sir_noisy[t_idx, city_idx, 1] = I_noisy
            sir_noisy[t_idx, city_idx, 2] = R_noisy

    return time_points, weather_data, sir_noisy, total_households

##############################################################################
# D) Plotting
##############################################################################

def plot_sir(time_points, sir_data, total_households, label_prefix, outfolder, seed, mode="train"):
    """
    Plot each city's infected trajectory, plus a combined figure.
    label_prefix: e.g. "train" or "test"
    mode: string "train" or "test"
    """
    # folder for plots
    plot_dir = os.path.join(outfolder, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    T, C, _ = sir_data.shape
    time_np = time_points.numpy()

    # Individual city plots
    for city_idx in range(C):
        infected_np = sir_data[:, city_idx, 1].numpy()
        city_pop = total_households[city_idx].item()
        # SAIDI ~ sum(infected)/pop
        saidi_true = np.sum(infected_np)/city_pop

        plt.figure(figsize=(8,6))
        plt.plot(time_np, infected_np, label=f"Infected (SAIDI: {saidi_true:.1f})", color="tab:blue")
        plt.fill_between(time_np, infected_np, color='lightblue', alpha=0.3)
        plt.xlabel("Time")
        plt.ylabel("No. Infected Households")
        plt.title(f"{label_prefix.capitalize()} {mode.capitalize()} - City {city_idx+1}")
        plt.legend(loc='upper right')
        plt.grid(True, linestyle='--', alpha=0.5)
        filename_city = os.path.join(plot_dir, f"{label_prefix}_{mode}_city{city_idx+1}_seed{seed}.pdf")
        plt.savefig(filename_city, format='pdf', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved {filename_city}")

    # Combined plot
    plt.figure(figsize=(10,6))
    for city_idx in range(C):
        infected_np = sir_data[:, city_idx, 1].numpy()
        plt.plot(time_np, infected_np, label=f"City{city_idx+1}")
    plt.xlabel("Time")
    plt.ylabel("Infected")
    plt.title(f"{label_prefix.capitalize()} {mode.capitalize()} - All Cities")
    plt.legend()
    filename_combined = os.path.join(plot_dir, f"{label_prefix}_{mode}_combined_seed{seed}.pdf")
    plt.savefig(filename_combined, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {filename_combined}")

##############################################################################
# E) Main: Generate TRAIN + TEST in one run
##############################################################################

def main():
    parser = argparse.ArgumentParser(description="Generate train + test synthetic data for SIR problem.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_cities", type=int, default=3)
    parser.add_argument("--t_start", type=float, default=0.)
    parser.add_argument("--t_end", type=float, default=50.)
    parser.add_argument("--t_steps", type=int, default=51)
    parser.add_argument("--outfolder", type=str, default="synthetic_power_data",
                        help="folder to save .pt data and plots/")
    args = parser.parse_args()

    # Make sure folder exists
    os.makedirs(args.outfolder, exist_ok=True)

    # 1) TRAIN parameters
    beta_train = torch.tensor([0.5, 0.4, 0.2])[:args.num_cities]
    gamma_train= torch.tensor([0.2, 0.15, 0.1])[:args.num_cities]

    # 2) TEST parameters
    beta_test  = torch.tensor([0.45, 0.35, 0.25])[:args.num_cities]
    gamma_test = torch.tensor([0.18, 0.14, 0.12])[:args.num_cities]

    # Generate "train" set
    time_train, weather_train, sir_train, pops_train = generate_sir_data(
        args.num_cities, args.seed, args.t_start, args.t_end, args.t_steps,
        beta_base=beta_train, gamma_base=gamma_train
    )

    # Save train to disk
    train_file = os.path.join(args.outfolder, f"synthetic_data_seed{args.seed}_train.pt")
    torch.save({
        "time_points": time_train,
        "weather_data": weather_train,
        "sir_data": sir_train,
        "total_households": pops_train
    }, train_file)
    print(f"Saved TRAIN data to {train_file}")

    # Plot train
    plot_sir(time_train, sir_train, pops_train, label_prefix="train", outfolder=args.outfolder, seed=args.seed, mode="train")

    # Generate "test" set
    time_test, weather_test, sir_test, pops_test = generate_sir_data(
        args.num_cities, args.seed+999, args.t_start, args.t_end, args.t_steps,
        beta_base=beta_test, gamma_base=gamma_test
    )
    # Notice: we used "args.seed+999" for variety. Or keep same seed if you prefer identical weather, etc.

    # Save test to disk
    test_file = os.path.join(args.outfolder, f"synthetic_data_seed{args.seed}_test.pt")
    torch.save({
        "time_points": time_test,
        "weather_data": weather_test,
        "sir_data": sir_test,
        "total_households": pops_test
    }, test_file)
    print(f"Saved TEST data to {test_file}")

    # Plot test
    plot_sir(time_test, sir_test, pops_test, label_prefix="test", outfolder=args.outfolder, seed=args.seed, mode="test")

    print("Done generating train + test data + plots.")

if __name__ == "__main__":
    main()
