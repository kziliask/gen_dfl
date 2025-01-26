# #!/usr/bin/env python3

# """
# end2end_cflowdfl_harden.py

# End-to-end cFlow approach:
#  - Loads a "train" file, trains a conditional flow + DFL loss
#  - Loads a "test" file, evaluates
# """

# import torch
# import torch.nn as nn
# import numpy as np
# import argparse
# import os
# import cvxpy as cp
# from cvxpylayers.torch import CvxpyLayer
# import matplotlib.pyplot as plt

# ##############################################################################
# # 1) cFlow Model
# ##############################################################################

# class SimpleConditionalFlow(nn.Module):
#     """
#     Minimal example flow: we produce shift, log_scale from weather -> affine transform for sir.
#     """
#     def __init__(self, sir_dim=3, weather_dim=3, hidden_dim=32):
#         super().__init__()
#         self.sir_dim = sir_dim
#         self.weather_dim = weather_dim
#         self.net = nn.Sequential(
#             nn.Linear(weather_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, 2*sir_dim)
#         )

#     def forward(self, sir, weather):
#         """
#         sir: shape (batch, sir_dim)
#         weather: shape (batch, weather_dim)
#         returns z, log_det
#         """
#         out = self.net(weather)  # (batch, 2*sir_dim)
#         shift, log_scale = torch.split(out, self.sir_dim, dim=1)
#         z = (sir - shift) * torch.exp(-log_scale)
#         log_det = -torch.sum(log_scale, dim=1)
#         return z, log_det

#     def sample(self, n_samples, weather):
#         """
#         Sample from the flow, given weather.
#         returns: shape (batch, n_samples, sir_dim)
#         """
#         B = weather.shape[0]
#         out = self.net(weather)
#         shift, log_scale = torch.split(out, self.sir_dim, dim=1)
#         # standard normal for z
#         z = torch.randn(B, n_samples, self.sir_dim, device=weather.device)
#         shift = shift.unsqueeze(1)      # (B,1,sir_dim)
#         log_scale = log_scale.unsqueeze(1)  # (B,1,sir_dim)
#         sir_samples = z * torch.exp(log_scale) + shift
#         return sir_samples

# ##############################################################################
# # 2) CVXPY layer
# ##############################################################################

# def create_cvxpy_layer(num_cities=1, lda=0.01, max_invest=1.0):
#     SAIDI_param = cp.Parameter(num_cities)
#     x_var = cp.Variable(num_cities, nonneg=True)
#     objective = cp.Maximize(SAIDI_param @ x_var - lda * cp.sum_squares(x_var))
#     constraints = [cp.sum(x_var) <= max_invest, x_var <= 1]
#     problem = cp.Problem(objective, constraints)
#     layer = CvxpyLayer(problem, parameters=[SAIDI_param], variables=[x_var])
#     return layer

# ##############################################################################
# # 3) Train cFlow + DFL
# ##############################################################################

# def train_end2end_cflow_dfl(cflow, weather_data, sir_data, total_households,
#                             cvxpylayer, n_epochs=10, n_samples=50):
#     optimizer = torch.optim.Adam(cflow.parameters(), lr=0.001)
#     T, C, _ = sir_data.shape

#     # Flatten time + city => (T*C) samples
#     X_list, Y_list = [], []
#     for t in range(T):
#         for i in range(C):
#             X_list.append(weather_data[t, i])
#             Y_list.append(sir_data[t, i])
#     X_cat = torch.stack(X_list, dim=0)  # shape (T*C, 3)
#     Y_cat = torch.stack(Y_list, dim=0)  # shape (T*C, 3)

#     all_losses = []
#     for epoch in range(n_epochs):
#         # Negative log-likelihood
#         z, log_det = cflow(Y_cat, X_cat)
#         log_prob = -0.5*torch.sum(z**2, dim=1) - 0.5*z.shape[1]*np.log(2*np.pi)
#         nll_loss = -(log_prob + log_det).mean()

#         # DFL portion
#         dfl = 0.0
#         # sample from cFlow
#         samples = cflow.sample(n_samples, X_cat)  # shape (T*C, n_samples, 3)
#         # We'll do a single "mean param" approach: mean_sir = average across samples
#         mean_sir = samples.mean(dim=1)  # shape (T*C, 3)
        
#         B = X_cat.shape[0]
#         for b_idx in range(B):
#             infected_b = mean_sir[b_idx, 1]
#             city_idx = b_idx % C
#             city_pop = total_households[city_idx]
#             param_val = (infected_b / city_pop).unsqueeze(0)
#             x_opt, = cvxpylayer(param_val)
#             # negative objective
#             dfl += - (param_val * x_opt)
#         dfl = dfl / B

#         loss = nll_loss + dfl
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         all_losses.append((nll_loss.item(), dfl.item()))
#         print(f"[train] Epoch {epoch+1}/{n_epochs} => NLL={nll_loss.item():.4f}, DFL={dfl.item():.4f}")

#     return all_losses, cflow

# ##############################################################################
# # 4) Evaluate cFlow model on test
# ##############################################################################

# def evaluate_cflow(cflow, weather_data, sir_data):
#     """
#     We'll measure negative log-likelihood as a simple metric on test data.
#     (We could also measure a final DFL objective again or do a separate step.)
#     """
#     MSE = nn.MSELoss()
#     T, C, _ = sir_data.shape
#     X_list, Y_list = [], []
#     for t in range(T):
#         for i in range(C):
#             X_list.append(weather_data[t, i])
#             Y_list.append(sir_data[t, i])
#     X_cat = torch.stack(X_list, dim=0)
#     Y_cat = torch.stack(Y_list, dim=0)

#     with torch.no_grad():
#         z, log_det = cflow(Y_cat, X_cat)
#         log_prob = -0.5*torch.sum(z**2, dim=1) - 0.5*z.shape[1]*np.log(2*np.pi)
#         nll = -(log_prob + log_det).mean().item()

#         # We can also do a quick MSE check on mean predictions vs. Y. 
#         # For instance, if we define "predict" as the shift from net, ignoring scale? 
#         # We'll do a naive approach: 
#         out = cflow.net(X_cat)  # shape (batch, 2*sir_dim)
#         shift, _ = torch.split(out, cflow.sir_dim, dim=1)
#         # shift is effectively the "mean" of distribution
#         test_mse = MSE(shift, Y_cat).item()

#     return nll, test_mse

# ##############################################################################
# # 5) Main: load train + test, train, evaluate
# ##############################################################################

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--train_file", type=str, required=True)
#     parser.add_argument("--test_file", type=str, required=True)
#     parser.add_argument("--epochs", type=int, default=10)
#     args = parser.parse_args()

#     # Load train
#     train_path = os.path.join("synthetic_power_data", args.train_file) \
#         if not os.path.exists(args.train_file) else args.train_file
#     train_data = torch.load(train_path)
#     train_weather = train_data["weather_data"]
#     train_sir = train_data["sir_data"]
#     train_households = train_data["total_households"]
#     print(f"Loaded TRAIN from {train_path}")

#     # Load test
#     test_path = os.path.join("synthetic_power_data", args.test_file) \
#         if not os.path.exists(args.test_file) else args.test_file
#     test_data = torch.load(test_path)
#     test_weather = test_data["weather_data"]
#     test_sir = test_data["sir_data"]
#     test_households = test_data["total_households"]
#     print(f"Loaded TEST from {test_path}")

#     # Build cFlow
#     cflow = SimpleConditionalFlow(sir_dim=3, weather_dim=3, hidden_dim=32)
#     cvxpylayer = create_cvxpy_layer(num_cities=1, lda=0.01, max_invest=1.0)

#     # Train
#     all_losses, cflow = train_end2end_cflow_dfl(
#         cflow, train_weather, train_sir, train_households,
#         cvxpylayer, n_epochs=args.epochs, n_samples=50
#     )

#     # Plot training losses
#     nll_vals, dfl_vals = zip(*all_losses)
#     plt.figure()
#     plt.plot(nll_vals, label="NLL")
#     plt.plot(dfl_vals, label="DFL")
#     plt.legend()
#     plt.title("End2End cFlow DFL Hardening - Training")
#     plt.savefig("end2end_cflowdfl_train_losses.png")
#     plt.close()
#     print("Saved end2end_cflowdfl_train_losses.png")

#     # Evaluate on test
#     test_nll, test_mse = evaluate_cflow(cflow, test_weather, test_sir)
#     print(f"[Test] NLL on test set: {test_nll:.4f}, MSE: {test_mse:.4f}")

# if __name__ == "__main__":
#     main()


