import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class DiffusionMLP(nn.Module):
    """Simple MLP for diffusion model"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

class ConditionalDiffusionModel(nn.Module):
    def __init__(self, x_dim, c_dim, hidden_dim=128, num_steps=1000):
        super().__init__()
        self.x_dim = x_dim  # dimension of condition x
        self.c_dim = c_dim  # dimension of target c
        self.num_steps = num_steps
        
        # Time embeddings
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Condition embedding (for x)
        self.condition_mlp = nn.Sequential(
            nn.Linear(x_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Main network - predicts noise in c given noisy c, condition x, and time
        self.net = DiffusionMLP(
            input_dim=c_dim + hidden_dim + hidden_dim,  # noisy c + time embedding + condition embedding
            hidden_dim=hidden_dim,
            output_dim=c_dim
        )
        
        # Beta schedule
        self.register_buffer('beta', torch.linspace(1e-4, 0.02, num_steps))
        self.register_buffer('alpha', 1 - self.beta)
        self.register_buffer('alpha_bar', torch.cumprod(self.alpha, dim=0))

    def forward(self, noisy_c, t, x):
        """
        Forward pass of the model
        Args:
            noisy_c: [B, C_dim] noisy target data
            t: [B] timesteps
            x: [B, X_dim] conditioning information
        """
        # Get embeddings
        time_emb = self.time_mlp(t)
        cond_emb = self.condition_mlp(x)
        
        # Concatenate all inputs
        model_input = torch.cat([noisy_c, time_emb, cond_emb], dim=1)
        
        # Predict noise
        return self.net(model_input)

    def get_loss(self, c, x, classifier_free_guidance_scale=3.0, null_condition_prob=0.1):
        """
        Compute the loss with classifier-free guidance
        Args:
            c: [B, C_dim] target data to model
            x: [B, X_dim] conditioning data
        """
        batch_size = c.shape[0]
        t = torch.randint(0, self.num_steps, (batch_size,), device=c.device)
        
        # Sample noise
        noise = torch.randn_like(c)
        alpha_bar_t = self.alpha_bar[t].unsqueeze(1)
        noisy_c = torch.sqrt(alpha_bar_t) * c + torch.sqrt(1 - alpha_bar_t) * noise
        
        # Create condition mask for classifier-free guidance
        condition_mask = torch.rand(batch_size) > null_condition_prob
        masked_x = torch.where(
            condition_mask.unsqueeze(1),
            x,
            torch.zeros_like(x)
        )
        
        # Predict noise
        noise_pred = self(noisy_c, t, masked_x)
        
        # Compute loss with classifier-free guidance
        if self.training:
            noise_pred_uncond = self(noisy_c, t, torch.zeros_like(x))
            noise_pred = (1 + classifier_free_guidance_scale) * noise_pred - classifier_free_guidance_scale * noise_pred_uncond
        
        loss = F.mse_loss(noise_pred, noise)
        return loss

    @torch.no_grad()
    def sample(self, num_samples, x, temperature=1.0):
        """
        Generate samples from p(c|x) using the diffusion model
        Args:
            num_samples: number of samples to generate
            x: [B, X_dim] conditioning data
        Returns:
            samples: [B, num_samples, C_dim]
        """
        device = next(self.parameters()).device
        batch_size = x.shape[0]
        
        # Expand x for multiple samples
        x_expanded = x.unsqueeze(1).expand(-1, num_samples, -1)
        x_expanded = x_expanded.reshape(-1, self.x_dim)
        
        # Start from random noise
        c = torch.randn(batch_size * num_samples, self.c_dim, device=device) * temperature
        
        # Reverse diffusion process
        for t in reversed(range(self.num_steps)):
            t_batch = torch.full((batch_size * num_samples,), t, device=device, dtype=torch.long)
            
            # Predict noise
            noise_pred = self(c, t_batch, x_expanded)
            
            alpha_t = self.alpha[t]
            alpha_bar_t = self.alpha_bar[t]
            beta_t = self.beta[t]
            
            # Add noise if not the last step
            if t > 0:
                noise = torch.randn_like(c) * temperature
            else:
                noise = 0
                
            c = (1 / torch.sqrt(alpha_t)) * (
                c - (beta_t / (torch.sqrt(1 - alpha_bar_t))) * noise_pred
            ) + torch.sqrt(beta_t) * noise
            
        # Reshape to [batch_size, num_samples, c_dim]
        samples = c.reshape(batch_size, num_samples, self.c_dim)
        return samples