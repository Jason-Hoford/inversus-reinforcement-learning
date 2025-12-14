"""Neural network policies for INVERSUS RL."""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

from .env_wrappers import SingleInversusRLEnv


class InversusCNNPolicy(nn.Module):
    """CNN-based policy and value network for INVERSUS with deeper architecture."""
    
    def __init__(self, channels: int, height: int, width: int, extra_dim: int, hidden_dim: int = 256):
        """
        Initialize policy network.
        
        Args:
            channels: Number of input channels (6)
            height: Grid height
            width: Grid width
            extra_dim: Extra feature dimension (4)
            hidden_dim: Hidden layer dimension (increased to 256)
        """
        super().__init__()
        
        # Deeper CNN with residual-style connections
        # Layer 1: Input → 32 channels
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=3, padding=1)
        self.norm1 = nn.LayerNorm([32, height, width])
        
        # Layer 2: 32 → 64 channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.norm2 = nn.LayerNorm([64, height, width])
        
        # Layer 3: 64 → 128 channels
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.norm3 = nn.LayerNorm([128, height, width])
        
        # Layer 4: 128 → 128 channels (with residual)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.norm4 = nn.LayerNorm([128, height, width])
        
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        
        # Compute conv output size
        with torch.no_grad():
            dummy_input = torch.zeros(1, channels, height, width)
            x = self.relu(self.norm1(self.conv1(dummy_input)))
            x = self.relu(self.norm2(self.conv2(x)))
            x = self.relu(self.norm3(self.conv3(x)))
            x_res = x  # Save for residual
            x = self.conv4(x)
            x = self.relu(self.norm4(x + x_res))  # Residual connection
            conv_out = self.flatten(x)
            conv_out_dim = conv_out.shape[1]
        
        # Actor head (policy) - larger hidden dimension
        self.fc_actor = nn.Sequential(
            nn.Linear(conv_out_dim + extra_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 13)  # 13 discrete actions
        )
        
        # Critic head (value) - larger hidden dimension  
        self.fc_critic = nn.Sequential(
            nn.Linear(conv_out_dim + extra_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, grid_tensor: torch.Tensor, extra_vector: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            grid_tensor: (B, C, H, W) batch of grid observations
            extra_vector: (B, extra_dim) batch of extra features
            
        Returns:
            (logits, value) where:
            - logits: (B, 13) action logits
            - value: (B, 1) value estimates
        """
        # Process grid through deeper CNN with residual connection
        x = self.relu(self.norm1(self.conv1(grid_tensor)))
        x = self.relu(self.norm2(self.conv2(x)))
        x = self.relu(self.norm3(self.conv3(x)))
        x_res = x  # Save for residual
        x = self.conv4(x)
        x = self.relu(self.norm4(x + x_res))  # Residual connection
        
        # Flatten
        x = self.flatten(x)
        
        # Concatenate with extra features
        x = torch.cat([x, extra_vector], dim=1)
        
        # Get action logits and value
        logits = self.fc_actor(x)
        value = self.fc_critic(x)
        
        return logits, value


def make_policy_from_env(env: SingleInversusRLEnv) -> InversusCNNPolicy:
    """
    Construct a policy network from environment observation shapes.
    
    Args:
        env: SingleInversusRLEnv instance
        
    Returns:
        InversusCNNPolicy with correct input dimensions
    """
    # Get observation shape by resetting
    obs = env.reset()
    grid_tensor, extra_vector = obs
    
    channels, height, width = grid_tensor.shape
    extra_dim = extra_vector.shape[0]
    
    return InversusCNNPolicy(channels, height, width, extra_dim)


