"""PPO (Proximal Policy Optimization) agent for INVERSUS."""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Dict, Any
from collections import deque

from .policies import InversusCNNPolicy


class PPOAgent:
    """PPO agent for training policies."""
    
    def __init__(
        self,
        policy: nn.Module,
        lr: float = 1e-4,  # Reduced from 3e-4 for more stable learning
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_ratio: float = 0.2,
        epochs: int = 4,  # Reduced from 10 to prevent overfitting/policy collapse
        batch_size: int = 512,
        entropy_coef: float = 0.02,  # Reduced from 0.05 to allow convergence
        value_coef: float = 0.1,  # Reduced from 0.5 so massive value gradients don't wreck shared CNN
        device: str = "cpu"
    ):
        """
        Initialize PPO agent.
        
        Args:
            policy: Policy network
            lr: Learning rate (tuned to 1e-4)
            gamma: Discount factor (tuned to 0.99)
            lam: GAE lambda
            clip_ratio: PPO clip ratio
            epochs: Number of update epochs (tuned to 4 for stability)
            batch_size: Batch size for updates (tuned to 512)
            entropy_coef: Entropy bonus coefficient (tuned to 0.02)
            value_coef: Value loss coefficient (tuned to 0.1)
            device: Device to run on ("cpu" or "cuda")
        """
        self.policy = policy.to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.lam = lam
        self.clip_ratio = clip_ratio
        self.epochs = epochs
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.device = device
        
        # Rollout buffers
        self.reset_buffers()
    
    def reset_buffers(self) -> None:
        """Reset rollout buffers."""
        self.obs_grid_buffer: List[np.ndarray] = []
        self.obs_extra_buffer: List[np.ndarray] = []
        self.action_buffer: List[int] = []
        self.log_prob_buffer: List[float] = []
        self.reward_buffer: List[float] = []
        self.value_buffer: List[float] = []
        self.done_buffer: List[bool] = []
    
    def act(
        self, 
        grid_tensors: np.ndarray, 
        extra_vectors: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample actions from policy.
        
        Args:
            grid_tensors: (B, C, H, W) batch of grid observations
            extra_vectors: (B, extra_dim) batch of extra features
            
        Returns:
            (actions, log_probs, values) where:
            - actions: (B,) array of action IDs
            - log_probs: (B,) tensor of log probabilities
            - values: (B,) tensor of value estimates
        """
        self.policy.eval()
        
        with torch.no_grad():
            # Convert to tensors
            grid_t = torch.FloatTensor(grid_tensors).to(self.device)
            extra_t = torch.FloatTensor(extra_vectors).to(self.device)
            
            # Forward pass
            logits, values = self.policy(grid_t, extra_t)
            
            # Sample actions
            dist = torch.distributions.Categorical(logits=logits)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)
            
            # Convert to numpy
            actions_np = actions.cpu().numpy()
            log_probs_np = log_probs.cpu().numpy()
            values_np = values.squeeze(-1).cpu().numpy()
        
        return actions_np, log_probs_np, values_np
    
    def store_step(
        self,
        grid_tensor: np.ndarray,
        extra_vector: np.ndarray,
        action: int,
        log_prob: float,
        value: float,
        reward: float,
        done: bool
    ) -> None:
        """Store a single timestep in the rollout buffer."""
        self.obs_grid_buffer.append(grid_tensor)
        self.obs_extra_buffer.append(extra_vector)
        self.action_buffer.append(action)
        self.log_prob_buffer.append(log_prob)
        self.reward_buffer.append(reward)
        self.value_buffer.append(value)
        self.done_buffer.append(done)
    
    def compute_advantages(self, last_value: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute advantages and returns using GAE.
        
        Args:
            last_value: Value estimate for terminal state
            
        Returns:
            (advantages, returns) as numpy arrays
        """
        rewards = np.array(self.reward_buffer, dtype=np.float32)
        values = np.array(self.value_buffer + [last_value], dtype=np.float32)
        dones = np.array(self.done_buffer, dtype=bool)
        
        advantages = np.zeros_like(rewards)
        last_gae = 0.0
        
        # Compute GAE backwards
        for t in reversed(range(len(rewards))):
            if dones[t]:
                delta = rewards[t] - values[t]
                last_gae = delta
            else:
                delta = rewards[t] + self.gamma * values[t + 1] - values[t]
                last_gae = delta + self.gamma * self.lam * last_gae
            advantages[t] = last_gae
        
        # Compute returns
        returns = advantages + values[:-1]
        
        return advantages, returns
    
    def update(self) -> Dict[str, float]:
        """
        Run PPO update over collected rollout.
        
        Returns:
            Dictionary of training statistics
        """
        if len(self.obs_grid_buffer) == 0:
            return {}
        
        # Compute advantages and returns
        advantages, returns = self.compute_advantages()
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convert to tensors
        obs_grid = torch.FloatTensor(np.stack(self.obs_grid_buffer)).to(self.device)
        obs_extra = torch.FloatTensor(np.stack(self.obs_extra_buffer)).to(self.device)
        actions = torch.LongTensor(self.action_buffer).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_prob_buffer).to(self.device)
        advantages_t = torch.FloatTensor(advantages).to(self.device)
        returns_t = torch.FloatTensor(returns).to(self.device)
        
        # Training mode
        self.policy.train()
        
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        
        # Multiple epochs
        num_samples = len(self.obs_grid_buffer)
        indices = np.arange(num_samples)
        
        for epoch in range(self.epochs):
            np.random.shuffle(indices)
            
            # Mini-batch updates
            for start in range(0, num_samples, self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                # Get batch
                batch_grid = obs_grid[batch_indices]
                batch_extra = obs_extra[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages_t[batch_indices]
                batch_returns = returns_t[batch_indices]
                
                # Forward pass
                logits, values = self.policy(batch_grid, batch_extra)
                dist = torch.distributions.Categorical(logits=logits)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                # PPO clipped surrogate loss
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = nn.functional.mse_loss(values.squeeze(-1), batch_returns)
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
        
        # Reset buffers
        self.reset_buffers()
        
        num_updates = self.epochs * (num_samples // self.batch_size + (1 if num_samples % self.batch_size else 0))
        
        return {
            "policy_loss": total_policy_loss / num_updates,
            "value_loss": total_value_loss / num_updates,
            "entropy": total_entropy / num_updates,
        }


