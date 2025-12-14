"""Training scripts for INVERSUS RL."""

import os
import csv
import time
import argparse
import numpy as np
import torch

from .env_wrappers import MultiEnvRunner, SingleInversusRLEnv, build_observation
from .policies import make_policy_from_env, InversusCNNPolicy
from .ppo_agent import PPOAgent
from inversus.game_types import PlayerId


class TrainingLogger:
    """Logger for training statistics."""
    
    def __init__(self, log_dir: str):
        """Initialize logger."""
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self.csv_path = os.path.join(log_dir, "training_log.csv")
        
        # Write header
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "step", "episode", "avg_reward", "win_rate", "avg_ep_len",
                "policy_loss", "value_loss", "entropy"
            ])
    
    def log(
        self,
        step: int,
        episode: int,
        avg_reward: float,
        win_rate: float,
        avg_ep_len: float,
        policy_loss: float = 0.0,
        value_loss: float = 0.0,
        entropy: float = 0.0
    ) -> None:
        """Log training statistics."""
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                step, episode, avg_reward, win_rate, avg_ep_len,
                policy_loss, value_loss, entropy
            ])


def train_vs_dummy(
    num_envs: int = 1, 
    total_steps: int = 500_000, 
    log_dir: str = "runs/inversus_vs_dummy", 
    opponent_difficulty: str = "easy",
    load_model: str = None
):
    """
    Train agent against dummy opponent.
    
    Args:
        num_envs: Number of parallel environments
        total_steps: Total training steps
        log_dir: Directory for logs and model saves
        opponent_difficulty: Difficulty of dummy opponent
        load_model: Path to pretrained model to load
    """
    print(f"Training vs dummy opponent with num_envs={num_envs}")
    print(f"Opponent Difficulty: {opponent_difficulty}")
    print(f"Total steps: {total_steps}")
    print(f"Log directory: {log_dir}")
    
    # Create environments
    env_runner = MultiEnvRunner(num_envs=num_envs, opponent_type="dummy", max_episode_steps=500, difficulty=opponent_difficulty)
    
    # Create policy
    sample_env = SingleInversusRLEnv(opponent_type="dummy", difficulty=opponent_difficulty)
    policy = make_policy_from_env(sample_env)
    
    # Load pretrained model if specified
    if load_model:
        print(f"Loading pretrained model from: {load_model}")
        try:
            policy.load_state_dict(torch.load(load_model))
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            return

    # Create agent (uses default lr=1e-4 from PPOAgent)
    agent = PPOAgent(policy)
    
    # Create logger
    logger = TrainingLogger(log_dir)
    
    # Training loop
    obs_grid, obs_extra = env_runner.reset()
    step_count = 0
    last_log_step = 0
    episode_count = 0
    
    # STABILITY FIX: Ensure large enough batch size regardless of num_envs
    target_steps_per_update = 2048
    steps_per_env = max(target_steps_per_update // num_envs, 128)
    steps_per_update = steps_per_env * num_envs
    
    episode_rewards = []
    episode_lengths = []
    episode_wins = []
    
    print("Starting training...")
    print(f"Collecting {steps_per_update} steps per update (Batch size: 512)...")
    start_time = time.time()
    
    while step_count < total_steps:
        # Collect rollout
        for _ in range(steps_per_env):
            # Agent acts
            actions, log_probs, values = agent.act(obs_grid, obs_extra)
            
            # Environment steps
            next_obs, rewards, dones, infos = env_runner.step(actions)
            next_obs_grid, next_obs_extra = next_obs
            
            # Store transitions
            for i in range(num_envs):
                agent.store_step(
                    obs_grid[i],
                    obs_extra[i],
                    int(actions[i]),
                    float(log_probs[i]),
                    float(values[i]),
                    float(rewards[i]),
                    bool(dones[i])
                )
                
                # Track episode stats
                if dones[i]:
                    episode_count += 1
                    episode_rewards.append(infos[i].get("episode_return", 0.0))
                    episode_lengths.append(infos[i].get("episode_steps", 0))
                    if infos[i].get("win", False):
                        episode_wins.append(1)
                    else:
                        episode_wins.append(0)
                    # Reset environment
                    reset_obs = env_runner.envs[i].reset()
                    next_obs_grid[i] = reset_obs[0]
                    next_obs_extra[i] = reset_obs[1]
            
            obs_grid, obs_extra = next_obs_grid, next_obs_extra
            step_count += num_envs
            
            if step_count >= total_steps:
                break
        
        # Update policy
        update_stats = agent.update()
        
        # Logging
        if episode_count > 0 and len(episode_rewards) > 0:
            recent_rewards = episode_rewards[-100:] if len(episode_rewards) >= 100 else episode_rewards
            recent_lengths = episode_lengths[-100:] if len(episode_lengths) >= 100 else episode_lengths
            recent_wins = episode_wins[-100:] if len(episode_wins) >= 100 else episode_wins
            
            avg_reward = np.mean(recent_rewards)
            win_rate = np.mean(recent_wins)
            avg_ep_len = np.mean(recent_lengths)
            
            if step_count - last_log_step >= 1000 or step_count >= total_steps:
                last_log_step = step_count
                logger.log(
                    step=step_count,
                    episode=episode_count,
                    avg_reward=avg_reward,
                    win_rate=win_rate,
                    avg_ep_len=avg_ep_len,
                    policy_loss=update_stats.get("policy_loss", 0.0),
                    value_loss=update_stats.get("value_loss", 0.0),
                    entropy=update_stats.get("entropy", 0.0)
                )
                elapsed = time.time() - start_time
                print(f"Step {step_count}/{total_steps} | "
                      f"Episodes: {episode_count} | "
                      f"Avg Reward: {avg_reward:.3f} | "
                      f"Win Rate: {win_rate:.3f} | "
                      f"Avg Ep Len: {avg_ep_len:.1f} | "
                      f"Time: {elapsed:.1f}s")
            
            # Save checkpoints every 50K steps
            if step_count % 50000 == 0 and step_count > 0:
                checkpoint_path = os.path.join(log_dir, f"policy_checkpoint_{step_count}.pt")
                torch.save(policy.state_dict(), checkpoint_path)
                print(f"Checkpoint saved to {checkpoint_path}")
    
    # Save final model
    model_path = os.path.join(log_dir, "policy_final.pt")
    torch.save(policy.state_dict(), model_path)
    print(f"Final model saved to {model_path}")


def train_selfplay(num_envs: int = 1, total_steps: int = 500_000, log_dir: str = "runs/inversus_selfplay", load_model: str = None):
    """
    Train agent via self-play.
    
    Args:
        num_envs: Number of parallel environments
        total_steps: Total training steps
        log_dir: Directory for logs and model saves
        load_model: Path to pretrained model to load
    """
    print(f"Training via self-play with num_envs={num_envs}")
    print(f"Total steps: {total_steps}")
    print(f"Log directory: {log_dir}")
    print(f"Starting Model: {load_model if load_model else 'Random Initialization'}")
    
    # Create environments
    env_runner = MultiEnvRunner(num_envs=num_envs, opponent_type="selfplay", max_episode_steps=500)
    
    # Create policy
    sample_env = SingleInversusRLEnv(opponent_type="selfplay")
    policy = make_policy_from_env(sample_env)
    
    # Load pretrained model if provided
    if load_model:
        print(f"Loading pretrained model from {load_model}...")
        try:
            state_dict = torch.load(load_model, map_location="cpu")
            policy.load_state_dict(state_dict)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            return

    # Create target policy (Opponent)
    target_policy = make_policy_from_env(sample_env)
    target_policy.load_state_dict(policy.state_dict()) # Initial opponent is clone of self
    target_policy.eval()
    
    # Create agent (uses default lr=1e-4 from PPOAgent)
    agent = PPOAgent(policy)
    
    # Create logger
    logger = TrainingLogger(log_dir)
    
    # Helper function for opponent policy
    def opponent_policy(obs):
        """Get action from target policy for opponent."""
        grid_tensor, extra_vector = obs
        with torch.no_grad():
            grid_t = torch.FloatTensor(grid_tensor).unsqueeze(0)
            extra_t = torch.FloatTensor(extra_vector).unsqueeze(0)
            logits, _ = target_policy(grid_t, extra_t)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            return action.item()
    
    # Training loop
    obs_grid, obs_extra = env_runner.reset()
    step_count = 0
    last_log_step = 0
    last_opponent_update_step = 0
    OPPONENT_UPDATE_FREQ = 20000 # Update opponent every 20k steps
    
    episode_count = 0
    
    # STABILITY FIX: Ensure large enough batch size regardless of num_envs
    # PPOAgent uses batch_size=512. We want at least 4 batches per update = 2048.
    # If num_envs=2, we need 1024 steps per env.
    target_steps_per_update = 2048
    steps_per_env = max(target_steps_per_update // num_envs, 128)
    steps_per_update = steps_per_env * num_envs  # Total steps collected per update loop
    
    episode_rewards = []
    episode_lengths = []
    episode_wins = []
    
    print("Starting training...")
    print(f"Collecting {steps_per_update} steps per update (Batch size: 512)...")
    start_time = time.time()
    
    while step_count < total_steps:
        # Collect rollout
        # We need to run steps_per_env iterations, collecting num_envs steps each time
        for _ in range(steps_per_env):
            # Agent acts
            actions, log_probs, values = agent.act(obs_grid, obs_extra)
            
            # Environment steps with opponent policy
            next_obs, rewards, dones, infos = env_runner.step(actions, opponent_policy=opponent_policy)
            next_obs_grid, next_obs_extra = next_obs
            
            # Store transitions
            for i in range(num_envs):
                agent.store_step(
                    obs_grid[i],
                    obs_extra[i],
                    int(actions[i]),
                    float(log_probs[i]),
                    float(values[i]),
                    float(rewards[i]),
                    bool(dones[i])
                )
                
                # Track episode stats
                if dones[i]:
                    episode_count += 1
                    episode_rewards.append(infos[i].get("episode_return", 0.0))
                    episode_lengths.append(infos[i].get("episode_steps", 0))
                    if infos[i].get("win", False):
                        episode_wins.append(1)
                    else:
                        episode_wins.append(0)
                    # Reset environment
                    reset_obs = env_runner.envs[i].reset()
                    next_obs_grid[i] = reset_obs[0]
                    next_obs_extra[i] = reset_obs[1]
            
            obs_grid, obs_extra = next_obs_grid, next_obs_extra
            step_count += num_envs
            
            if step_count >= total_steps:
                break
        
        # Update policy
        update_stats = agent.update()
        
        # Self-Play: Check for opponent update
        if step_count - last_opponent_update_step >= OPPONENT_UPDATE_FREQ:
            print(f"Updating opponent policy at step {step_count}...")
            target_policy.load_state_dict(policy.state_dict())
            last_opponent_update_step = step_count
        
        # Logging
        if episode_count > 0 and len(episode_rewards) > 0:
            recent_rewards = episode_rewards[-100:] if len(episode_rewards) >= 100 else episode_rewards
            recent_lengths = episode_lengths[-100:] if len(episode_lengths) >= 100 else episode_lengths
            recent_wins = episode_wins[-100:] if len(episode_wins) >= 100 else episode_wins
            
            avg_reward = np.mean(recent_rewards)
            win_rate = np.mean(recent_wins)
            avg_ep_len = np.mean(recent_lengths)
            
            if step_count - last_log_step >= 1000 or step_count >= total_steps:
                last_log_step = step_count
                logger.log(
                    step=step_count,
                    episode=episode_count,
                    avg_reward=avg_reward,
                    win_rate=win_rate,
                    avg_ep_len=avg_ep_len,
                    policy_loss=update_stats.get("policy_loss", 0.0),
                    value_loss=update_stats.get("value_loss", 0.0),
                    entropy=update_stats.get("entropy", 0.0)
                )
                elapsed = time.time() - start_time
                print(f"Step {step_count}/{total_steps} | "
                      f"Episodes: {episode_count} | "
                      f"Avg Reward: {avg_reward:.3f} | "
                      f"Win Rate: {win_rate:.3f} | "
                      f"Avg Ep Len: {avg_ep_len:.1f} | "
                      f"Time: {elapsed:.1f}s")
            
            # Save checkpoints every 50K steps
            if step_count % 50000 == 0 and step_count > 0:
                checkpoint_path = os.path.join(log_dir, f"policy_checkpoint_{step_count}.pt")
                torch.save(policy.state_dict(), checkpoint_path)
                print(f"Checkpoint saved to {checkpoint_path}")
    
    # Save final model
    model_path = os.path.join(log_dir, "policy_final.pt")
    torch.save(policy.state_dict(), model_path)
    print(f"Final model saved to {model_path}")


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(description="Train INVERSUS RL agent")
    parser.add_argument("--mode", choices=["vs_dummy", "selfplay"], default="vs_dummy")
    parser.add_argument("--num_envs", type=int, default=1, choices=range(1, 17), help="Number of parallel environments")
    parser.add_argument("--total_steps", type=int, default=500000)
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--opponent_difficulty", type=str, default="easy", choices=["easy", "hard"], help="Dummy opponent difficulty")
    parser.add_argument("--load_model", type=str, default=None, help="Path to pretrained model to load")
    
    args = parser.parse_args()
    
    if args.log_dir is None:
        args.log_dir = f"runs/inversus_{args.mode}_envs{args.num_envs}"
    
    if args.mode == "vs_dummy":
        train_vs_dummy(
            num_envs=args.num_envs, 
            total_steps=args.total_steps, 
            log_dir=args.log_dir,
            opponent_difficulty=args.opponent_difficulty,
            load_model=args.load_model
        )
    else:
        train_selfplay(
            num_envs=args.num_envs, 
            total_steps=args.total_steps, 
            log_dir=args.log_dir,
            load_model=args.load_model
        )


if __name__ == "__main__":
    main()
