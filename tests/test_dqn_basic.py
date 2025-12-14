"""Basic tests for DQN components."""

import pytest
import numpy as np

try:
    from inversus_rl.env_wrappers import SingleInversusRLEnv
    from inversus_rl.dqn_networks import make_dqn_from_env, InversusDuelingDQN
    from inversus_rl.dqn_agent import RainbowDQNAgent
    from inversus_rl.replay_buffer import PrioritizedReplayBuffer
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from inversus_rl.env_wrappers import SingleInversusRLEnv
    from inversus_rl.dqn_networks import make_dqn_from_env, InversusDuelingDQN
    from inversus_rl.dqn_agent import RainbowDQNAgent
    from inversus_rl.replay_buffer import PrioritizedReplayBuffer


def test_dqn_import_and_init():
    """Test that DQN components can be imported and initialized."""
    env = SingleInversusRLEnv(opponent_type="dummy")
    q_net = make_dqn_from_env(env)
    target_net = make_dqn_from_env(env)
    
    agent = RainbowDQNAgent(q_network=q_net, target_network=target_net)
    
    obs = env.reset()
    grid, extra = obs
    action = agent.act(grid, extra, frame_idx=0)
    assert 0 <= action < 13


def test_dqn_network_forward():
    """Test DQN network forward pass."""
    env = SingleInversusRLEnv(opponent_type="dummy")
    q_net = make_dqn_from_env(env)
    
    obs = env.reset()
    grid, extra = obs
    
    import torch
    grid_t = torch.FloatTensor(grid).unsqueeze(0)
    extra_t = torch.FloatTensor(extra).unsqueeze(0)
    
    q_values = q_net(grid_t, extra_t)
    assert q_values.shape == (1, 13)  # Batch size 1, 13 actions


def test_replay_buffer():
    """Test prioritized replay buffer."""
    buffer = PrioritizedReplayBuffer(capacity=1000)
    
    # Create dummy transition
    grid = np.random.rand(6, 10, 15).astype(np.float32)
    extra = np.random.rand(4).astype(np.float32)
    next_grid = np.random.rand(6, 10, 15).astype(np.float32)
    next_extra = np.random.rand(4).astype(np.float32)
    
    # Push some transitions
    for _ in range(10):
        buffer.push(
            grid, extra, 1, 0.5, next_grid, next_extra, False
        )
    
    assert len(buffer) == 10
    
    # Sample batch
    batch, indices, weights = buffer.sample(batch_size=5, frame_idx=1000)
    assert batch['grid'].shape[0] == 5
    assert batch['extra'].shape[0] == 5
    assert len(indices) == 5
    assert len(weights) == 5


def test_dqn_agent_epsilon_decay():
    """Test epsilon decay schedule."""
    env = SingleInversusRLEnv(opponent_type="dummy")
    q_net = make_dqn_from_env(env)
    target_net = make_dqn_from_env(env)
    
    agent = RainbowDQNAgent(
        q_network=q_net,
        target_network=target_net,
        epsilon_start=1.0,
        epsilon_final=0.05,
        epsilon_decay=1000
    )
    
    # Check epsilon values
    assert agent.epsilon(0) == 1.0
    assert agent.epsilon(1000) == 0.05
    assert agent.epsilon(2000) == 0.05  # Should stay at final


def test_dqn_agent_update():
    """Test DQN agent update (with enough samples)."""
    env = SingleInversusRLEnv(opponent_type="dummy")
    q_net = make_dqn_from_env(env)
    target_net = make_dqn_from_env(env)
    
    agent = RainbowDQNAgent(
        q_network=q_net,
        target_network=target_net,
        min_replay_size=10,
        batch_size=8
    )
    
    # Fill buffer with transitions
    obs = env.reset()
    grid, extra = obs
    
    for _ in range(20):
        action = agent.act(grid, extra, frame_idx=0)
        next_obs, reward, done, _ = env.step(action)
        next_grid, next_extra = next_obs
        
        agent.push_transition(
            grid, extra, action, reward, next_grid, next_extra, done
        )
        
        if done:
            obs = env.reset()
        else:
            obs = next_obs
        grid, extra = obs
    
    # Try update
    stats = agent.update(frame_idx=100)
    assert stats is not None
    assert 'q_loss' in stats
    assert 'mean_q' in stats
    assert 'epsilon' in stats


