"""Tests for RL environment wrappers."""

import pytest
import numpy as np

try:
    from inversus_rl.env_wrappers import (
        SingleInversusRLEnv, MultiEnvRunner, discrete_to_action, build_observation
    )
    from inversus_rl.policies import make_policy_from_env
    from inversus.core import InversusEnv
    from inversus.game_types import ActionType, Direction, PlayerId
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from inversus_rl.env_wrappers import (
        SingleInversusRLEnv, MultiEnvRunner, discrete_to_action, build_observation
    )
    from inversus_rl.policies import make_policy_from_env
    from inversus.core import InversusEnv
    from inversus.game_types import ActionType, Direction, PlayerId


def test_discrete_to_action():
    """Test discrete action conversion."""
    # Test NONE
    action = discrete_to_action(0)
    assert action.type == ActionType.NONE
    assert action.direction is None
    
    # Test MOVE
    action = discrete_to_action(1)
    assert action.type == ActionType.MOVE
    assert action.direction == Direction.UP
    
    # Test SHOOT
    action = discrete_to_action(5)
    assert action.type == ActionType.SHOOT
    assert action.direction == Direction.UP
    
    # Test CHARGE_SHOOT
    action = discrete_to_action(9)
    assert action.type == ActionType.CHARGE_SHOOT
    assert action.direction == Direction.UP


def test_single_env_reset_and_step():
    """Test SingleInversusRLEnv reset and step."""
    env = SingleInversusRLEnv(opponent_type="dummy", max_episode_steps=100)
    
    # Reset
    obs = env.reset()
    grid_tensor, extra_vector = obs
    
    # Check shapes
    assert grid_tensor.shape == (6, env.env.height, env.env.width)
    assert extra_vector.shape == (4,)
    assert grid_tensor.dtype == np.float32
    assert extra_vector.dtype == np.float32
    
    # Step with random action
    action_id = 1  # MOVE UP
    obs, reward, done, info = env.step(action_id)
    grid_tensor, extra_vector = obs
    
    # Check shapes again
    assert grid_tensor.shape == (6, env.env.height, env.env.width)
    assert extra_vector.shape == (4,)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)
    
    # Check info keys
    assert "win" in info
    assert "lose" in info
    assert "episode_steps" in info


def test_multi_env_runner_shapes():
    """Test MultiEnvRunner batched outputs."""
    num_envs = 3
    runner = MultiEnvRunner(num_envs=num_envs, opponent_type="dummy", max_episode_steps=100)
    
    # Reset
    obs = runner.reset()
    grid_tensors, extra_vectors = obs
    
    # Check batched shapes
    assert grid_tensors.shape == (num_envs, 6, runner.envs[0].env.height, runner.envs[0].env.width)
    assert extra_vectors.shape == (num_envs, 4)
    
    # Step with random actions
    action_ids = np.array([1, 2, 3])  # Different actions for each env
    obs, rewards, dones, infos = runner.step(action_ids)
    grid_tensors, extra_vectors = obs
    
    # Check shapes
    assert grid_tensors.shape == (num_envs, 6, runner.envs[0].env.height, runner.envs[0].env.width)
    assert extra_vectors.shape == (num_envs, 4)
    assert rewards.shape == (num_envs,)
    assert dones.shape == (num_envs,)
    assert len(infos) == num_envs


def test_build_observation():
    """Test observation building."""
    env = InversusEnv()
    env.reset()
    
    obs = build_observation(env, PlayerId.P1)
    grid_tensor, extra_vector = obs
    
    # Check shapes
    assert grid_tensor.shape == (6, env.height, env.width)
    assert extra_vector.shape == (4,)
    
    # Check channel 0 and 1 are mutually exclusive (tile is either black or white)
    for y in range(env.height):
        for x in range(env.width):
            assert grid_tensor[0, y, x] + grid_tensor[1, y, x] == 1.0
    
    # Check player position channel
    p1 = env.get_player(PlayerId.P1)
    assert grid_tensor[2, p1.y, p1.x] == 1.0
    
    # Check enemy position channel
    p2 = env.get_player(PlayerId.P2)
    assert grid_tensor[3, p2.y, p2.x] == 1.0


def test_policy_from_env():
    """Test policy creation from environment."""
    try:
        import torch
    except ImportError:
        pytest.skip("torch not installed, skipping policy test")
    
    env = SingleInversusRLEnv()
    policy = make_policy_from_env(env)
    
    # Get observation
    obs = env.reset()
    grid_tensor, extra_vector = obs
    
    # Test forward pass
    grid_t = torch.FloatTensor(grid_tensor).unsqueeze(0)
    extra_t = torch.FloatTensor(extra_vector).unsqueeze(0)
    
    logits, value = policy(grid_t, extra_t)
    
    assert logits.shape == (1, 13)  # 13 actions
    assert value.shape == (1, 1)  # Single value

