"""RL environment wrappers for INVERSUS."""

import numpy as np
from typing import Tuple, Optional, Dict, Any, List
import random

try:
    from inversus.core import InversusEnv
    from inversus.game_types import TileColor, Direction, ActionType, Action, PlayerId
    from inversus.config import MAX_AMMO
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from inversus.core import InversusEnv
    from inversus.game_types import TileColor, Direction, ActionType, Action, PlayerId
    from inversus.config import MAX_AMMO


def discrete_to_action(action_id: int) -> Action:
    """
    Convert a discrete action ID to an Action.
    
    Action space (13 actions):
    0  -> NONE
    1  -> MOVE UP
    2  -> MOVE RIGHT
    3  -> MOVE DOWN
    4  -> MOVE LEFT
    5  -> SHOOT UP
    6  -> SHOOT RIGHT
    7  -> SHOOT DOWN
    8  -> SHOOT LEFT
    9  -> CHARGE_SHOOT UP
    10 -> CHARGE_SHOOT RIGHT
    11 -> CHARGE_SHOOT DOWN
    12 -> CHARGE_SHOOT LEFT
    """
    if action_id == 0:
        return Action(ActionType.NONE, None)
    elif action_id == 1:
        return Action(ActionType.MOVE, Direction.UP)
    elif action_id == 2:
        return Action(ActionType.MOVE, Direction.RIGHT)
    elif action_id == 3:
        return Action(ActionType.MOVE, Direction.DOWN)
    elif action_id == 4:
        return Action(ActionType.MOVE, Direction.LEFT)
    elif action_id == 5:
        return Action(ActionType.SHOOT, Direction.UP)
    elif action_id == 6:
        return Action(ActionType.SHOOT, Direction.RIGHT)
    elif action_id == 7:
        return Action(ActionType.SHOOT, Direction.DOWN)
    elif action_id == 8:
        return Action(ActionType.SHOOT, Direction.LEFT)
    elif action_id == 9:
        return Action(ActionType.CHARGE_SHOOT, Direction.UP)
    elif action_id == 10:
        return Action(ActionType.CHARGE_SHOOT, Direction.RIGHT)
    elif action_id == 11:
        return Action(ActionType.CHARGE_SHOOT, Direction.DOWN)
    elif action_id == 12:
        return Action(ActionType.CHARGE_SHOOT, Direction.LEFT)
    else:
        raise ValueError(f"Invalid action_id: {action_id}, must be 0-12")


def dummy_opponent_policy(env: InversusEnv, difficulty: str = "easy") -> Action:
    """
    Simple scripted dummy opponent policy.
    Difficulty: 'easy' (sitting duck), 'hard' (original)
    """
    p1 = env.get_player(PlayerId.P1)
    p2 = env.get_player(PlayerId.P2)
    
    if not p2.alive:
        return Action(ActionType.NONE, None)

    # Difficulty Settings
    if difficulty == "easy":
        move_prob = 0.001       # 0.1% (Basically never moves)
        shoot_prob = 0.0        # 0% (Never shoots)
        random_move_prob = 0.0  # 0%
    else:
        # Hard / Normal
        move_prob = 1.0         # Always move if logic says so
        shoot_prob = 0.01       # 1% per frame
        random_move_prob = 0.2  # 20% distraction

    # Move random move logic to TOP so it can interrupt shooting (Distracted)
    directions = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]
    if random.random() < random_move_prob:
        random.shuffle(directions)
        return Action(ActionType.MOVE, directions[0])
    
    # Shooting Logic
    should_shoot = random.random() < shoot_prob
    if should_shoot and p1.x == p2.x and p2.ammo > 0:
        if p1.y < p2.y:
            return Action(ActionType.SHOOT, Direction.UP)
        elif p1.y > p2.y:
            return Action(ActionType.SHOOT, Direction.DOWN)
            
    if should_shoot and p1.y == p2.y and p2.ammo > 0:
        if p1.x < p2.x:
            return Action(ActionType.SHOOT, Direction.LEFT)
        elif p1.x > p2.x:
            return Action(ActionType.SHOOT, Direction.RIGHT)
    
    # Movement Logic
    # In easy mode, we only move rarely
    if difficulty == "easy":
        if random.random() > move_prob:
             return Action(ActionType.NONE, None)
             
    # Otherwise, try safe move
    random.shuffle(directions)
    for direction in directions:
        # Check if move would be valid
        dx, dy = 0, 0
        if direction == Direction.UP:
            dy = -1
        elif direction == Direction.DOWN:
            dy = 1
        elif direction == Direction.LEFT:
            dx = -1
        elif direction == Direction.RIGHT:
            dx = 1
        
        nx = p2.x + dx
        ny = p2.y + dy
        
        if env._tile_in_bounds(nx, ny):
            tile = env._get_tile(nx, ny)
            if tile != p2.color:  # Walkable
                return Action(ActionType.MOVE, direction)
                
    # Fallback if no move is possible (e.g. trapped)
    return Action(ActionType.NONE, None)


def build_observation(env: InversusEnv, player_id: PlayerId = PlayerId.P1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build observation for a player.
    
    Returns:
        (grid_tensor, extra_vector) where:
        - grid_tensor: (6, H, W) float32 array
        - extra_vector: (4,) float32 array [player_ammo_norm, enemy_ammo_norm, player_alive, enemy_alive]
    """
    height = env.height
    width = env.width
    
    # Initialize 12-channel grid tensor
    # 0: tile_black, 1: tile_white
    # 2: p1_pos, 3: p2_pos
    # 4-7: p1_bullets (UP, RIGHT, DOWN, LEFT)
    # 8-11: p2_bullets (UP, RIGHT, DOWN, LEFT)
    grid_tensor = np.zeros((12, height, width), dtype=np.float32)
    
    # Get players
    player = env.get_player(player_id)
    enemy_id = PlayerId.P2 if player_id == PlayerId.P1 else PlayerId.P1
    enemy = env.get_player(enemy_id)
    
    # Channel 0: tile_is_black
    # Channel 1: tile_is_white
    grid = env.get_grid_copy()
    for y in range(height):
        for x in range(width):
            if grid[y][x] == TileColor.BLACK:
                grid_tensor[0, y, x] = 1.0
            else:
                grid_tensor[1, y, x] = 1.0
    
    # Channel 2: player_position
    if player.alive:
        grid_tensor[2, player.y, player.x] = 1.0
    
    # Channel 3: enemy_position
    if enemy.alive:
        grid_tensor[3, enemy.y, enemy.x] = 1.0
    
    # Helper to map direction to index offset (0=UP, 1=RIGHT, 2=DOWN, 3=LEFT)
    dir_map = {
        Direction.UP: 0,
        Direction.RIGHT: 1,
        Direction.DOWN: 2,
        Direction.LEFT: 3
    }
    
    # Bullet channels
    for bullet in env.get_bullets():
        if not env._tile_in_bounds(bullet.x, bullet.y):
            continue
            
        d_idx = dir_map.get(bullet.dir, 0)
        
        if bullet.owner == player_id:
            # Channels 4-7: Player bullets
            grid_tensor[4 + d_idx, bullet.y, bullet.x] = 1.0
        else:
            # Channels 8-11: Enemy bullets
            grid_tensor[8 + d_idx, bullet.y, bullet.x] = 1.0
    
    # Extra vector: [player_ammo_norm, enemy_ammo_norm, player_alive, enemy_alive]
    extra_vector = np.array([
        player.ammo / MAX_AMMO if player.alive else 0.0,
        enemy.ammo / MAX_AMMO if enemy.alive else 0.0,
        1.0 if player.alive else 0.0,
        1.0 if enemy.alive else 0.0,
    ], dtype=np.float32)
    
    return grid_tensor, extra_vector


class SingleInversusRLEnv:
    """Wrapper around InversusEnv for single-agent RL (agent controls P1)."""
    
    def __init__(self, opponent_type: str = "dummy", difficulty: str = "easy", max_episode_steps: int = 500, seed: Optional[int] = None):
        """
        Initialize RL environment.
        
        Args:
            opponent_type: "dummy" or "selfplay"
            difficulty: Difficulty level for dummy opponent ("easy", "hard")
            max_episode_steps: Maximum steps before forcing done=True
            seed: Random seed for environment
        """
        self.opponent_type = opponent_type
        self.difficulty = difficulty
        self.max_episode_steps = max_episode_steps
        self.env = InversusEnv(seed=seed)
        self.step_count = 0
        self.episode_return = 0.0
        # Track previous state for reward shaping
        self.prev_p1_alive = True
        self.prev_p2_alive = True
        self.prev_bullet_count = 0
        
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Reset environment and return initial observation."""
        if seed is not None:
            self.env.reset(seed=seed)
        else:
            self.env.reset()
        self.step_count = 0
        self.episode_return = 0.0
        # Reset tracking variables
        self.prev_p1_alive = True
        self.prev_p2_alive = True
        self.prev_bullet_count = 0
        return build_observation(self.env, PlayerId.P1)
    
    def step(
        self, 
        action_id: int, 
        opponent_policy=None
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], float, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            action_id: Discrete action ID (0-12)
            opponent_policy: Policy function for selfplay mode (takes env, returns action_id)
            
        Returns:
            (obs, reward, done, info)
        """
        # Convert action for P1
        action_p1 = discrete_to_action(action_id)
        
        # Get action for P2
        if self.opponent_type == "dummy":
            action_p2 = dummy_opponent_policy(self.env, difficulty=self.difficulty)
        elif self.opponent_type == "selfplay":
            if opponent_policy is None:
                raise ValueError("opponent_policy required for selfplay mode")
            # Build observation from P2's perspective
            obs_p2 = build_observation(self.env, PlayerId.P2)
            # Get action from policy (assuming it returns action_id)
            action_id_p2 = opponent_policy(obs_p2)
            action_p2 = discrete_to_action(action_id_p2)
        else:
            raise ValueError(f"Unknown opponent_type: {self.opponent_type}")
        
        # Store previous state for reward shaping
        prev_p1_alive = self.prev_p1_alive
        prev_p2_alive = self.prev_p2_alive
        
        # Get PREVIOUS player state
        p1_pre = self.env.get_player(PlayerId.P1)
        prev_x = p1_pre.x
        prev_y = p1_pre.y
        
        # Calculate walkable tiles (Territory) before step
        walkable_color = TileColor.WHITE if self.env.player_color == TileColor.BLACK else TileColor.BLACK
        prev_walkable_count = sum(row.count(walkable_color) for row in self.env.grid)
        
        # Step environment
        self.env.step_players(action_p1, action_p2)
        self.step_count += 1
        
        # Get current state
        p1 = self.env.get_player(PlayerId.P1)
        p2 = self.env.get_player(PlayerId.P2)
        
        # Build observation
        obs = build_observation(self.env, PlayerId.P1)
        
        # ========== DENSE REWARD SHAPING ==========
        reward = 0.0
        done = False
        info: Dict[str, Any] = {}
        
        
        # 0. Territory Expansion Reward (Teach it to open paths)
        curr_walkable_count = sum(row.count(walkable_color) for row in self.env.grid)
        # 0. Territory Expansion Reward (Teach it to open paths)
        curr_walkable_count = sum(row.count(walkable_color) for row in self.env.grid)
        tile_diff = curr_walkable_count - prev_walkable_count
        if tile_diff > 0:
            reward += tile_diff * 0.01  # Reduced from 0.5 (Scale fix)
            
        # 1. Hit detection rewards (detect if someone just died)
        if prev_p2_alive and not p2.alive:
            # P1 just killed P2!
            reward += 1.0  # Reduced from 50.0 (Scale fix)
            info["landed_hit"] = True
        else:
            info["landed_hit"] = False
            
        if prev_p1_alive and not p1.alive:
            # P1 just got killed!
            reward -= 0.01  # Reduced from -0.5 (Scale fix)
            info["got_hit"] = True
        else:
            info["got_hit"] = False
        
        # 2. Ammo management penalty (running out of ammo is bad)
        if p1.alive and p1.ammo == 0:
            reward -= 0.001
            
        # 3. Dense Shaping: Proximity and Aiming (Teach it to hunt)
        # Distance shaping
        if p1.alive and p2.alive:
            dist = abs(p1.x - p2.x) + abs(p1.y - p2.y)
            max_dist = self.env.width + self.env.height
            # Dense reward for being close (0.0 to 0.05)
            # Reduced to just a faint scent
            reward += 0.002 * (1.0 - dist / max_dist)  # Boosted slightly from 0.001
            
            # Aiming shaping (Bonus for aligning coordinates)
            is_aligned = (p1.x == p2.x) or (p1.y == p2.y)
            if is_aligned:
                reward += 0.002  # Boosted slightly
                
            # TRIGGER DISCIPLINE: Reward for shooting while aligned AND TOWARDS enemy
            # CRITICAL FIX: Only reward if we actually have ammo! (Prevent dry-fire farming)
            if 5 <= action_id <= 12 and is_aligned and p1.ammo > 0:
                action = discrete_to_action(action_id)
                shot_dir = action.direction
                
                # Verify shot is TOWARDS the enemy
                is_aiming_at_enemy = False
                if p1.x == p2.x: # Vertical alignment
                    if p1.y < p2.y and shot_dir == Direction.DOWN: is_aiming_at_enemy = True
                    if p1.y > p2.y and shot_dir == Direction.UP: is_aiming_at_enemy = True
                elif p1.y == p2.y: # Horizontal alignment
                    if p1.x < p2.x and shot_dir == Direction.RIGHT: is_aiming_at_enemy = True
                    if p1.x > p2.x and shot_dir == Direction.LEFT: is_aiming_at_enemy = True
                
                if is_aiming_at_enemy:
                    reward += 0.05 # Boosted from 0.01 to encourage pulling the trigger
        
        # 4. Check if round is over (sparse terminal rewards)
        if self.env.is_round_over():
            done = True
            winner = self.env.get_winner()
            if winner == PlayerId.P1:
                reward += 10.0  # JACKPOT
                info["win"] = True
                info["lose"] = False
            elif winner == PlayerId.P2:
                reward -= 0.1  # Negligible loss penalty (Dying is better than Loitering)
                info["win"] = False
                info["lose"] = True
            else:
                # Tie
                info["win"] = False
                info["lose"] = False
        else:
            # Time penalty to encourage finishing quickly (and punish loitering)
            reward -= 0.001  # Reduced from 0.005 (Don't punish existence so much)
            info["win"] = False
            info["lose"] = False
        
        # Update tracking variables
        self.prev_p1_alive = p1.alive
        self.prev_p2_alive = p2.alive
        
        # Check max episode steps
        if self.step_count >= self.max_episode_steps:
            done = True
            # Penalty for timeout (couldn't win in time)
            if not self.env.is_round_over():
                reward -= 2.0  # Reduced from 5.0
        
        self.episode_return += reward
        info["episode_steps"] = self.step_count
        info["episode_return"] = self.episode_return
        
        return obs, reward, done, info


class MultiEnvRunner:
    """Runs multiple parallel environments for vectorized RL training."""
    
    def __init__(self, num_envs: int, opponent_type: str = "dummy", difficulty: str = "easy", max_episode_steps: int = 500, seed: Optional[int] = None):
        """
        Initialize multi-environment runner.
        
        Args:
            num_envs: Number of parallel environments
            opponent_type: "dummy" or "selfplay"
            difficulty: Difficulty level for dummy opponent ("easy", "hard")
            max_episode_steps: Maximum steps per episode
            seed: Base seed (each env gets seed + env_idx)
        """
        self.num_envs = num_envs
        self.envs = [
            SingleInversusRLEnv(opponent_type, difficulty, max_episode_steps, seed=(seed + i) if seed is not None else None)
            for i in range(num_envs)
        ]
        self.episode_returns = [0.0] * num_envs
        self.episode_lengths = [0] * num_envs
        self.episode_wins = [0] * num_envs
        self.episode_losses = [0] * num_envs
    
    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reset all environments.
        
        Returns:
            (grid_tensors, extra_vectors) where:
            - grid_tensors: (num_envs, 6, H, W)
            - extra_vectors: (num_envs, 4)
        """
        obs_list = [env.reset() for env in self.envs]
        grid_tensors = np.stack([obs[0] for obs in obs_list], axis=0)
        extra_vectors = np.stack([obs[1] for obs in obs_list], axis=0)
        return grid_tensors, extra_vectors
    
    def step(
        self, 
        action_ids: np.ndarray, 
        opponent_policy=None
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """
        Step all environments.
        
        Args:
            action_ids: (num_envs,) array of action IDs
            opponent_policy: Policy for selfplay mode
            
        Returns:
            (obs, rewards, dones, infos)
        """
        obs_list = []
        reward_list = []
        done_list = []
        info_list = []
        
        for i, env in enumerate(self.envs):
            obs, reward, done, info = env.step(int(action_ids[i]), opponent_policy)
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)
            
            # Track episode stats
            if done:
                self.episode_returns[i] = info.get("episode_return", 0.0)
                self.episode_lengths[i] = info.get("episode_steps", 0)
                if info.get("win", False):
                    self.episode_wins[i] += 1
                if info.get("lose", False):
                    self.episode_losses[i] += 1
        
        # Stack observations
        grid_tensors = np.stack([obs[0] for obs in obs_list], axis=0)
        extra_vectors = np.stack([obs[1] for obs in obs_list], axis=0)
        
        rewards = np.array(reward_list, dtype=np.float32)
        dones = np.array(done_list, dtype=bool)
        
        return (grid_tensors, extra_vectors), rewards, dones, info_list

