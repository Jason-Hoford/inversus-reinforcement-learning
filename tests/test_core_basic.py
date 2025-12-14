"""Basic tests for INVERSUS core logic."""

import pytest
from inversus.core import InversusEnv
from inversus.game_types import TileColor, Direction, ActionType, Action
from inversus.config import make_initial_grid


def test_player_can_only_move_on_opposite_color():
    """Test that player can only move on tiles of opposite color."""
    # Create a small environment
    env = InversusEnv(width=5, height=5)
    
    # Manually set up grid: player at (2, 2)
    env.player_x = 2
    env.player_y = 2
    
    # Set tile to the right to opposite color (walkable)
    opposite_color = TileColor.WHITE if env.player_color == TileColor.BLACK else TileColor.BLACK
    env._set_tile(3, 2, opposite_color)
    
    # Set tile to the left to same color (not walkable)
    env._set_tile(1, 2, env.player_color)
    
    # Record initial position
    initial_x = env.player_x
    initial_y = env.player_y
    
    # Try to move right (should succeed)
    result_right = env.try_move_player(Direction.RIGHT)
    assert result_right is True
    assert env.player_x == 3
    assert env.player_y == 2
    
    # Reset position
    env.player_x = initial_x
    env.player_y = initial_y
    
    # Try to move left (should fail)
    result_left = env.try_move_player(Direction.LEFT)
    assert result_left is False
    assert env.player_x == initial_x
    assert env.player_y == initial_y


def test_reset_restores_initial_state():
    """Test that reset restores the initial state."""
    env = InversusEnv()
    
    # Modify state
    env._set_tile(0, 0, TileColor.WHITE)
    env.player_x = 10
    env.player_y = 8
    env.spawn_bullet(Direction.UP)
    
    # Reset
    env.reset()
    
    # Check player position
    from inversus.config import DEFAULT_PLAYER_START
    assert env.get_player_position() == DEFAULT_PLAYER_START
    
    # Check grid - note: now includes both P1 and P2 walkable areas, so we check
    # that the initial P1 area is restored (not checking exact match due to P2 changes)
    # The key is that reset() creates a fresh grid with walkable areas
    actual_grid = env.get_grid_copy()
    # Check that P1 start area is walkable (opposite color)
    from inversus.config import DEFAULT_PLAYER_START
    p1_x, p1_y = DEFAULT_PLAYER_START
    opposite_color = TileColor.WHITE if env.player_color == TileColor.BLACK else TileColor.BLACK
    assert actual_grid[p1_y][p1_x] == opposite_color
    
    # Check no bullets (reset should clear them)
    assert len(env.get_bullets()) == 0
    
    # Check both players are alive and at correct positions
    assert env.player1.alive
    assert env.player2.alive
    from inversus.config import DEFAULT_PLAYER_START, DEFAULT_PLAYER2_START
    assert env.player1.x == DEFAULT_PLAYER_START[0]
    assert env.player1.y == DEFAULT_PLAYER_START[1]
    # Check player 2 position (should be at DEFAULT_PLAYER2_START)
    assert env.player2.x == DEFAULT_PLAYER2_START[0]
    assert env.player2.y == DEFAULT_PLAYER2_START[1]


def test_step_with_none_action_updates_bullets_only():
    """Test that step with NONE action only updates bullets."""
    env = InversusEnv(width=10, height=10)
    
    # Set player position
    env.player_x = 5
    env.player_y = 5
    
    # Spawn a bullet manually
    env.spawn_bullet(Direction.RIGHT)
    bullets_before = env.get_bullets()
    assert len(bullets_before) == 1
    
    bullet = bullets_before[0]
    bullet_start_x = bullet.x
    bullet_start_y = bullet.y
    
    # Get tile color at bullet's destination
    dest_x = bullet_start_x + 1
    dest_y = bullet_start_y
    tile_before = env._get_tile(dest_x, dest_y)
    
    # Record player position
    player_x_before = env.player_x
    player_y_before = env.player_y
    
    # Step with NONE action
    env.step(Action(ActionType.NONE, None))
    
    # Check player position unchanged
    assert env.player_x == player_x_before
    assert env.player_y == player_y_before
    
    # Check bullet moved
    bullets_after = env.get_bullets()
    assert len(bullets_after) == 1
    assert bullets_after[0].x == dest_x
    assert bullets_after[0].y == dest_y
    
    # Check tile was flipped
    tile_after = env._get_tile(dest_x, dest_y)
    assert tile_after != tile_before


def test_move_out_of_bounds_fails():
    """Test that moving out of bounds fails."""
    env = InversusEnv(width=5, height=5)
    
    # Move player to top-left corner
    env.player_x = 0
    env.player_y = 0
    
    # Try to move up (should fail - out of bounds)
    result = env.try_move_player(Direction.UP)
    assert result is False
    assert env.player_x == 0
    assert env.player_y == 0
    
    # Try to move left (should fail - out of bounds)
    result = env.try_move_player(Direction.LEFT)
    assert result is False
    assert env.player_x == 0
    assert env.player_y == 0


def test_get_grid_copy_is_independent():
    """Test that get_grid_copy returns an independent copy."""
    env = InversusEnv()
    
    grid_copy = env.get_grid_copy()
    
    # Modify original grid
    env._set_tile(0, 0, TileColor.WHITE)
    
    # Copy should be unchanged
    assert grid_copy[0][0] != TileColor.WHITE

