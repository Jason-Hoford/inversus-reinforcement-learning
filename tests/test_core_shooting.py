"""Tests for shooting behavior in INVERSUS."""

import pytest
from inversus.core import InversusEnv
from inversus.game_types import TileColor, Direction, ActionType, Action


def test_shoot_flips_tiles_in_line_until_out_of_bounds():
    """Test that shooting flips tiles in a line until out of bounds."""
    # Create a narrow horizontal grid
    env = InversusEnv(width=5, height=1)
    
    # Set player position
    env.player_x = 2
    env.player_y = 0
    
    # Ensure all tiles start as same color (player color)
    for x in range(env.width):
        env._set_tile(x, 0, env.player_color)
    
    # Get initial tile colors
    initial_tiles = [env._get_tile(x, 0) for x in range(env.width)]
    
    # Shoot right and update bullets until bullet leaves grid
    env.step(Action(ActionType.SHOOT, Direction.RIGHT))
    
    # Update bullets enough times for bullet to leave (width=5, player at x=2, so 3 updates needed)
    for _ in range(3):
        env.update_bullets()
    
    # Check tiles to the right of player (indices 3 and 4) have been flipped
    # Tile at x=3 should be flipped (bullet passed through)
    tile_3 = env._get_tile(3, 0)
    assert tile_3 != env.player_color
    
    # Tile at x=4 should be flipped (bullet passed through)
    tile_4 = env._get_tile(4, 0)
    assert tile_4 != env.player_color
    
    # Tiles at player position or left should not have changed
    assert env._get_tile(0, 0) == initial_tiles[0]
    assert env._get_tile(1, 0) == initial_tiles[1]
    assert env._get_tile(2, 0) == initial_tiles[2]  # Player position


def test_bullet_removed_when_out_of_bounds():
    """Test that bullets are removed when they go out of bounds."""
    env = InversusEnv(width=5, height=5)
    
    # Spawn bullet near right edge shooting right
    env.player_x = 3
    env.player_y = 2
    env.spawn_bullet(Direction.RIGHT)
    
    # Bullet should be at (3, 2) initially
    bullets = env.get_bullets()
    assert len(bullets) == 1
    assert bullets[0].x == 3
    
    # Update once: bullet moves to (4, 2) - still in bounds
    env.update_bullets()
    bullets = env.get_bullets()
    assert len(bullets) == 1
    assert bullets[0].x == 4
    
    # Update again: bullet moves to (5, 2) - out of bounds, should be removed
    env.update_bullets()
    bullets = env.get_bullets()
    assert len(bullets) == 0


def test_multiple_bullets_update_independently():
    """Test that multiple bullets update independently."""
    env = InversusEnv(width=10, height=10)
    
    # Set player position
    env.player_x = 5
    env.player_y = 5
    
    # Spawn two bullets: one going left, one going right
    env.spawn_bullet(Direction.LEFT)
    env.spawn_bullet(Direction.RIGHT)
    
    bullets_before = env.get_bullets()
    assert len(bullets_before) == 2
    
    # Find bullets
    left_bullet = next(b for b in bullets_before if b.dir == Direction.LEFT)
    right_bullet = next(b for b in bullets_before if b.dir == Direction.RIGHT)
    
    # Both should start at player position
    assert left_bullet.x == 5
    assert left_bullet.y == 5
    assert right_bullet.x == 5
    assert right_bullet.y == 5
    
    # Get tile colors at destinations
    left_dest_x = 4
    right_dest_x = 6
    left_tile_before = env._get_tile(left_dest_x, 5)
    right_tile_before = env._get_tile(right_dest_x, 5)
    
    # Step once
    env.step(Action(ActionType.NONE, None))
    
    bullets_after = env.get_bullets()
    assert len(bullets_after) == 2
    
    # Check bullets moved in their respective directions
    left_bullet_after = next(b for b in bullets_after if b.dir == Direction.LEFT)
    right_bullet_after = next(b for b in bullets_after if b.dir == Direction.RIGHT)
    
    assert left_bullet_after.x == 4
    assert left_bullet_after.y == 5
    assert right_bullet_after.x == 6
    assert right_bullet_after.y == 5
    
    # Check tiles were flipped
    left_tile_after = env._get_tile(left_dest_x, 5)
    right_tile_after = env._get_tile(right_dest_x, 5)
    
    assert left_tile_after != left_tile_before
    assert right_tile_after != right_tile_before


def test_step_shoot_spawns_bullet_and_updates_grid():
    """Test that step with SHOOT action spawns bullet and updates grid."""
    env = InversusEnv(width=10, height=10)
    
    # Set player position
    env.player_x = 5
    env.player_y = 5
    
    # Record initial state
    bullets_before = env.get_bullets()
    bullets_count_before = len(bullets_before)
    
    # Get tile color in shooting direction
    shoot_x = 6  # Right of player
    shoot_y = 5
    tile_before = env._get_tile(shoot_x, shoot_y)
    
    # Step with SHOOT action
    env.step(Action(ActionType.SHOOT, Direction.RIGHT))
    
    # Check bullet count increased
    bullets_after = env.get_bullets()
    bullets_count_after = len(bullets_after)
    assert bullets_count_after == bullets_count_before + 1
    
    # After update_bullets (called by step), bullet should have moved and flipped tile
    # The tile at (6, 5) should be flipped
    tile_after = env._get_tile(shoot_x, shoot_y)
    assert tile_after != tile_before
    
    # Bullet should now be at (6, 5) since update_bullets() was called by step()
    new_bullet = bullets_after[0]
    assert new_bullet.x == 6
    assert new_bullet.y == 5
    assert new_bullet.dir == Direction.RIGHT


def test_bullet_flips_tile_at_new_position_not_old():
    """Test that bullet flips tile at new position, not old position."""
    env = InversusEnv(width=10, height=10)
    
    # Set player position
    env.player_x = 5
    env.player_y = 5
    
    # Get tile colors
    player_tile = env._get_tile(5, 5)
    next_tile = env._get_tile(6, 5)
    
    # Shoot right
    env.step(Action(ActionType.SHOOT, Direction.RIGHT))
    
    # Check: player tile should be unchanged (bullet spawned there but didn't flip it)
    assert env._get_tile(5, 5) == player_tile
    
    # Check: next tile should be flipped (bullet moved there and flipped it)
    assert env._get_tile(6, 5) != next_tile

