"""Tests for bullet flip rules - bullets only open paths, don't destroy existing ones."""

import pytest
from inversus.core import InversusEnv
from inversus.game_types import TileColor, Direction, PlayerId, Bullet


def test_bullet_only_flips_owner_color_tiles():
    """Test that bullets only flip tiles that match the owner's color."""
    env = InversusEnv(width=5, height=1)
    env.reset()
    
    # P1 has color BLACK, so they walk on WHITE
    # Set up two adjacent tiles:
    env._set_tile(1, 0, TileColor.BLACK)   # same as owner color (will be flipped)
    env._set_tile(2, 0, TileColor.WHITE)   # opposite of owner color (should NOT be flipped)
    
    # Place P1 at x=0
    env.player1.x = 0
    env.player1.y = 0
    
    # Place a bullet owned by P1 at x=0 headed right
    env.bullets = [Bullet(x=0, y=0, dir=Direction.RIGHT, owner=PlayerId.P1)]
    
    # First update: bullet moves to x=1 where tile is BLACK (owner's color)
    env.update_bullets()
    
    # Bullet should now be at x=1 and flip that tile from BLACK -> WHITE
    assert env.bullets[0].x == 1
    assert env.bullets[0].y == 0
    assert env._get_tile(1, 0) == TileColor.WHITE  # Flipped!
    
    # Second update: bullet moves to x=2 where tile is already WHITE (walkable)
    env.update_bullets()
    
    # Bullet should now be at x=2 and tile should remain WHITE (not flipped)
    assert env.bullets[0].x == 2
    assert env.bullets[0].y == 0
    assert env._get_tile(2, 0) == TileColor.WHITE  # Unchanged!


def test_bullet_from_p2_only_flips_white_tiles():
    """Test that P2 bullets only flip WHITE tiles (P2's color)."""
    env = InversusEnv(width=5, height=1)
    env.reset()
    
    # P2 has color WHITE, so they walk on BLACK
    # Set up two adjacent tiles:
    env._set_tile(1, 0, TileColor.WHITE)   # same as P2's color (will be flipped)
    env._set_tile(2, 0, TileColor.BLACK)   # opposite of P2's color (should NOT be flipped)
    
    # Place P2 at x=0
    env.player2.x = 0
    env.player2.y = 0
    
    # Place a bullet owned by P2 at x=0 headed right
    env.bullets = [Bullet(x=0, y=0, dir=Direction.RIGHT, owner=PlayerId.P2)]
    
    # First update: bullet moves to x=1 where tile is WHITE (P2's color)
    env.update_bullets()
    
    # Bullet should now be at x=1 and flip that tile from WHITE -> BLACK
    assert env.bullets[0].x == 1
    assert env._get_tile(1, 0) == TileColor.BLACK  # Flipped!
    
    # Second update: bullet moves to x=2 where tile is already BLACK (walkable for P2)
    env.update_bullets()
    
    # Bullet should now be at x=2 and tile should remain BLACK (not flipped)
    assert env.bullets[0].x == 2
    assert env._get_tile(2, 0) == TileColor.BLACK  # Unchanged!


def test_bullet_does_not_destroy_existing_path():
    """Test that bullets don't destroy paths the owner can already walk on."""
    env = InversusEnv(width=10, height=1)
    env.reset()
    
    # P1 has color BLACK, walks on WHITE
    # Create a path of WHITE tiles (walkable for P1)
    for x in range(3, 7):
        env._set_tile(x, 0, TileColor.WHITE)
    
    # Place P1 at x=2
    env.player1.x = 2
    env.player1.y = 0
    
    # Spawn bullet from P1 going right through the WHITE path
    env.player1.ammo = 6
    env.spawn_bullet(Direction.RIGHT, PlayerId.P1)
    
    # Update bullets multiple times to move through the path
    for _ in range(5):
        env.update_bullets()
    
    # All tiles in the path should still be WHITE (not flipped)
    for x in range(3, 7):
        assert env._get_tile(x, 0) == TileColor.WHITE


def test_bullet_opens_new_path():
    """Test that bullets open new paths by flipping owner's color tiles."""
    env = InversusEnv(width=10, height=1)
    env.reset()
    
    # P1 has color BLACK, walks on WHITE
    # Create a barrier of BLACK tiles (P1's color, not walkable)
    for x in range(3, 7):
        env._set_tile(x, 0, TileColor.BLACK)
    
    # Place P1 at x=2
    env.player1.x = 2
    env.player1.y = 0
    
    # Spawn bullet from P1 going right through the BLACK barrier
    env.player1.ammo = 6
    env.spawn_bullet(Direction.RIGHT, PlayerId.P1)
    
    # Update bullets multiple times to move through the barrier
    for _ in range(5):
        env.update_bullets()
    
    # All tiles in the barrier should now be WHITE (flipped, now walkable)
    for x in range(3, 7):
        assert env._get_tile(x, 0) == TileColor.WHITE

