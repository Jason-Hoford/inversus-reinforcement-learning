"""Tests for bullet-bullet collision and cancellation."""

import pytest
from inversus.core import InversusEnv
from inversus.game_types import TileColor, Direction, PlayerId, Bullet


def test_bullets_cancel_each_other_on_same_tile():
    """Test that bullets from different owners cancel when they meet on the same tile."""
    env = InversusEnv(width=5, height=1)
    env.reset()
    
    # P1 forbids BLACK, P2 forbids WHITE
    # Force a known tile color at collision position, say BLACK at x=2
    env._set_tile(2, 0, TileColor.BLACK)
    
    # Place bullets so that both move into x=2 on the next update
    # P1 bullet moving right from x=1
    b1 = Bullet(x=1, y=0, dir=Direction.RIGHT, owner=PlayerId.P1)
    # P2 bullet moving left from x=3
    b2 = Bullet(x=3, y=0, dir=Direction.LEFT, owner=PlayerId.P2)
    
    env.bullets = [b1, b2]
    
    env.update_bullets()
    
    # Both bullets should be gone (canceled)
    assert len(env.bullets) == 0
    
    # Tile at collision should remain as originally set (BLACK, not flipped)
    assert env._get_tile(2, 0) == TileColor.BLACK


def test_collision_does_not_hit_player():
    """Test that if bullets collide on a tile where a player is standing, the player is not killed."""
    env = InversusEnv(width=5, height=1)
    env.reset()
    
    # Put player1 at x=2
    env.player1.x, env.player1.y = 2, 0
    env.player1.alive = True
    
    # Two bullets colliding at (2, 0) where player1 is standing
    env.bullets = [
        Bullet(x=1, y=0, dir=Direction.RIGHT, owner=PlayerId.P2),
        Bullet(x=3, y=0, dir=Direction.LEFT, owner=PlayerId.P1),
    ]
    
    env.update_bullets()
    
    # Player1 should still be alive (bullets canceled, didn't hit)
    assert env.player1.alive is True
    
    # Both bullets should be gone
    assert len(env.bullets) == 0


def test_same_owner_bullets_do_not_cancel():
    """Test that multiple bullets from the same owner moving into the same tile do not cancel."""
    env = InversusEnv(width=5, height=1)
    env.reset()
    
    # P1 forbids BLACK, so BLACK -> flipped to WHITE
    env._set_tile(2, 0, TileColor.BLACK)
    
    # Two P1 bullets moving right toward x=2
    env.bullets = [
        Bullet(x=1, y=0, dir=Direction.RIGHT, owner=PlayerId.P1),
        Bullet(x=0, y=0, dir=Direction.RIGHT, owner=PlayerId.P1),
    ]
    
    env.update_bullets()
    
    # At least one P1 bullet should remain (they don't cancel each other)
    assert len(env.bullets) >= 1
    # All remaining bullets should be from P1
    for bullet in env.bullets:
        assert bullet.owner == PlayerId.P1
    
    # Tile at x=2 should have been flipped to WHITE (by at least one bullet)
    assert env._get_tile(2, 0) == TileColor.WHITE


def test_collision_preserves_path_already_created():
    """Test that paths created by bullets before collision remain unchanged."""
    env = InversusEnv(width=5, height=1)
    env.reset()
    
    # P1 forbids BLACK, P2 forbids WHITE
    # Set up initial state: all WHITE tiles (so P2 can't walk, but P1 can)
    # Actually, let's use BLACK for P1's side and WHITE for P2's side
    for x in range(5):
        if x < 3:
            env._set_tile(x, 0, TileColor.BLACK)  # P1's side
        else:
            env._set_tile(x, 0, TileColor.WHITE)  # P2's side
    
    # P1 bullet starts at x=0, moving right (will flip x=1 to WHITE, then collide at x=2)
    # P2 bullet starts at x=4, moving left (will flip x=3 to BLACK, then collide at x=2)
    env.bullets = [
        Bullet(x=0, y=0, dir=Direction.RIGHT, owner=PlayerId.P1),
        Bullet(x=4, y=0, dir=Direction.LEFT, owner=PlayerId.P2),
    ]
    
    # First update: bullets move and flip tiles
    env.update_bullets()
    # P1 bullet at x=1, flipped x=1 to WHITE (BLACK -> WHITE, P1's forbidden color)
    # P2 bullet at x=3, flipped x=3 to BLACK (WHITE -> BLACK, P2's forbidden color)
    assert env._get_tile(1, 0) == TileColor.WHITE  # Flipped by P1
    assert env._get_tile(3, 0) == TileColor.BLACK  # Flipped by P2
    
    # Second update: bullets move toward each other
    env.update_bullets()
    # P1 bullet at x=2, P2 bullet at x=2 - they collide
    assert len(env.bullets) == 0  # Both canceled
    
    # Paths created before collision should remain
    assert env._get_tile(1, 0) == TileColor.WHITE  # Still flipped
    assert env._get_tile(3, 0) == TileColor.BLACK  # Still flipped
    
    # Collision tile (x=2) should remain BLACK (not flipped due to collision)
    assert env._get_tile(2, 0) == TileColor.BLACK


def test_multiple_bullets_from_different_owners_cancel():
    """Test that multiple bullets from different owners all cancel when meeting."""
    env = InversusEnv(width=5, height=1)
    env.reset()
    
    # Set collision tile
    env._set_tile(2, 0, TileColor.BLACK)
    
    # Three bullets: two from P1, one from P2, all moving to x=2
    # Need to position them so they all end up at x=2 in one update
    env.bullets = [
        Bullet(x=1, y=0, dir=Direction.RIGHT, owner=PlayerId.P1),  # -> x=2
        Bullet(x=2, y=0, dir=Direction.RIGHT, owner=PlayerId.P1),  # -> x=3, wait no
    ]
    
    # Actually, let's position them correctly: all should move to x=2
    env.bullets = [
        Bullet(x=1, y=0, dir=Direction.RIGHT, owner=PlayerId.P1),  # -> x=2
        Bullet(x=2, y=0, dir=Direction.LEFT, owner=PlayerId.P1),   # -> x=1, no
    ]
    
    # Better approach: have them all converge on x=2 from different directions
    env.bullets = [
        Bullet(x=1, y=0, dir=Direction.RIGHT, owner=PlayerId.P1),  # -> x=2
        Bullet(x=2, y=0, dir=Direction.UP, owner=PlayerId.P1),     # -> (2, -1) out of bounds, no
    ]
    
    # Simplest: two P1 bullets and one P2 bullet all moving to x=2
    # P1 bullets from left, P2 bullet from right
    env.bullets = [
        Bullet(x=1, y=0, dir=Direction.RIGHT, owner=PlayerId.P1),  # -> x=2
        Bullet(x=1, y=0, dir=Direction.RIGHT, owner=PlayerId.P1),  # -> x=2 (same position)
        Bullet(x=3, y=0, dir=Direction.LEFT, owner=PlayerId.P2),  # -> x=2
    ]
    
    env.update_bullets()
    
    # All bullets should be canceled (mixed owners at same tile)
    assert len(env.bullets) == 0
    
    # Tile should remain unchanged
    assert env._get_tile(2, 0) == TileColor.BLACK


def test_bullets_cancel_at_different_positions():
    """Test that bullets can cancel at multiple different positions in the same update."""
    env = InversusEnv(width=7, height=1)
    env.reset()
    
    # Set up two collision points
    env._set_tile(2, 0, TileColor.BLACK)
    env._set_tile(4, 0, TileColor.BLACK)
    
    # Collision at x=2: P1 from left, P2 from right
    # Collision at x=4: P1 from left, P2 from right
    env.bullets = [
        Bullet(x=1, y=0, dir=Direction.RIGHT, owner=PlayerId.P1),  # -> x=2
        Bullet(x=3, y=0, dir=Direction.LEFT, owner=PlayerId.P2),  # -> x=2
        Bullet(x=3, y=0, dir=Direction.RIGHT, owner=PlayerId.P1),  # -> x=4
        Bullet(x=5, y=0, dir=Direction.LEFT, owner=PlayerId.P2),  # -> x=4
    ]
    
    env.update_bullets()
    
    # All bullets should be canceled
    assert len(env.bullets) == 0
    
    # Both collision tiles should remain unchanged
    assert env._get_tile(2, 0) == TileColor.BLACK
    assert env._get_tile(4, 0) == TileColor.BLACK

