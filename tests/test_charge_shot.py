"""Tests for charged/wide shot mechanics."""

import pytest
from inversus.core import InversusEnv
from inversus.game_types import TileColor, Direction, ActionType, Action, PlayerId
from inversus.config import MAX_AMMO, WIDE_SHOT_AMMO_COST


def test_charge_shot_spawns_three_bullets_and_consumes_ammo():
    """Test that a charge shot spawns three bullets and consumes the correct ammo."""
    env = InversusEnv(width=7, height=7)
    env.reset()
    
    # Ensure P1 has full ammo
    env.player1.ammo = MAX_AMMO
    
    # Place P1 in center so side lanes are in bounds
    env.player1.x, env.player1.y = 3, 3
    
    # Fire a vertical wide shot (UP)
    success = env.spawn_wide_shot(PlayerId.P1, Direction.UP)
    assert success is True
    
    # Ammo should be reduced by WIDE_SHOT_AMMO_COST
    assert env.player1.ammo == MAX_AMMO - WIDE_SHOT_AMMO_COST
    
    # There should be 3 bullets
    assert len(env.bullets) == 3
    
    # For vertical shot (UP), all bullets should have same y, different x
    xs = sorted(b.x for b in env.bullets)
    ys = {b.y for b in env.bullets}
    dirs = {b.dir for b in env.bullets}
    owners = {b.owner for b in env.bullets}
    
    assert xs == [2, 3, 4]  # Left, center, right
    assert ys == {3}  # All at same y
    assert dirs == {Direction.UP}
    assert owners == {PlayerId.P1}


def test_charge_shot_horizontal_spawns_correctly():
    """Test that a horizontal charge shot spawns bullets correctly."""
    env = InversusEnv(width=7, height=7)
    env.reset()
    
    env.player1.ammo = MAX_AMMO
    env.player1.x, env.player1.y = 3, 3
    
    # Fire a horizontal wide shot (RIGHT)
    success = env.spawn_wide_shot(PlayerId.P1, Direction.RIGHT)
    assert success is True
    
    assert len(env.bullets) == 3
    
    # For horizontal shot (RIGHT), all bullets should have same x, different y
    xs = {b.x for b in env.bullets}
    ys = sorted(b.y for b in env.bullets)
    dirs = {b.dir for b in env.bullets}
    
    assert xs == {3}  # All at same x
    assert ys == [2, 3, 4]  # Above, center, below
    assert dirs == {Direction.RIGHT}


def test_charge_shot_requires_enough_ammo():
    """Test that charge shot requires enough ammo."""
    env = InversusEnv(width=7, height=7)
    env.reset()
    
    env.player1.ammo = WIDE_SHOT_AMMO_COST - 1
    env.player1.x, env.player1.y = 3, 3
    
    success = env.spawn_wide_shot(PlayerId.P1, Direction.RIGHT)
    assert success is False
    assert env.player1.ammo == WIDE_SHOT_AMMO_COST - 1  # Unchanged
    assert len(env.bullets) == 0


def test_charge_shot_respects_bounds_for_side_lanes():
    """Test that charge shot respects bounds for side lanes."""
    env = InversusEnv(width=7, height=7)
    env.reset()
    
    env.player1.ammo = MAX_AMMO
    
    # Place P1 at left edge (x=0) so left side lane would be out of bounds
    env.player1.x, env.player1.y = 0, 3
    
    # Fire vertical wide shot (UP)
    success = env.spawn_wide_shot(PlayerId.P1, Direction.UP)
    assert success is True  # Should succeed as long as center bullet spawns
    
    # Should have 2 bullets (center and right, left is out of bounds)
    assert len(env.bullets) == 2
    
    xs = sorted(b.x for b in env.bullets)
    assert xs == [0, 1]  # Center (x=0) and right (x=1), no left (x=-1 out of bounds)
    
    # Place P1 at top edge (y=0) for horizontal shot
    env.bullets = []
    env.player1.x, env.player1.y = 3, 0
    env.player1.ammo = MAX_AMMO
    
    # Fire horizontal wide shot (RIGHT)
    success = env.spawn_wide_shot(PlayerId.P1, Direction.RIGHT)
    assert success is True
    
    # Should have 2 bullets (center and below, above is out of bounds)
    assert len(env.bullets) == 2
    
    ys = sorted(b.y for b in env.bullets)
    assert ys == [0, 1]  # Center (y=0) and below (y=1), no above (y=-1 out of bounds)


def test_charge_shot_integration_with_step_players():
    """Test that charge shot works correctly when called via step_players."""
    env = InversusEnv(width=7, height=7)
    env.reset()
    
    env.player1.ammo = MAX_AMMO
    env.player1.x, env.player1.y = 3, 3
    
    # Use CHARGE_SHOOT action
    action = Action(ActionType.CHARGE_SHOOT, Direction.UP)
    env.step_players(action, Action(ActionType.NONE, None))
    
    # Should have spawned 3 bullets
    assert len(env.bullets) == 3
    
    # Ammo should be reduced
    assert env.player1.ammo == MAX_AMMO - WIDE_SHOT_AMMO_COST
    
    # Bullets should have already moved once (step_players calls update_bullets)
    # All bullets should have moved up by 1 from their starting positions
    for bullet in env.bullets:
        assert bullet.y == 2  # Moved from y=3 to y=2
        assert bullet.x in [2, 3, 4]  # Still at their x positions
    
    # Update bullets again to verify they continue moving
    env.update_bullets()
    for bullet in env.bullets:
        assert bullet.y == 1  # Moved from y=2 to y=1
        assert bullet.x in [2, 3, 4]  # Still at their x positions


def test_charge_shot_bullets_behave_like_normal_bullets():
    """Test that charge shot bullets behave like normal bullets (flip tiles, collide, etc.)."""
    env = InversusEnv(width=7, height=7)
    env.reset()
    
    env.player1.ammo = MAX_AMMO
    env.player1.x, env.player1.y = 3, 3
    
    # Set up tiles: all BLACK (P1's forbidden color)
    for y in range(7):
        for x in range(7):
            env._set_tile(x, y, TileColor.BLACK)
    
    # Fire charge shot UP
    env.spawn_wide_shot(PlayerId.P1, Direction.UP)
    
    # Update bullets once
    env.update_bullets()
    
    # All three bullets should have moved and flipped their tiles to WHITE
    for bullet in env.bullets:
        assert bullet.y == 2  # Moved up
        assert env._get_tile(bullet.x, bullet.y) == TileColor.WHITE  # Flipped


def test_charge_shot_cannot_be_used_by_dead_player():
    """Test that dead players cannot use charge shots."""
    env = InversusEnv(width=7, height=7)
    env.reset()
    
    env.player1.ammo = MAX_AMMO
    env.player1.alive = False
    
    success = env.spawn_wide_shot(PlayerId.P1, Direction.UP)
    assert success is False
    assert len(env.bullets) == 0

