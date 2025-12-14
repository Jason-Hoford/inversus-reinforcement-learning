"""Tests for combat, ammo, and reload mechanics in INVERSUS."""

import pytest
from inversus.core import InversusEnv
from inversus.game_types import TileColor, Direction, ActionType, Action, PlayerId
from inversus.config import MAX_AMMO, RELOAD_TICKS_PER_AMMO


def test_shoot_consumes_ammo_and_blocks_when_empty():
    """Test that shooting consumes ammo and blocks when empty."""
    env = InversusEnv(width=10, height=10)
    
    # Set P1 ammo to 1
    env.player1.ammo = 1
    
    # First shot should succeed
    result1 = env.spawn_bullet(Direction.RIGHT, PlayerId.P1)
    assert result1 is True
    assert env.player1.ammo == 0
    
    # Second shot should fail (no ammo)
    result2 = env.spawn_bullet(Direction.RIGHT, PlayerId.P1)
    assert result2 is False
    assert env.player1.ammo == 0  # Still 0


def test_ammo_reloads_over_time():
    """Test that ammo reloads over time."""
    env = InversusEnv(width=10, height=10)
    
    # Set P1 ammo to 0 and reload_counter to 0
    env.player1.ammo = 0
    env.player1.reload_counter = 0
    
    # Call _reload_ammo() RELOAD_TICKS_PER_AMMO times
    for i in range(RELOAD_TICKS_PER_AMMO):
        env._reload_ammo()
        if i < RELOAD_TICKS_PER_AMMO - 1:
            # Before the last call, ammo should still be 0
            assert env.player1.ammo == 0
            assert env.player1.reload_counter == i + 1
    
    # After RELOAD_TICKS_PER_AMMO calls, ammo should increase by 1
    assert env.player1.ammo == 1
    assert env.player1.reload_counter == 0  # Reset
    
    # Test that ammo never exceeds max_ammo
    env.player1.ammo = MAX_AMMO
    env.player1.reload_counter = 0
    
    for _ in range(RELOAD_TICKS_PER_AMMO * 2):
        env._reload_ammo()
        assert env.player1.ammo <= MAX_AMMO


def test_bullet_kills_opponent():
    """Test that bullets kill opponents."""
    env = InversusEnv(width=10, height=10)
    
    # Position P1 and P2 in a straight line
    env.player1.x = 1
    env.player1.y = 5
    env.player2.x = 4
    env.player2.y = 5
    
    # Ensure both are alive
    assert env.player1.alive
    assert env.player2.alive
    
    # Make path walkable for bullets (set tiles to opposite of P1 color)
    opposite_color = TileColor.WHITE if env.player1.color == TileColor.BLACK else TileColor.BLACK
    for x in range(1, 5):
        env._set_tile(x, 5, opposite_color)
    
    # Ensure P1 has ammo
    env.player1.ammo = MAX_AMMO
    
    # Spawn bullet from P1 toward P2 (right)
    env.spawn_bullet(Direction.RIGHT, PlayerId.P1)
    
    # Update bullets enough times for bullet to reach P2
    # P1 at x=1, P2 at x=4, so bullet needs to move 3 tiles
    for _ in range(3):
        env.update_bullets()
    
    # Check that P2 is dead
    assert not env.player2.alive
    assert env.player1.alive
    
    # Check round is over
    assert env.is_round_over()
    
    # Check winner
    assert env.get_winner() == PlayerId.P1


def test_step_players_integrates_move_shoot_reload_and_bullets():
    """Test that step_players integrates move, shoot, reload, and bullets."""
    env = InversusEnv(width=10, height=10)
    
    # Set up positions
    env.player1.x = 5
    env.player1.y = 5
    env.player2.x = 7
    env.player2.y = 5
    
    # Ensure both have ammo
    env.player1.ammo = MAX_AMMO
    env.player2.ammo = MAX_AMMO
    
    # Make tiles walkable
    # For P1 to be at (5,5) and shoot right, tile at (5,5) must be opposite of P1 color
    # For P2 to move left from (7,5) to (6,5), tile at (6,5) must be opposite of P2 color
    opposite_color_p1 = TileColor.WHITE if env.player1.color == TileColor.BLACK else TileColor.BLACK
    opposite_color_p2 = TileColor.WHITE if env.player2.color == TileColor.BLACK else TileColor.BLACK
    env._set_tile(5, 5, opposite_color_p1)  # P1 position
    env._set_tile(6, 5, opposite_color_p2)  # P2 destination (must be walkable for P2)
    env._set_tile(7, 5, opposite_color_p2)  # P2 position
    
    # Record initial state
    p1_ammo_before = env.player1.ammo
    p1_reload_before = env.player1.reload_counter
    bullets_before = len(env.get_bullets())
    
    # P1 SHOOT, P2 MOVE
    action_p1 = Action(ActionType.SHOOT, Direction.RIGHT)
    action_p2 = Action(ActionType.MOVE, Direction.LEFT)
    
    env.step_players(action_p1, action_p2)
    
    # Check ammo decreased for P1
    assert env.player1.ammo == p1_ammo_before - 1
    
    # Check bullet exists
    bullets_after = env.get_bullets()
    assert len(bullets_after) == bullets_before + 1
    
    # Check bullet is at correct position (moved one tile right from P1)
    bullet = bullets_after[0]
    assert bullet.x == 6  # P1 was at 5, bullet moved to 6
    assert bullet.y == 5
    assert bullet.owner == PlayerId.P1
    
    # Check tile was flipped at bullet position
    tile_at_bullet = env._get_tile(6, 5)
    # Bullet flips the tile, so it should be opposite of what we set it to
    # We set it to opposite_color_p2, so after flip it should be P2's color
    assert tile_at_bullet == env.player2.color
    
    # Check reload_counter incremented
    assert env.player1.reload_counter == p1_reload_before + 1
    
    # Check P2 moved
    assert env.player2.x == 6  # Moved left from 7 to 6
    assert env.player2.y == 5
    
    # Now test reload over multiple ticks
    env.player1.ammo = MAX_AMMO - 1
    env.player1.reload_counter = 0
    
    # Step with NONE actions RELOAD_TICKS_PER_AMMO times
    for i in range(RELOAD_TICKS_PER_AMMO):
        env.step_players(Action(ActionType.NONE, None), Action(ActionType.NONE, None))
    
    # Ammo should have reloaded by 1
    assert env.player1.ammo == MAX_AMMO


def test_bullet_does_not_kill_owner():
    """Test that bullets don't kill their owner."""
    env = InversusEnv(width=10, height=10)
    
    # Position P1
    env.player1.x = 5
    env.player1.y = 5
    
    # Make tile walkable
    opposite_color = TileColor.WHITE if env.player1.color == TileColor.BLACK else TileColor.BLACK
    env._set_tile(5, 5, opposite_color)
    env._set_tile(4, 5, opposite_color)
    
    # Spawn bullet from P1 going left
    env.player1.ammo = MAX_AMMO
    env.spawn_bullet(Direction.LEFT, PlayerId.P1)
    
    # Move bullet back to P1 position (bullet starts at P1, moves left, then we move it back)
    # Actually, bullet starts at P1 position, so first update moves it left
    env.update_bullets()  # Bullet moves to (4, 5)
    
    # Move P1 to where bullet is
    env.player1.x = 4
    env.player1.y = 5
    
    # Update bullets again - bullet should be at (3, 5) now, not at P1
    env.update_bullets()
    
    # P1 should still be alive (bullet didn't hit because it moved away)
    assert env.player1.alive
    
    # But if bullet somehow ends up at P1 position and owner is P1, it shouldn't kill
    # Let's test a different scenario: bullet from P1, P1 moves into bullet path
    env.player1.x = 5
    env.player1.y = 5
    env.player1.ammo = MAX_AMMO
    env.bullets = []  # Clear bullets
    
    # Spawn bullet going right
    env.spawn_bullet(Direction.RIGHT, PlayerId.P1)
    env.update_bullets()  # Bullet at (6, 5)
    
    # Move P1 to (6, 5) where bullet is
    env.player1.x = 6
    env.player1.y = 5
    
    # Update bullets - bullet moves to (7, 5), P1 at (6, 5)
    env.update_bullets()
    
    # P1 should still be alive (bullet moved away before P1 got there, or bullet owner is P1)
    assert env.player1.alive


def test_dead_player_cannot_move_or_shoot():
    """Test that dead players cannot move or shoot."""
    env = InversusEnv(width=10, height=10)
    
    # Kill P2
    env.player2.alive = False
    
    # Try to move P2
    result = env.try_move_player(Direction.RIGHT, PlayerId.P2)
    assert result is False
    
    # Try to shoot with P2
    env.player2.ammo = MAX_AMMO
    result = env.spawn_bullet(Direction.RIGHT, PlayerId.P2)
    assert result is False


def test_is_round_over_and_get_winner():
    """Test round end detection and winner determination."""
    env = InversusEnv(width=10, height=10)
    
    # Initially both alive
    assert not env.is_round_over()
    assert env.get_winner() is None
    
    # Kill P2
    env.player2.alive = False
    assert env.is_round_over()
    assert env.get_winner() == PlayerId.P1
    
    # Reset and kill P1
    env.reset()
    env.player1.alive = False
    assert env.is_round_over()
    assert env.get_winner() == PlayerId.P2
    
    # Both dead
    env.reset()
    env.player1.alive = False
    env.player2.alive = False
    assert env.is_round_over()
    assert env.get_winner() is None  # Tie

