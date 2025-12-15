"""Core game logic for INVERSUS (no rendering dependencies)."""

from typing import List, Tuple, Optional
import random
from copy import deepcopy

from .game_types import (
    TileColor, Direction, ActionType, Action, PlayerId, PlayerState, Bullet
)
from .config import (
    DEFAULT_WIDTH, DEFAULT_HEIGHT, DEFAULT_PLAYER_COLOR, DEFAULT_PLAYER_START,
    DEFAULT_PLAYER2_COLOR, DEFAULT_PLAYER2_START,
    MAX_AMMO, RELOAD_TICKS_PER_AMMO, WIDE_SHOT_AMMO_COST, make_initial_grid
)


class InversusEnv:
    """INVERSUS game environment with pure logic (no rendering)."""
    
    def __init__(
        self,
        width: Optional[int] = None,
        height: Optional[int] = None,
        player_color: Optional[TileColor] = None,
        seed: Optional[int] = None
    ) -> None:
        """
        Initialize the INVERSUS environment.
        
        Args:
            width: Grid width (defaults to DEFAULT_WIDTH)
            height: Grid height (defaults to DEFAULT_HEIGHT)
            player_color: Player 1 color (defaults to DEFAULT_PLAYER_COLOR)
            seed: Random seed for future extensions
        """
        self.width = width if width is not None else DEFAULT_WIDTH
        self.height = height if height is not None else DEFAULT_HEIGHT
        self.player_color = player_color if player_color is not None else DEFAULT_PLAYER_COLOR
        
        # Initialize RNG if seed is provided
        self.rng = random.Random(seed) if seed is not None else None
        
        # Initialize game state
        self.grid: List[List[TileColor]] = []
        # Two-player support
        self.player1: Optional[PlayerState] = None
        self.player2: Optional[PlayerState] = None
        # Legacy single-player fields (for backward compatibility)
        self._player_x: int = 0
        self._player_y: int = 0
        self.bullets: List[Bullet] = []
        
        self.reset()
    
    def reset(self, seed: Optional[int] = None) -> None:
        """Reset the environment to the initial state with RANDOM spawns."""
        if seed is not None:
            self.rng = random.Random(seed)
        elif self.rng is None:
             self.rng = random.Random()
        
        # 1. Randomize Map Dimensions (Optional, but let's keep 15x10 for now)
        # 2. Initialize Grid
        # Create initial grid (using player 1 color for backward compatibility)
        self.grid = make_initial_grid(self.width, self.height, self.player_color)
        
        # 3. Randomize Player 1 Start
        # Valid range: Keep away from edges to avoid immediate wall-stuck
        p1_x = self.rng.randint(1, self.width - 2)
        p1_y = self.rng.randint(1, self.height - 2)
        
        self.player1 = PlayerState(
            player_id=PlayerId.P1,
            x=p1_x,
            y=p1_y,
            color=self.player_color,
            ammo=MAX_AMMO,
            max_ammo=MAX_AMMO,
            reload_counter=0,
            alive=True
        )
        
        # 4. Randomize Player 2 Start (Ensure distance)
        # Try to find a position at least 4 tiles away
        for _ in range(20): # Try 20 times
            p2_x = self.rng.randint(1, self.width - 2)
            p2_y = self.rng.randint(1, self.height - 2)
            dist = abs(p2_x - p1_x) + abs(p2_y - p1_y)
            if dist > 4:
                break
        
        # Apply P2 Logic...
        # Update grid for P2 (Make P2 area walkable)
        # ... We need to update make_initial_grid logic or manually clear area for P2
        # Let's manually clear the area around P2 so it's fair
        opposite_color = TileColor.WHITE if self.player_color == TileColor.BLACK else TileColor.BLACK
        
        # Create plus-shaped walkable area for P2
        positions = [
            (p2_x, p2_y),      # center
            (p2_x + 1, p2_y),  # right
            (p2_x - 1, p2_y),  # left
            (p2_x, p2_y + 1),  # down
            (p2_x, p2_y - 1),  # up
        ]
        for x, y in positions:
            if 0 <= x < self.width and 0 <= y < self.height:
                self.grid[y][x] = opposite_color
                
        # Also clear area for P1 (since we randomized it)
        # (Re-do P1 area just in case Grid init didn't cover it)
        p1_positions = [
            (p1_x, p1_y),
            (p1_x + 1, p1_y),
            (p1_x - 1, p1_y),
            (p1_x, p1_y + 1),
            (p1_x, p1_y - 1),
        ]
        for x, y in p1_positions:
            if 0 <= x < self.width and 0 <= y < self.height:
                self.grid[y][x] = TileColor.WHITE if self.player1.color == TileColor.BLACK else TileColor.BLACK # Walkable for P1
                
        # Initialize player 2 object
        self.player2 = PlayerState(
            player_id=PlayerId.P2,
            x=p2_x,
            y=p2_y,
            color=DEFAULT_PLAYER2_COLOR,
            ammo=MAX_AMMO,
            max_ammo=MAX_AMMO,
            reload_counter=0,
            alive=True
        )
        
        # Create walkable area for player 2 as well
        opposite_color_p2 = TileColor.WHITE if DEFAULT_PLAYER2_COLOR == TileColor.BLACK else TileColor.BLACK
        positions_p2 = [
            (p2_x, p2_y),
            (p2_x + 1, p2_y),
            (p2_x - 1, p2_y),
            (p2_x, p2_y + 1),
            (p2_x, p2_y - 1),
        ]
        for x, y in positions_p2:
            if 0 <= x < self.width and 0 <= y < self.height:
                self.grid[y][x] = opposite_color_p2
        
        # Legacy fields for backward compatibility (synced with player1)
        # Set private fields directly to avoid property setter during init
        self._player_x = p1_x
        self._player_y = p1_y
        
        # Clear bullets
        self.bullets = []
    
    @property
    def player_x(self) -> int:
        """Legacy property: get player 1 x position."""
        return self.player1.x if self.player1 else self._player_x
    
    @player_x.setter
    def player_x(self, value: int) -> None:
        """Legacy property: set player 1 x position."""
        self._player_x = value
        if self.player1:
            self.player1.x = value
    
    @property
    def player_y(self) -> int:
        """Legacy property: get player 1 y position."""
        return self.player1.y if self.player1 else self._player_y
    
    @player_y.setter
    def player_y(self, value: int) -> None:
        """Legacy property: set player 1 y position."""
        self._player_y = value
        if self.player1:
            self.player1.y = value
        
        # Clear bullets
        self.bullets = []
    
    def get_grid_copy(self) -> List[List[TileColor]]:
        """Return a deep copy of the current grid."""
        return deepcopy(self.grid)
    
    def get_player(self, pid: PlayerId) -> PlayerState:
        """Get the player state for the given player ID."""
        if pid == PlayerId.P1:
            if self.player1 is None:
                raise ValueError("Player 1 not initialized")
            return self.player1
        elif pid == PlayerId.P2:
            if self.player2 is None:
                raise ValueError("Player 2 not initialized")
            return self.player2
        else:
            raise ValueError(f"Invalid player ID: {pid}")
    
    def get_player_position(self, pid: Optional[PlayerId] = None) -> Tuple[int, int]:
        """
        Get the current player position as (x, y).
        
        Args:
            pid: Player ID. If None, returns player 1 position (backward compatibility).
        """
        if pid is None:
            # Backward compatibility: return player 1 position
            return (self.player1.x, self.player1.y) if self.player1 else (self.player_x, self.player_y)
        player = self.get_player(pid)
        return (player.x, player.y)
    
    def is_player_alive(self, pid: PlayerId) -> bool:
        """Check if a player is alive."""
        player = self.get_player(pid)
        return player.alive
    
    def get_bullets(self) -> List[Bullet]:
        """Return a shallow copy of the bullet list."""
        return self.bullets.copy()
    
    def _tile_in_bounds(self, x: int, y: int) -> bool:
        """Check if a tile coordinate is within bounds."""
        return 0 <= x < self.width and 0 <= y < self.height
    
    def _get_tile(self, x: int, y: int) -> TileColor:
        """Get the tile color at the given position."""
        if not self._tile_in_bounds(x, y):
            raise IndexError(f"Tile position ({x}, {y}) is out of bounds")
        return self.grid[y][x]
    
    def _set_tile(self, x: int, y: int, color: TileColor) -> None:
        """Set the tile color at the given position."""
        if not self._tile_in_bounds(x, y):
            raise IndexError(f"Tile position ({x}, {y}) is out of bounds")
        self.grid[y][x] = color
    
    def _is_walkable_for_player(self, x: int, y: int, player_color: TileColor) -> bool:
        """
        Check if a tile is walkable for a player.
        
        A player can only stand on tiles of the opposite color.
        """
        if not self._tile_in_bounds(x, y):
            return False
        tile = self._get_tile(x, y)
        return tile != player_color
    
    def try_move_player(self, direction: Direction, pid: Optional[PlayerId] = None) -> bool:
        """
        Attempt to move a player one tile in the given direction.
        
        Args:
            direction: Direction to move
            pid: Player ID (defaults to P1 for backward compatibility)
            
        Returns:
            True if movement succeeded, False otherwise
        """
        if pid is None:
            pid = PlayerId.P1
        player = self.get_player(pid)
        
        if not player.alive:
            return False
        
        # Compute direction offsets
        dx, dy = 0, 0
        if direction == Direction.UP:
            dy = -1
        elif direction == Direction.DOWN:
            dy = 1
        elif direction == Direction.LEFT:
            dx = -1
        elif direction == Direction.RIGHT:
            dx = 1
        
        # Calculate new position
        nx = player.x + dx
        ny = player.y + dy
        
        # Check bounds
        if not self._tile_in_bounds(nx, ny):
            return False
        
        # Check if walkable
        if not self._is_walkable_for_player(nx, ny, player.color):
            return False
        
        # Move player
        player.x = nx
        player.y = ny
        
        # Legacy fields are synced via properties, no need to update explicitly
        
        return True
    
    def spawn_bullet(self, direction: Direction, pid: Optional[PlayerId] = None) -> bool:
        """
        Spawn a bullet for the given player if they have ammo.
        
        Args:
            direction: Direction the bullet will travel
            pid: Player ID (defaults to P1 for backward compatibility)
            
        Returns:
            True if bullet was spawned, False otherwise
        """
        if pid is None:
            pid = PlayerId.P1
        player = self.get_player(pid)
        
        if not player.alive:
            return False
        
        if player.ammo <= 0:
            return False
        
        # Consume ammo
        player.ammo -= 1
        
        # Create bullet
        bullet = Bullet(x=player.x, y=player.y, dir=direction, owner=pid)
        self.bullets.append(bullet)
        
        return True
    
    def spawn_wide_shot(self, pid: PlayerId, direction: Direction) -> bool:
        """
        Spawn a 3-lane 'wide shot' for the given player:
        - Consumes WIDE_SHOT_AMMO_COST ammo if enough ammo is available.
        - Spawns 3 bullets: center lane + one offset above/below (vertical) or left/right (horizontal).
        - Returns True if bullets were spawned, False otherwise (e.g. not enough ammo).
        
        Args:
            pid: Player ID
            direction: Direction the wide shot will travel
            
        Returns:
            True if at least the center bullet was spawned, False otherwise
        """
        player = self.get_player(pid)
        
        if not player.alive:
            return False
        
        if player.ammo < WIDE_SHOT_AMMO_COST:
            return False
        
        # Consume ammo
        player.ammo -= WIDE_SHOT_AMMO_COST
        
        # Get player position
        px, py = player.x, player.y
        
        # Compute starting positions for the three lanes
        if direction in (Direction.UP, Direction.DOWN):
            # Vertical shot: center at (px, py), side lanes at (px-1, py) and (px+1, py)
            positions = [
                (px, py),      # center
                (px - 1, py),  # left
                (px + 1, py),  # right
            ]
        else:  # LEFT or RIGHT
            # Horizontal shot: center at (px, py), side lanes at (px, py-1) and (px, py+1)
            positions = [
                (px, py),      # center
                (px, py - 1),  # above
                (px, py + 1),  # below
            ]
        
        # Spawn bullets for in-bounds positions
        bullets_spawned = 0
        for x, y in positions:
            if self._tile_in_bounds(x, y):
                bullet = Bullet(x=x, y=y, dir=direction, owner=pid)
                self.bullets.append(bullet)
                bullets_spawned += 1
        
        # Return True if at least center bullet was spawned
        return bullets_spawned > 0
    
    def _reload_ammo(self) -> None:
        """
        Increment reload counters and restore ammo by 1 when enough ticks accumulate,
        up to max_ammo.
        """
        for player in [self.player1, self.player2]:
            if player is None or not player.alive:
                continue
            
            if player.ammo < player.max_ammo:
                player.reload_counter += 1
                
                if player.reload_counter >= RELOAD_TICKS_PER_AMMO:
                    player.ammo += 1
                    player.reload_counter = 0
    
    def update_bullets(self) -> None:
        """
        Advance all bullets by one step with bullet-bullet collision detection:
        - Move each bullet one tile in its direction
        - Remove bullets that go out of bounds
        - Detect collisions: bullets from different owners at same tile cancel each other
        - For non-colliding bullets: flip tiles and check for player hits
        """
        from collections import defaultdict
        
        # Phase 1: Move all bullets and group by target tile
        targets: dict[tuple[int, int], List[Bullet]] = defaultdict(list)
        
        for bullet in self.bullets:
            # Compute direction offsets
            dx, dy = 0, 0
            if bullet.dir == Direction.UP:
                dy = -1
            elif bullet.dir == Direction.DOWN:
                dy = 1
            elif bullet.dir == Direction.LEFT:
                dx = -1
            elif bullet.dir == Direction.RIGHT:
                dx = 1
            
            # Calculate new position
            nx = bullet.x + dx
            ny = bullet.y + dy
            
            # Check bounds
            if not self._tile_in_bounds(nx, ny):
                # Bullet goes out of bounds, discard it
                continue
            
            # Create a new bullet instance at the new position
            moved_bullet = Bullet(x=nx, y=ny, dir=bullet.dir, owner=bullet.owner)
            targets[(nx, ny)].append(moved_bullet)
        
        # Phase 2: Resolve collisions and apply effects
        new_bullets: List[Bullet] = []
        
        for (x, y), bullets_here in targets.items():
            # Check if bullets from different owners are colliding
            owners = {b.owner for b in bullets_here}
            
            if len(owners) > 1:
                # Mixed owners → bullet-bullet collision:
                # - Remove all bullets at this tile (don't add to new_bullets)
                # - Do NOT flip the tile
                # - Do NOT apply player hits at this tile
                continue
            
            # Otherwise, all bullets here have the same owner
            # Process one bullet (they're all identical in effect)
            bullet = bullets_here[0]
            owner = self.get_player(bullet.owner)
            owner_color = owner.color
            
            # Apply tile flip rule ONLY if tile matches owner's forbidden color
            current_color = self._get_tile(x, y)
            if current_color == owner_color:
                flipped_color = TileColor.WHITE if current_color == TileColor.BLACK else TileColor.BLACK
                self._set_tile(x, y, flipped_color)
            
            # Check for player hits (only for non-colliding bullets)
            if self.player1 and self.player1.alive and bullet.owner != PlayerId.P1:
                if x == self.player1.x and y == self.player1.y:
                    self.player1.alive = False
            
            if self.player2 and self.player2.alive and bullet.owner != PlayerId.P2:
                if x == self.player2.x and y == self.player2.y:
                    self.player2.alive = False
            
            # Keep bullet for next update (bullets continue after hits)
            new_bullets.append(bullet)
        
        self.bullets = new_bullets
    
    def is_round_over(self) -> bool:
        """Check if the round is over (at least one player is dead)."""
        alive_p1 = self.player1.alive if self.player1 else False
        alive_p2 = self.player2.alive if self.player2 else False
        return not (alive_p1 and alive_p2)
    
    def get_winner(self) -> Optional[PlayerId]:
        """Get the winner of the round, or None if tie or round not over."""
        if not self.is_round_over():
            return None
        
        alive_p1 = self.player1.alive if self.player1 else False
        alive_p2 = self.player2.alive if self.player2 else False
        
        if not alive_p1 and alive_p2:
            return PlayerId.P2
        if not alive_p2 and alive_p1:
            return PlayerId.P1
        return None  # tie (both dead or both alive somehow)
    
    def step_players(self, action_p1: Action, action_p2: Action) -> None:
        """
        Apply actions for both players in one tick:
        - Resolve P1 action (move or shoot)
        - Resolve P2 action (move or shoot)
        - Then reload ammo
        - Then update bullets (move, flip tiles, check hits)
        
        Args:
            action_p1: Action for player 1
            action_p2: Action for player 2
        """
        # Process P1 action
        if self.player1 and self.player1.alive:
            if action_p1.type == ActionType.MOVE and action_p1.direction is not None:
                self.try_move_player(action_p1.direction, PlayerId.P1)
            elif action_p1.type == ActionType.SHOOT and action_p1.direction is not None:
                self.spawn_bullet(action_p1.direction, PlayerId.P1)
            elif action_p1.type == ActionType.CHARGE_SHOOT and action_p1.direction is not None:
                self.spawn_wide_shot(PlayerId.P1, action_p1.direction)
        
        # Process P2 action
        if self.player2 and self.player2.alive:
            if action_p2.type == ActionType.MOVE and action_p2.direction is not None:
                self.try_move_player(action_p2.direction, PlayerId.P2)
            elif action_p2.type == ActionType.SHOOT and action_p2.direction is not None:
                self.spawn_bullet(action_p2.direction, PlayerId.P2)
            elif action_p2.type == ActionType.CHARGE_SHOOT and action_p2.direction is not None:
                self.spawn_wide_shot(PlayerId.P2, action_p2.direction)
        
        # Reload ammo
        self._reload_ammo()
        
        # Update bullets
        self.update_bullets()
    
    def step(self, action: Action) -> None:
        """
        Apply an action for one tick (backward compatibility wrapper).
        
        This method maintains backward compatibility with single-player API.
        It applies the action to player 1 and does nothing for player 2.
        
        Args:
            action: The action to perform (for player 1)
        """
        # Use step_players with P2 doing nothing
        self.step_players(action, Action(ActionType.NONE, None))
