"""Configuration defaults for the INVERSUS game."""

from typing import List
from .game_types import TileColor


DEFAULT_WIDTH = 15
DEFAULT_HEIGHT = 10
DEFAULT_PLAYER_COLOR = TileColor.BLACK
DEFAULT_PLAYER_START = (1, 1)  # (x, y) top-left corner
DEFAULT_PLAYER2_COLOR = TileColor.WHITE  # opposite of P1
DEFAULT_PLAYER2_START = (5, 5)  # (x, y) Closer for faster training (Curriculum)

# Ammo mechanics
MAX_AMMO = 6
RELOAD_TICKS_PER_AMMO = 30  # ~0.5 sec if running at 60 FPS
WIDE_SHOT_AMMO_COST = 3  # Ammo cost for charged/wide shots (3-tile wide beam)


def make_initial_grid(width: int, height: int, player_color: TileColor) -> List[List[TileColor]]:
    """
    Create an initial grid with a plus-shaped walkable area around the player start.
    
    Args:
        width: Grid width
        height: Grid height
        player_color: The player's color (opposite color tiles are walkable)
        
    Returns:
        A 2D list [height][width] of TileColor values
    """
    # Fill grid with player color
    grid = [[player_color for _ in range(width)] for _ in range(height)]
    
    # Get player start position
    x0, y0 = DEFAULT_PLAYER_START
    
    # Ensure start position is within bounds
    if 0 <= x0 < width and 0 <= y0 < height:
        # Get opposite color
        opposite_color = TileColor.WHITE if player_color == TileColor.BLACK else TileColor.BLACK
        
        # Create plus-shaped walkable area
        positions = [
            (x0, y0),      # center
            (x0 + 1, y0),  # right
            (x0 - 1, y0),  # left
            (x0, y0 + 1),  # down
            (x0, y0 - 1),  # up
        ]
        
        for x, y in positions:
            if 0 <= x < width and 0 <= y < height:
                grid[y][x] = opposite_color
    
    return grid

