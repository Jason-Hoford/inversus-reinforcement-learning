"""Interactive pygame demo for INVERSUS."""

import pygame
from typing import Optional

# Handle both relative imports (when run as module) and absolute imports (when run directly)
try:
    from .core import InversusEnv
    from .game_types import TileColor, Direction, ActionType, Action
except ImportError:
    from inversus.core import InversusEnv
    from inversus.game_types import TileColor, Direction, ActionType, Action


def setup_half_split_grid(env: InversusEnv) -> None:
    """Set up a half-black, half-white grid with vertical split."""
    for y in range(env.height):
        for x in range(env.width):
            if x < env.width // 2:
                env.grid[y][x] = TileColor.BLACK
            else:
                env.grid[y][x] = TileColor.WHITE


def main() -> None:
    """Run the interactive pygame demo."""
    # Initialize pygame
    pygame.init()
    
    # Window settings
    window_width = 900
    window_height = 600
    
    screen = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption("INVERSUS Demo")
    clock = pygame.time.Clock()
    
    # Create environment
    env = InversusEnv()
    setup_half_split_grid(env)
    
    # Calculate tile size
    tile_width = window_width // env.width
    tile_height = window_height // env.height
    
    # Colors - black/white/gray theme
    TILE_BLACK = (10, 10, 10)  # Almost black
    TILE_WHITE = (240, 240, 240)  # Bright white
    PLAYER_COLOR = (200, 200, 200)  # Light gray (P1 walks on white, so lighter)
    PLAYER_OUTLINE = (150, 150, 150)  # Darker outline
    BULLET_COLOR = (120, 120, 120)  # Medium gray
    
    running = True
    
    while running:
        # Handle events
        action: Optional[Action] = None
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_UP:
                    action = Action(ActionType.MOVE, Direction.UP)
                elif event.key == pygame.K_DOWN:
                    action = Action(ActionType.MOVE, Direction.DOWN)
                elif event.key == pygame.K_LEFT:
                    action = Action(ActionType.MOVE, Direction.LEFT)
                elif event.key == pygame.K_RIGHT:
                    action = Action(ActionType.MOVE, Direction.RIGHT)
                elif event.key == pygame.K_w:
                    action = Action(ActionType.SHOOT, Direction.UP)
                elif event.key == pygame.K_s:
                    action = Action(ActionType.SHOOT, Direction.DOWN)
                elif event.key == pygame.K_a:
                    action = Action(ActionType.SHOOT, Direction.LEFT)
                elif event.key == pygame.K_d:
                    action = Action(ActionType.SHOOT, Direction.RIGHT)
        
        # Step environment
        if action is not None:
            env.step(action)
        else:
            # Update bullets even if no action (they keep moving)
            env.update_bullets()
        
        # Clear screen
        screen.fill((128, 128, 128))  # Gray background
        
        # Draw grid
        grid = env.get_grid_copy()
        for y in range(env.height):
            for x in range(env.width):
                tile_color = grid[y][x]
                color = TILE_BLACK if tile_color == TileColor.BLACK else TILE_WHITE
                rect = pygame.Rect(x * tile_width, y * tile_height, tile_width, tile_height)
                pygame.draw.rect(screen, color, rect)
                # Draw grid lines
                pygame.draw.rect(screen, (64, 64, 64), rect, 1)
        
        # Draw player
        px, py = env.get_player_position()
        player_rect = pygame.Rect(
            px * tile_width + tile_width // 4,
            py * tile_height + tile_height // 4,
            tile_width // 2,
            tile_height // 2
        )
        pygame.draw.rect(screen, PLAYER_COLOR, player_rect)
        pygame.draw.rect(screen, PLAYER_OUTLINE, player_rect, 2)
        
        # Draw bullets
        for bullet in env.get_bullets():
            bullet_rect = pygame.Rect(
                bullet.x * tile_width + tile_width // 3,
                bullet.y * tile_height + tile_height // 3,
                tile_width // 3,
                tile_height // 3
            )
            pygame.draw.rect(screen, BULLET_COLOR, bullet_rect)
        
        # Update display
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()


if __name__ == "__main__":
    main()
