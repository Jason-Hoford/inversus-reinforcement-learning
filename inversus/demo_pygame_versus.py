"""Interactive pygame demo for INVERSUS 2-player versus mode."""

import pygame
import random
from typing import Optional

# Handle both relative imports (when run as module) and absolute imports (when run directly)
try:
    from .core import InversusEnv
    from .game_types import TileColor, Direction, ActionType, Action, PlayerId
except ImportError:
    from inversus.core import InversusEnv
    from inversus.game_types import TileColor, Direction, ActionType, Action, PlayerId


def setup_half_split_grid(env: InversusEnv) -> None:
    """Set up a half-black, half-white grid with vertical split."""
    for y in range(env.height):
        for x in range(env.width):
            if x < env.width // 2:
                env.grid[y][x] = TileColor.BLACK
            else:
                env.grid[y][x] = TileColor.WHITE


def choose_ai_action(env: InversusEnv) -> Action:
    """
    Very naive AI:
    - If roughly aligned horizontally or vertically with P1 and has ammo, shoot.
    - If has enough ammo for wide shot and aligned, sometimes use CHARGE_SHOOT.
    - Otherwise, try random legal moves; if none, ActionType.NONE.
    """
    from inversus.config import WIDE_SHOT_AMMO_COST
    
    p1 = env.get_player(PlayerId.P1)
    p2 = env.get_player(PlayerId.P2)
    
    if not p2.alive:
        return Action(ActionType.NONE, None)
    
    # If same x coordinate and has ammo, try to shoot vertically
    if p1.x == p2.x and p2.ammo >= WIDE_SHOT_AMMO_COST and random.random() < 0.3:
        # 30% chance to use wide shot if enough ammo
        if p1.y < p2.y:
            return Action(ActionType.CHARGE_SHOOT, Direction.UP)
        elif p1.y > p2.y:
            return Action(ActionType.CHARGE_SHOOT, Direction.DOWN)
    elif p1.x == p2.x and p2.ammo > 0:
        if p1.y < p2.y:
            return Action(ActionType.SHOOT, Direction.UP)
        elif p1.y > p2.y:
            return Action(ActionType.SHOOT, Direction.DOWN)
    
    # If same y coordinate and has ammo, try to shoot horizontally
    if p1.y == p2.y and p2.ammo >= WIDE_SHOT_AMMO_COST and random.random() < 0.3:
        # 30% chance to use wide shot if enough ammo
        if p1.x < p2.x:
            return Action(ActionType.CHARGE_SHOOT, Direction.LEFT)
        elif p1.x > p2.x:
            return Action(ActionType.CHARGE_SHOOT, Direction.RIGHT)
    elif p1.y == p2.y and p2.ammo > 0:
        if p1.x < p2.x:
            return Action(ActionType.SHOOT, Direction.LEFT)
        elif p1.x > p2.x:
            return Action(ActionType.SHOOT, Direction.RIGHT)
    
    # Otherwise, try random move
    directions = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]
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
    
    # No valid move found
    return Action(ActionType.NONE, None)


def main() -> None:
    """Run the interactive pygame demo."""
    # Initialize pygame
    pygame.init()
    
    # Window settings
    window_width = 900
    window_height = 600
    
    screen = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption("INVERSUS Versus Demo")
    clock = pygame.time.Clock()
    
    # Font for text
    font = pygame.font.Font(None, 36)
    small_font = pygame.font.Font(None, 24)
    
    # Create environment
    env = InversusEnv()
    setup_half_split_grid(env)
    
    # Calculate tile size
    tile_width = window_width // env.width
    tile_height = window_height // env.height
    
    # Colors - black/white/gray theme
    TILE_BLACK = (10, 10, 10)  # Almost black
    TILE_WHITE = (240, 240, 240)  # Bright white
    PLAYER1_COLOR = (200, 200, 200)  # Light gray (P1 walks on white, so lighter)
    PLAYER1_OUTLINE = (150, 150, 150)  # Darker outline
    PLAYER2_COLOR = (40, 40, 40)  # Dark gray (P2 walks on black, so darker)
    PLAYER2_OUTLINE = (80, 80, 80)  # Lighter outline
    BULLET_P1_COLOR = (140, 140, 140)  # Medium-light gray for P1 bullets
    BULLET_P2_COLOR = (100, 100, 100)  # Medium-dark gray for P2 bullets
    AMMO_BAR_BG = (64, 64, 64)
    AMMO_BAR_FG = (0, 255, 0)
    
    running = True
    waiting_for_reset = False
    
    while running:
        # Handle events
        action_p1: Optional[Action] = None
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_SPACE and waiting_for_reset:
                    env.reset()
                    setup_half_split_grid(env)
                    waiting_for_reset = False
                elif not waiting_for_reset:
                    # Player 1 controls
                    # Priority: CHARGE_SHOOT > SHOOT > MOVE
                    if event.key == pygame.K_i:
                        action_p1 = Action(ActionType.CHARGE_SHOOT, Direction.UP)
                    elif event.key == pygame.K_k:
                        action_p1 = Action(ActionType.CHARGE_SHOOT, Direction.DOWN)
                    elif event.key == pygame.K_j:
                        action_p1 = Action(ActionType.CHARGE_SHOOT, Direction.LEFT)
                    elif event.key == pygame.K_l:
                        action_p1 = Action(ActionType.CHARGE_SHOOT, Direction.RIGHT)
                    elif event.key == pygame.K_UP:
                        action_p1 = Action(ActionType.MOVE, Direction.UP)
                    elif event.key == pygame.K_DOWN:
                        action_p1 = Action(ActionType.MOVE, Direction.DOWN)
                    elif event.key == pygame.K_LEFT:
                        action_p1 = Action(ActionType.MOVE, Direction.LEFT)
                    elif event.key == pygame.K_RIGHT:
                        action_p1 = Action(ActionType.MOVE, Direction.RIGHT)
                    elif event.key == pygame.K_w:
                        action_p1 = Action(ActionType.SHOOT, Direction.UP)
                    elif event.key == pygame.K_s:
                        action_p1 = Action(ActionType.SHOOT, Direction.DOWN)
                    elif event.key == pygame.K_a:
                        action_p1 = Action(ActionType.SHOOT, Direction.LEFT)
                    elif event.key == pygame.K_d:
                        action_p1 = Action(ActionType.SHOOT, Direction.RIGHT)
        
        # Step environment only if both players are alive
        if not waiting_for_reset:
            if action_p1 is None:
                action_p1 = Action(ActionType.NONE, None)
            
            action_p2 = choose_ai_action(env)
            
            env.step_players(action_p1, action_p2)
            
            # Check if round is over
            if env.is_round_over():
                waiting_for_reset = True
        
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
        
        # Draw player 1 if alive
        p1 = env.get_player(PlayerId.P1)
        if p1.alive:
            player1_rect = pygame.Rect(
                p1.x * tile_width + tile_width // 4,
                p1.y * tile_height + tile_height // 4,
                tile_width // 2,
                tile_height // 2
            )
            pygame.draw.rect(screen, PLAYER1_COLOR, player1_rect)
            pygame.draw.rect(screen, PLAYER1_OUTLINE, player1_rect, 2)
        
        # Draw player 2 if alive
        p2 = env.get_player(PlayerId.P2)
        if p2.alive:
            player2_rect = pygame.Rect(
                p2.x * tile_width + tile_width // 4,
                p2.y * tile_height + tile_height // 4,
                tile_width // 2,
                tile_height // 2
            )
            pygame.draw.rect(screen, PLAYER2_COLOR, player2_rect)
            pygame.draw.rect(screen, PLAYER2_OUTLINE, player2_rect, 2)
        
        # Draw bullets with owner-based colors
        for bullet in env.get_bullets():
            bullet_color = BULLET_P1_COLOR if bullet.owner == PlayerId.P1 else BULLET_P2_COLOR
            bullet_rect = pygame.Rect(
                bullet.x * tile_width + tile_width // 3,
                bullet.y * tile_height + tile_height // 3,
                tile_width // 3,
                tile_height // 3
            )
            pygame.draw.rect(screen, bullet_color, bullet_rect)
        
        # Draw ammo bars
        # Player 1 ammo (bottom left)
        ammo_bar_width = 150
        ammo_bar_height = 20
        ammo_bar_x = 10
        ammo_bar_y = window_height - ammo_bar_height - 10
        
        # Background
        pygame.draw.rect(screen, AMMO_BAR_BG, 
                        (ammo_bar_x, ammo_bar_y, ammo_bar_width, ammo_bar_height))
        # Foreground (filled portion)
        if p1.alive:
            ammo_ratio = p1.ammo / p1.max_ammo
            filled_width = int(ammo_bar_width * ammo_ratio)
            pygame.draw.rect(screen, AMMO_BAR_FG,
                           (ammo_bar_x, ammo_bar_y, filled_width, ammo_bar_height))
        # Text
        ammo_text = small_font.render(f"P1: {p1.ammo}/{p1.max_ammo}", True, (255, 255, 255))
        screen.blit(ammo_text, (ammo_bar_x + 5, ammo_bar_y + 2))
        
        # Player 2 ammo (bottom right)
        ammo_bar_x2 = window_width - ammo_bar_width - 10
        # Background
        pygame.draw.rect(screen, AMMO_BAR_BG,
                        (ammo_bar_x2, ammo_bar_y, ammo_bar_width, ammo_bar_height))
        # Foreground
        if p2.alive:
            ammo_ratio = p2.ammo / p2.max_ammo
            filled_width = int(ammo_bar_width * ammo_ratio)
            pygame.draw.rect(screen, AMMO_BAR_FG,
                           (ammo_bar_x2, ammo_bar_y, filled_width, ammo_bar_height))
        # Text
        ammo_text2 = small_font.render(f"P2: {p2.ammo}/{p2.max_ammo}", True, (255, 255, 255))
        screen.blit(ammo_text2, (ammo_bar_x2 + 5, ammo_bar_y + 2))
        
        # Draw round end message
        if waiting_for_reset:
            winner = env.get_winner()
            if winner == PlayerId.P1:
                message = "P1 WINS! Press SPACE to reset"
                color = PLAYER1_COLOR
            elif winner == PlayerId.P2:
                message = "P2 WINS! Press SPACE to reset"
                color = PLAYER2_COLOR
            else:
                message = "TIE! Press SPACE to reset"
                color = (255, 255, 255)
            
            text_surface = font.render(message, True, color)
            text_rect = text_surface.get_rect(center=(window_width // 2, window_height // 2))
            screen.blit(text_surface, text_rect)
        
        # Update display
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()


if __name__ == "__main__":
    main()
