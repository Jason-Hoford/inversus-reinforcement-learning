
import argparse
import time
import torch
import numpy as np
import pygame
import sys
import os

from inversus.core import InversusEnv
from inversus.game_types import PlayerId, Action, ActionType, Direction, TileColor
from inversus_rl.policies import InversusCNNPolicy
from inversus_rl.env_wrappers import build_observation, discrete_to_action, dummy_opponent_policy
from inversus.config import DEFAULT_WIDTH, DEFAULT_HEIGHT

class GamePlayer:
    def __init__(self, model_path: str, mode: str = "vs_user", fps: int = 60, speed: float = 1.0, opponent_difficulty: str = "easy"):
        self.mode = mode  # "vs_user" (P2 is Human), "ai_vs_ai" (P2 is same model), "vs_dummy" (P2 is dummy)
        self.fps = fps
        self.speed = speed
        self.opponent_difficulty = opponent_difficulty
        self.device = "cpu"
        
        # Load environment
        self.env = InversusEnv(width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT) # Use defaults to match training
        
        # Load model for P1 (Always AI)
        self.policy_p1 = self._load_policy(model_path)
        
        # Load model for P2 (If AI vs AI)
        self.policy_p2 = None
        if self.mode == "ai_vs_ai":
            self.policy_p2 = self.policy_p1 # Share same policy
        
        # Pygame setup
        pygame.init()
        self.cell_size = 30
        self.screen_width = self.env.width * self.cell_size
        self.screen_height = self.env.height * self.cell_size
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption(f"INVERSUS AI - Mode: {mode} ({opponent_difficulty}) | Speed: {speed:.1f}x")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 24)

    def _load_policy(self, model_path: str) -> InversusCNNPolicy:
        print(f"Loading model from {model_path}...")
        # Infer dimensions from env
        dummy_obs, dummy_extra = build_observation(self.env, PlayerId.P1)
        channels = dummy_obs.shape[0]
        height = dummy_obs.shape[1]
        width = dummy_obs.shape[2]
        extra_dim = dummy_extra.shape[0]
        
        policy = InversusCNNPolicy(channels, height, width, extra_dim).to(self.device)
        policy.load_state_dict(torch.load(model_path, map_location=self.device))
        policy.eval()
        return policy

    def _get_ai_action(self, policy: InversusCNNPolicy, player_id: PlayerId) -> Action:
        obs, extra = build_observation(self.env, player_id)
        
        with torch.no_grad():
            grid_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            extra_t = torch.FloatTensor(extra).unsqueeze(0).to(self.device)
            logits, _ = policy(grid_t, extra_t)
            action_id = torch.argmax(logits, dim=1).item()
            
        return discrete_to_action(action_id)

    def _get_human_action(self) -> Action:
        # P2 controls: Arrows for Move, L for Shoot
        keys = pygame.key.get_pressed()
        
        # Check for Charge Modifier (Shift)
        is_charge = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]
        
        # Movement
        if keys[pygame.K_UP]:
            return Action(ActionType.MOVE, Direction.UP)
        elif keys[pygame.K_DOWN]:
            return Action(ActionType.MOVE, Direction.DOWN)
        elif keys[pygame.K_LEFT]:
            return Action(ActionType.MOVE, Direction.LEFT)
        elif keys[pygame.K_RIGHT]:
            return Action(ActionType.MOVE, Direction.RIGHT)
        
        # Shooting (IJKL)
        # Shift + Shoot = Charge Shot (Board Shot)
        action_type = ActionType.CHARGE_SHOOT if is_charge else ActionType.SHOOT
        
        if keys[pygame.K_i]:
            return Action(action_type, Direction.UP)
        elif keys[pygame.K_k]:
            return Action(action_type, Direction.DOWN)
        elif keys[pygame.K_j]:
            return Action(action_type, Direction.LEFT)
        elif keys[pygame.K_l]:
            return Action(action_type, Direction.RIGHT)
            
        return Action(ActionType.NONE, None)

    def run(self):
        running = True
        paused = False
        
        self.env.reset()
        
        while running:
            # Event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                    elif event.key == pygame.K_r:
                        self.env.reset()
                    elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                        self.speed = min(5.0, self.speed + 0.1)
                        pygame.display.set_caption(f"INVERSUS AI - Mode: {self.mode} | Speed: {self.speed:.1f}x")
                    elif event.key == pygame.K_MINUS:
                        self.speed = max(0.1, self.speed - 0.1)
                        pygame.display.set_caption(f"INVERSUS AI - Mode: {self.mode} | Speed: {self.speed:.1f}x")

            if not paused:
                # 1. Get Actions
                # P1 is always AI
                if self.env.player1.alive:
                    action_p1 = self._get_ai_action(self.policy_p1, PlayerId.P1)
                else:
                    action_p1 = Action(ActionType.NONE, None)
                
                # P2 depends on mode
                if self.env.player2.alive:
                    if self.mode == "vs_user":
                        action_p2 = self._get_human_action()
                    elif self.mode == "ai_vs_ai":
                        action_p2 = self._get_ai_action(self.policy_p2, PlayerId.P2)
                    else: # vs_dummy
                        action_p2 = dummy_opponent_policy(self.env, difficulty=self.opponent_difficulty)
                else:
                    action_p2 = Action(ActionType.NONE, None)

                # 2. Step Environment
                self.env.step_players(action_p1, action_p2)
                
                # Check game over
                if self.env.is_round_over():
                    winner = self.env.get_winner()
                    print(f"Round Over! Winner: {winner}")
                    time.sleep(1)
                    self.env.reset()

            # 3. Render
            self._render()
            
            # Control speed
            self.clock.tick(int(self.fps * self.speed))

        pygame.quit()

    def _render(self):
        self.screen.fill((50, 50, 50)) # Grey background
        
        # Draw tiles
        for y in range(self.env.height):
            for x in range(self.env.width):
                rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                if self.env.grid[y][x] == TileColor.BLACK:
                    color = (0, 0, 0)
                else:
                    color = (255, 255, 255)
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, (100, 100, 100), rect, 1) # Grid line

        # Draw players
        p1 = self.env.player1
        if p1.alive:
            center = (int((p1.x + 0.5) * self.cell_size), int((p1.y + 0.5) * self.cell_size))
            pygame.draw.circle(self.screen, (255, 0, 0), center, int(self.cell_size * 0.4)) # Red P1
            # Ammo indicator
            self._draw_ammo(center, p1.ammo)
            
        p2 = self.env.player2
        if p2.alive:
            center = (int((p2.x + 0.5) * self.cell_size), int((p2.y + 0.5) * self.cell_size))
            pygame.draw.circle(self.screen, (0, 0, 255), center, int(self.cell_size * 0.4)) # Blue P2
            self._draw_ammo(center, p2.ammo)

        # Draw bullets
        for b in self.env.bullets:
            bx = int((b.x + 0.5) * self.cell_size)
            by = int((b.y + 0.5) * self.cell_size)
            color = (255, 100, 100) if b.owner == PlayerId.P1 else (100, 100, 255)
            pygame.draw.circle(self.screen, color, (bx, by), 4)

        pygame.display.flip()
        
    def _draw_ammo(self, center, ammo):
        # Simple dots around player
        # (Implementation detail omitted for brevity, standard circle dots)
        pass

def main():
    parser = argparse.ArgumentParser(description="Watch INVERSUS AI play")
    parser.add_argument("model_path", type=str, help="Path to trained model (.pt file)")
    parser.add_argument("--mode", type=str, default="vs_dummy", choices=["vs_user", "ai_vs_ai", "vs_dummy"], help="Game mode")
    parser.add_argument("--fps", type=int, default=30, help="Base FPS")
    parser.add_argument("--speed", type=float, default=1.0, help="Speed multiplier (can be changed in-game with +/-)")
    parser.add_argument("--opponent_difficulty", type=str, default="easy", choices=["easy", "hard"], help="Dummy opponent difficulty")
    
    args = parser.parse_args()
    
    player = GamePlayer(args.model_path, args.mode, args.fps, args.speed, args.opponent_difficulty)
    player.run()

if __name__ == "__main__":
    main()
