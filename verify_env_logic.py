
import numpy as np
from inversus.core import InversusEnv, PlayerId, Direction, Action, ActionType
from inversus_rl.env_wrappers import build_observation, discrete_to_action

def verify_coordinates():
    print("=== Verifying Coordinates & Observations ===")
    env = InversusEnv(width=10, height=10)
    env.reset()
    
    # Force positions
    # Agent at (5, 5)
    # Enemy at (5, 2) (Above agent)
    env.player1.x, env.player1.y = 5, 5
    env.player2.x, env.player2.y = 5, 2
    
    # Get observation
    obs, extra = build_observation(env, PlayerId.P1)
    
    print(f"P1 Pos: ({env.player1.x}, {env.player1.y})")
    print(f"P2 Pos: ({env.player2.x}, {env.player2.y})")
    
    # Check Tensor Shape (C, H, W) -> (12, 10, 10)
    print(f"Observation Shape: {obs.shape}")
    
    # Check P1 Channel (Channel 2)
    # Expect 1.0 at obs[2, 5, 5]
    p1_val = obs[2, 5, 5]
    print(f"P1 Channel (2) at [5,5]: {p1_val} (Expected 1.0)")
    
    # Check P2 Channel (Channel 3)
    # Expect 1.0 at obs[3, 2, 5] -> Note: grid is usually [y][x], so [2][5]
    p2_val = obs[3, 2, 5]
    print(f"P2 Channel (3) at [2,5] (y=2,x=5): {p2_val} (Expected 1.0)")
    
    if p1_val != 1.0 or p2_val != 1.0:
        print("FAIL: Coordinate mismatch in observation!")
    else:
        print("PASS: Coordinates match.")

def verify_actions():
    print("\n=== Verifying Actions & Ballistics ===")
    env = InversusEnv(width=10, height=10)
    env.reset()
    
    # P1 at (5, 5)
    env.player1.x, env.player1.y = 5, 5
    # Remove P2 to avoid interference
    env.player2.alive = False
    
    # Test Shooting UP
    # Action 5 = SHOOT UP
    print("Executing Action 5 (SHOOT UP)...")
    action = discrete_to_action(5)
    print(f"Mapped to: {action}")
    
    # Step game
    env.step_players(action, Action(ActionType.NONE, None))
    
    # Check Bullet
    bullets = env.get_bullets()
    print(f"Bullet Count: {len(bullets)}")
    if len(bullets) > 0:
        b = bullets[0]
        print(f"Bullet Pos: ({b.x}, {b.y}), Dir: {b.dir}")
        # Expect bullet at (5, 4) (Up = y-1)
        if b.x == 5 and b.y == 4 and b.dir == Direction.UP:
            print("PASS: Bullet spawned correctly UP.")
        else:
            print(f"FAIL: Bullet wrong. Expected (5,4) UP. Got ({b.x},{b.y}) {b.dir}")
    else:
        print("FAIL: No bullet spawned.")

if __name__ == "__main__":
    verify_coordinates()
    verify_actions()
