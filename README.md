# INVERSUS AI 🤖

Welcome to the **Inversus Reinforcement Learning** project! This codebase simulates a game inspired by *INVERSUS* and trains an AI agent to play it using PPO (Proximal Policy Optimization).

## 🚀 Quick Start

### 1. Train the Agent
Start by training the agent against a "Dummy" opponent (a simple scripted bot). This bootstraps the learning process.

```bash
# Train for 100k steps (approx 10-20 min)
python -m inversus_rl.training --mode vs_dummy --total_steps 100000 --num_envs 4
```
*   `--num_envs 4`: Runs 4 games in parallel to speed up training.
*   Logs will be saved to `runs/inversus_vs_dummy`.

### 2. Visualize Progress
Check if the agent is learning (look for "Win Rate" increasing).

```bash
python inversus_rl/visualize_training.py --log_dir runs/inversus_vs_dummy
```
*   This generates plots (Win Rate, Reward, Loss) in the log directory.

### 3. Watch the AI Play
Once trained, you can watch the AI play or play against it!

```bash
# Watch AI vs Dummy
python -m inversus_rl.play runs/inversus_vs_dummy/policy_final.pt --mode vs_dummy

# Play YOURSELF vs AI (You are P2, Arrows to Move, IJKL to Shoot)
python -m inversus_rl.play runs/inversus_vs_dummy/policy_final.pt --mode vs_user

# Watch AI vs AI (Self-Play)
python -m inversus_rl.play runs/inversus_vs_dummy/policy_final.pt --mode ai_vs_ai
```

## 🎮 Play Controls

When running in `vs_user` mode:
*   **Move**: Arrow Keys (↑ ↓ ← →)
*   **Shoot**: I J K L keys (Up, Left, Down, Right)
*   **Charge Shot**: Shift + I J K L (Flipping tiles)
*   **Speed Up**: `+` or `=` key
*   **Slow Down**: `-` key
*   **Pause**: Spacebar
*   **Reset**: R key

---

## 📊 Understanding Results

If you see logs like:
`Step 71680... | Avg Reward: -10.854 | Win Rate: 0.040`

*   **Win Rate 0.040 (4%)**: This is **NORMAL** and good! It means the agent is starting to win occasionally (4% is better than 0%). It takes millions of steps to reach 50%+.
*   **Avg Reward -10.8**: This is also normal. Most games end in a loss (-10 penalty) plus time penalties. As win rate climbs, this will increase.
*   **Action**: Keep training! 70k steps is just the beginning.

## 🧠 Advanced Training: Self-Play

Once the agent beats the dummy consistently (>50% win rate), switch to **Self-Play**. This is where the agent plays against itself to become superhuman.

```bash
# Train using an existing model as a starting point (optional but recommended)
# (Adjust source code to load model if needed, or start fresh)

python -m inversus_rl.training --mode selfplay --total_steps 1000000 --num_envs 8
```
*   In self-play, the opponent is a past version of the agent.
*   The opponent updates every 10,000 steps.
*   Requires more steps (1M+) to see high-level strategies emerge.

## 🛠️ Key Files & Fixes

We have implemented several critical fixes to make learning possible:

1.  **Enhanced Vision (12 Channels)**: The agent now sees bullet *velocity* (Input channels: 0-1 Tiles, 2-3 Players, 4-7 P1 Bullets, 8-11 P2 Bullets). It is no longer blind to threats!
2.  **Nerfed Dummy**: The scripted dummy now has a reaction delay and makes occasional mistakes, allowing the agent to win initially and start learning.
3.  **Corrected Self-Play**: The self-play updates are now frequent enough (10k steps) to provide a steady curriculum of difficulty.
4.  **Parallel Training**: Use `--num_envs` to train much faster.

## 📁 Project Structure

*   `inversus/`: Core game logic (engine).
*   `inversus_rl/`:
    *   `training.py`: Main training script.
    *   `play.py`: The "Ultimate" player script.
    *   `env_wrappers.py`: RL environment interface (observations, rewards).
    *   `policies.py`: Neural network architecture (CNN).
    *   `ppo_agent.py`: PPO algorithm implementation.

## 📜 Development Journey (The "Reward Shaping" Saga)

This project is a case study in how Reinforcement Learning agents find loopholes. We went from **0% Win Rate** to **87% Win Rate** by identifying and fixing 5 specific "Lazy Strategies":

### 1. The "Blind" Agent (0% WR)
*   **Bug**: The reward for moving closer (+0.00008) was smaller than the time penalty (-0.01).
*   **Result**: The agent determined that "Moving hurts," so it wandered randomly.
*   **Fix**: Boosted Proximity Reward by 500x.

### 2. The "Hugging" Agent (45% WR)
*   **Bug**: We gave +1.0 per step for being close to the enemy.
*   **Result**: The agent realized that standing next to the enemy for 500 steps (+250 pts) was better than killing them (+100 pts). It became a pacifist.
*   **Fix**: Capped Proximity Reward to <50 pts total.

### 3. The "Trapped" Agent (9% WR)
*   **Bug**: The agent spawns in a wall cage. Shooting walls gave 0 points.
*   **Result**: It stayed in spawn because it didn't know how to break out.
*   **Fix**: Added **Territory Reward (+0.5/tile)** to incentivize "Digging".

### 4. The "Farmer" Agent (35% WR)
*   **Bug**: 'Trigger Discipline' reward gave +1.0 for aiming at the enemy, even with 0 ammo.
*   **Result**: The agent stood still, aimed, and dry-fired 100 times to farm points.
*   **Fix**: Added `if ammo > 0` check.

### 5. The "Coward" Agent (24% WR)
*   **Bug**: Death Penalty was -5.0. Waiting 500 steps cost -2.5.
*   **Result**: It learned that "Fighting and Dying" was mathematically worse than "Hiding".
*   **Fix**: Reduced Death Penalty to **-0.5**. Made Aggression mathematically safer than hiding.

### ✅ The Final "Jackpot" Model (87% WR - Easy Mode)
We fixed these by creating a **High Contrast Economy**:
*   **Passive Rewards**: ~10 pts (Breadcrumbs).
*   **Win Reward**: **+500 pts** (Jackpot).
*   **Result**: The agent ignores distractions and hunts for the win.

---

## 🦅 Phase 4: Hard Mode & Curriculum Transfer

After conquering the "Sitting Duck", we faced the **Hard Dummy** (Moves & Shoots).
*   **Problem**: Training from scratch failed (30% WR). The agent couldn't learn to dodge *and* aim simultaneously.
*   **Solution**: **Curriculum Transfer**. We loaded the "Graduate" model from Easy Mode and forced it to fight the Hard Dummy.
*   **Adjustment**: We scaled rewards down (+500 -> +10) to prevent numerical instability, and adjusted penalties to encourage aggression instead of "cowardice".
*   **Result**: **48% Win Rate** against a perfect Aimbot script. This is the skill ceiling for scripted opponents.

### New Commands

**Train against Hard Dummy (Curriculum Transfer):**
```bash
python -m inversus_rl.training --mode vs_dummy --total_steps 500000 --num_envs 4 --opponent_difficulty hard --load_model runs/inversus_vs_dummy_base_v2/policy_final.pt
```

**Watch Agent vs Hard Dummy:**
```bash
# Slow motion (0.5x speed) to see dodging behavior
python -m inversus_rl.play runs/inversus_hard_aggressive/policy_final.pt --mode vs_dummy --opponent_difficulty hard --speed 0.5
```

### ⚔️ Phase 5: Self-Play (The Final Frontier)

The ultimate goal is to train the agent against **itself**. This allows it to discover strategies beyond what a scripted bot can perform (e.g., blocking, trapping).

**Run Self-Play Training:**
```bash
python -m inversus_rl.training --mode selfplay --total_steps 1000000 --num_envs 4 --log_dir runs/inversus_selfplay_v1 --load_model runs/inversus_hard_aggressive/policy_final.pt
```
*   **Time**: This will take a long time (hours/days).
*   **Logic**: The opponent is initially a clone of the agent. Every 20,000 steps, the opponent is updated to the latest version of the agent.
*   **Goal**: Watch for the win rate to fluctuate around 50% (Equilibrium) as both sides get stronger.
