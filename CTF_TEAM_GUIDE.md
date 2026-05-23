# Team-Based Competitive & Cooperative CTF Game Guide

## Overview

This enhanced CTF (Capture-The-Flag) environment implements realistic team-based game mechanics with:

- **Two competing teams** (Red vs Blue)
- **Cooperative mechanics** within teams
- **Competitive mechanics** between teams
- **Realistic game dynamics** (flag capture, defense, teamwork)

## Game Mechanics

### Teams & Spawning

- **Red Team**: Agents 0, 1, 2
  - Base location: (3, 3) on a 15×15 grid
  - Objective: Capture Blue flag and bring to Red base

- **Blue Team**: Agents 3, 4, 5
  - Base location: (11, 11)
  - Objective: Capture Red flag and bring to Blue base

Agents spawn near their team's base with random offset.

### Core Gameplay

1. **Flag Pickup**
   - Agents can pick up opponent's flag when near opponent's base (< 1.5 distance)
   - Carrying agent is shown in game state

2. **Flag Capture (Win Condition)**
   - Carrying agent brings flag back to own base → team wins
   - Big reward (10.0) for successful capture
   - Episode ends immediately when flag is captured

3. **Defense & Tagging**
   - Defenders near own base can tag flag carriers
   - Tagged carriers drop the flag
   - Defenders gain small reward for being near base

4. **Episode Length**
   - Max 500 steps
   - Game ends when a team captures opponent's flag OR timeout

### Reward Structure

Each agent receives **individual + cooperative rewards**:

| Action | Individual Reward | Team Cooperation |
|--------|-------------------|------------------|
| Capture flag | +10.0 | Shared with teammates |
| Carry opponent flag | +0.1 | Shared with teammates |
| Near opponent flag | +0.05 | Shared with teammates |
| Near teammate | +0.01 | Shared |
| Defend own base | +0.02 | Shared |

**Combined reward** = Individual reward + 0.5 × Team average reward

This incentivizes both individual performance and teamwork.

## Running Team-Based Simulations

### 1. Train Team-Based IQL Agents

```bash
# Quick test (3 episodes)
EPISODES=3 ./marl_env/bin/python -m experiments.train_ctf_teams

# Full training (100 episodes)
EPISODES=100 ./marl_env/bin/python -m experiments.train_ctf_teams

# Very long training (500+ episodes for convergence)
EPISODES=500 ./marl_env/bin/python -m experiments.train_ctf_teams
```

**Output**: Progress bar showing win rates and average scores

Expected progress:
- Episode 10: ~10-20% win rate
- Episode 50: ~30-50% win rate  
- Episode 100: ~50-70% win rate

### 2. Visualize Games in Progress

Watch agents play in real-time:

```bash
./marl_env/bin/python -m experiments.visualize_ctf
```

**Visualization symbols**:
- `R` / `B` = Red/Blue flags at bases
- `r` / `b` = Red/Blue team agents
- `1` / `2` = Stacked agents (multiple at same location)
- `.` = Empty space

Shows:
- Agent positions and movement
- Flag locations and carriers
- Current score and step count
- Team win/loss outcomes

### 3. Analyze Training Results

Plot reward curves:

```bash
./marl_env/bin/python -c "
import numpy as np
import matplotlib.pyplot as plt

rewards = np.load('results/ctf_team_rewards.npy')
plt.figure(figsize=(12, 5))
plt.plot(rewards, alpha=0.6)
plt.rolling_avg = np.convolve(rewards, np.ones(10)/10, mode='valid')
plt.plot(np.arange(9, len(rewards)), plt.rolling_avg, 'r-', linewidth=2, label='10-ep moving avg')
plt.xlabel('Episode')
plt.ylabel('Score (flags captured)')
plt.title('CTF Team Training Progress')
plt.legend()
plt.grid(True)
plt.savefig('results/ctf_training_plot.png', dpi=150, bbox_inches='tight')
plt.show()
"
```

## Advanced: Customization

### Modify Grid Size

```python
env = CTFEnv(grid_size=20, team_size=3, max_steps=500)
```

### Change Base Positions

Edit [envs/ctf_env.py](envs/ctf_env.py):

```python
self.red_flag_base = (grid_size // 4, grid_size // 4)    # Change these
self.blue_flag_base = (3 * grid_size // 4, 3 * grid_size // 4)
```

### Adjust Rewards

Edit `_calculate_rewards()` in [envs/ctf_env.py](envs/ctf_env.py):

```python
reward += 10.0   # Capture bonus (increase to 20.0 for harder target)
reward += 0.1    # Carry bonus
reward += 0.05   # Exploration bonus
```

### Modify Cooperation Weight

In [experiments/train_ctf_teams.py](experiments/train_ctf_teams.py):

```python
combined_reward = rewards[agent_id] + 0.5 * team_bonuses[agent_id]
                                      ^
                                      |
                   Change this 0.5 to emphasize/de-emphasize cooperation
                   0.0 = purely individual
                   1.0 = 50-50 individual-team
                   2.0+ = heavy cooperation bonus
```

## Game Flow Example

```
Episode 1:
  Step 0: All agents spawn near bases
  Step 15: Red agent gets close to Blue flag
  Step 30: Red agent picks up Blue flag
  Step 60: Red agent reaches Red base with flag
  → Red team wins! Score: 1-0
  Rewards: Carrying agent +10, teammates +5 (shared), defenders +small bonus
  
Episode 2:
  ... (blue team tries to capture red flag)
  ... (agents learn to cooperate and improve tactics)
```

## Key Differences from Original Environment

| Feature | Original | Enhanced |
|---------|----------|----------|
| Teams | Single flag, all agents compete | Two teams, cooperative & competitive |
| Bases | Static flag position | Two base locations with defensive mechanics |
| Rewards | 0 or 1 for capture | Detailed reward shaping for intermediate progress |
| Gameplay | Reach a position | Coordinate to capture and defend |
| Strategy | Race to flag | Team strategy (some attack, some defend) |
| Learning | Individual goal | Emergent teamwork |

## Performance Metrics to Track

```python
# In your analysis script
red_wins = number of episodes where red_score > 0
blue_wins = number of episodes where blue_score > 0
win_rate = (red_wins + blue_wins) / total_episodes
avg_steps_to_capture = average steps when flag is captured
coordination_quality = variance in teammate distances
```

## Next Steps for Enhancement

1. **Communication**: Add message passing between teammates
2. **Multi-task learning**: Add secondary objectives (collect resources)
3. **Curriculum learning**: Start with smaller grids, increase difficulty
4. **Role assignment**: Automatic offensive/defensive role specialization
5. **Neural networks**: Replace tabular IQL with DQN for larger state spaces

## Troubleshooting

**Q: Agents not learning to cooperate**
- Increase `0.5` coefficient in combined_reward calculation
- Increase team_size or grid_size for more interesting strategies
- Run more episodes (aim for 500+)

**Q: Blue team always wins**
- Check base positions are symmetric
- Verify spawn ranges are equal for both teams
- Check reward structure isn't biased

**Q: Training is very slow**
- Reduce grid_size to 10-12
- Reduce team_size to 2
- Reduce max_steps to 200

**Q: Can't see visualization output**
- Run `./marl_env/bin/python -m experiments.visualize_ctf > game_log.txt`
- Check game_log.txt for output
