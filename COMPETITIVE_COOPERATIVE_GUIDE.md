# Competitive & Cooperative Multi-Agent CTF Simulation

## Quick Start

### Run Team-Based Training (Recommended)

```bash
# Quick 5-episode test
EPISODES=5 ./marl_env/bin/python -m experiments.train_ctf_teams

# Standard training (100 episodes)
EPISODES=100 ./marl_env/bin/python -m experiments.train_ctf_teams

# Extended training (500 episodes for better convergence)
EPISODES=500 ./marl_env/bin/python -m experiments.train_ctf_teams
```

### Watch Agents Play

```bash
# Real-time game visualization
./marl_env/bin/python -m experiments.visualize_ctf

# See agent movement and flag dynamics
# Red team (r) vs Blue team (b)
# Each flag (R/B) must be captured and brought home
```

## What Makes This Competitive & Cooperative?

### Cooperative Aspects

1. **Shared Team Rewards**: Agents on the same team all receive bonuses when team achieves goals
2. **Proximity Bonus**: Agents get small rewards for being near teammates
3. **Defense Coordination**: Multiple defenders work together to protect flag
4. **Distributed Goals**: Attacking agents and defending agents must specialize

### Competitive Aspects

1. **Zero-Sum Objectives**: One team's victory is the other team's loss
2. **Resource Conflict**: Both teams compete for the same flags
3. **Strategic Opposition**: Defense vs Attack creates emergent adversarial behavior
4. **Team Rankings**: Win/loss statistics show which team is better

### Game Dynamics

```
Attacking Strategy:
  - Coordinate to reach opponent base
  - Avoid defenders
  - Work together to carry flag home

Defending Strategy:
  - Patrol near flag/base
  - Tag incoming attackers
  - Protect flag from theft

Emergent Behavior:
  - Some agents naturally become "offense"
  - Others become "defense"
  - Teams develop strategies over training
  - Blue team learns to swarm; Red team learns to spread
```

## Architecture

### Team-Based Training Script

**File**: `experiments/train_ctf_teams.py`

```python
# Training loop
for episode in episodes:
    obs = env.reset()
    
    while not done:
        # 1. Each agent chooses action independently
        actions = [agent.choose_action(state) for agent in agents]
        
        # 2. Environment executes all actions
        next_obs, rewards, done = env.step(actions)
        
        # 3. Cooperative bonus: share team rewards
        for agent in red_team:
            team_reward = average(rewards[all_red_agents])
            agent.update(state, action, individual_reward + 0.5*team_reward, next_state)
        
        # 4. Agents decay exploration
        for agent in agents:
            agent.decay_epsilon()
```

### Environment Rewards

**File**: `envs/ctf_env.py`

```
Individual Rewards (Agent-specific):
  - Carrying opponent flag: +0.1
  - Reaching opponent base: +0.05
  - Defending near own base: +0.02
  
Team Rewards (Shared):
  - Capturing flag: +10.0 (shared equally)
  - Multiple teammates near each other: +0.01 each
  
Combined: agent_reward + 0.5 * team_average
```

## Game Modes Available

### 1. Standard CTF (Default)

Two 3-agent teams, symmetric bases, flag-capture objective.

```bash
EPISODES=100 ./marl_env/bin/python -m experiments.train_ctf_teams
```

### 2. Asymmetric Teams

Red: 2 agents (elite), Blue: 4 agents (militia)

```python
from experiments.game_modes import AsymmetricCTFEnv
env = AsymmetricCTFEnv()
```

### 3. King of the Hill

Control central point, first to 100 points wins

```python
from experiments.game_modes import KingOfTheHillMode
env = KingOfTheHillMode()
```

### 4. Team Deathmatch

Agents score by getting close to opponents

```python
from experiments.game_modes import TeamDeathmatchMode
env = TeamDeathmatchMode()
```

### 5. Relay Race

Teams pass baton through checkpoints

```python
from experiments.game_modes import RelayRaceMode
env = RelayRaceMode()
```

### 6. Tower Defense

Red defends tower, Blue attacks it

```python
from experiments.game_modes import TowerDefenseMode
env = TowerDefenseMode()
```

## Key Features for Cooperation & Competition

### Cooperation Mechanisms

1. **Shared Rewards**: Team members get bonus when team succeeds
2. **Spatial Proximity**: Small reward for being near teammates
3. **Role Specialization**: Agents learn to attack or defend
4. **Communication Implicit**: Via action patterns and positioning

### Competition Mechanisms

1. **Opposing Objectives**: Each team wants what opponent has
2. **Limited Resources**: Only one flag per team
3. **Direct Conflict**: Defenders vs Attackers
4. **Win/Loss**: Clear victor after each game

### Emergent Behaviors

Teams learn to:
- **Cluster for attack**: Multiple agents moving together toward objective
- **Spread for defense**: Agents cover more area
- **Anticipate opponent**: Learn opponent tendencies
- **Adapt strategies**: Adjust tactics based on prior matches

## Analysis & Metrics

### Track Team Performance

```python
import numpy as np

results = np.load('results/ctf_team_rewards.npy')

print(f"Average score per episode: {np.mean(results):.2f}")
print(f"Score trend (last 10 episodes): {np.mean(results[-10:]):.2f}")
print(f"Max score achieved: {np.max(results):.2f}")
```

### Measure Cooperation Quality

```python
# Calculate how often agents win together (vs rarely winning)
win_variance = np.var(results)  # Lower = more consistent teamwork

# Measure coordination: avg distance between teammates
from envs.ctf_env import CTFEnv
env = CTFEnv()

# ... play episode and measure distances
avg_teammate_distance = ...
```

### Competition Strength

```python
# Measure difficulty of wins
# Early episodes: random agents = easier wins
# Later episodes: trained agents = harder wins
win_difficulty_trend = [
    (ep, score)  # Score when capture happened
    for ep, score in enumerate(results)
    if score > 0
]
```

## Customization Examples

### Increase Cooperation Emphasis

```python
# In train_ctf_teams.py:
combined_reward = rewards[agent_id] + 1.0 * team_bonuses[agent_id]  # Increased from 0.5
```

### Make Capture Easier/Harder

```python
# In ctf_env.py, _calculate_rewards():
reward += 20.0  # Easier (reward doubled)
reward += 5.0   # Harder (reward halved)
```

### Asymmetric Rewards (Cooperative vs Competitive)

```python
# Give more reward to defenders (cooperation)
if agent in defense_team:
    reward += extra_defense_bonus

# Give more reward to attackers (competition)
if agent in attack_team:
    reward += extra_attack_bonus
```

### Change Team Size Dynamically

```python
env = CTFEnv(grid_size=20, team_size=4)  # 4 vs 4 instead of 3 vs 3
```

## Performance Expectations

### Learning Curve

| Episodes | Behavior |
|----------|----------|
| 1-20 | Random movements, no captures |
| 21-50 | Some captures begin, agents explore |
| 51-100 | Regular captures, strategy emerges |
| 101-200 | Clear attack/defense roles |
| 200+ | Optimized strategies, good coordination |

### Typical Results

- **No training** (random actions): 0% capture rate
- **20 episodes**: 5-10% capture rate
- **50 episodes**: 20-40% capture rate
- **100 episodes**: 40-60% capture rate
- **500+ episodes**: 70-90% capture rate

## Troubleshooting

**Q: Blue team always wins**

A: Check for symmetry in bases and spawn points:
```python
assert env.red_flag_base == (3, 3)
assert env.blue_flag_base == (11, 11)
# Should be symmetric on 15x15 grid
```

**Q: Agents not cooperating**

A: Increase cooperation weight:
```python
# From 0.5 to 1.0 or higher
combined_reward = rewards[agent_id] + 1.0 * team_bonuses[agent_id]
```

**Q: Training is very slow**

A: Reduce problem complexity:
```bash
EPISODES=50 ./marl_env/bin/python -m experiments.train_ctf_teams
# Or reduce team_size in train script:
# env = CTFEnv(grid_size=10, team_size=2, max_steps=200)
```

**Q: Can't see visualization**

A: Save output to file:
```bash
./marl_env/bin/python -m experiments.visualize_ctf > game.txt
cat game.txt | head -100
```

## Next Steps

1. **Tune cooperation vs competition** - Adjust reward weights
2. **Try different game modes** - Use game_modes.py examples
3. **Add communication** - Let agents send messages
4. **Neural networks** - Replace IQL with DQN for larger state spaces
5. **Self-play** - Train agents against each other iteratively
6. **Curriculum learning** - Start simple, increase difficulty

## Files Reference

- `envs/ctf_env.py` - Core team-based CTF environment
- `experiments/train_ctf_teams.py` - Team-based training script
- `experiments/visualize_ctf.py` - Game visualization
- `experiments/game_modes.py` - Alternative game modes
- `CTF_TEAM_GUIDE.md` - Detailed guide
- `algorithms/iql.py` - Independent Q-Learning agents

---

**Ready to build competitive multi-agent games?** Start with:

```bash
EPISODES=100 ./marl_env/bin/python -m experiments.train_ctf_teams
./marl_env/bin/python -m experiments.visualize_ctf
```
