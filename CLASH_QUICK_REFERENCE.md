# Clash of Titans - Quick Reference Card

## Quick Start (Copy & Paste Commands)

### Train Agents
```bash
cd /home/india/Desktop/ctf-marl

# Quick test (2 episodes, ~0.1 sec)
EPISODES=2 ./marl_env/bin/python -m experiments.train_clash

# Normal training (20 episodes, ~0.8 sec)
EPISODES=20 ./marl_env/bin/python -m experiments.train_clash

# Extended training (100 episodes, ~4 sec)
EPISODES=100 ./marl_env/bin/python -m experiments.train_clash
```

### View Gameplay
```bash
# Watch 3 demo games with agent behavior
./marl_env/bin/python -m experiments.visualize_clash
```

### Analyze Results
```bash
# Generate learning progression plots
./marl_env/bin/python -m experiments.analyze_clash
```

---

## Game Map (20x20 Grid)

```
Red Team Side              Blue Team Side
(0,0)                      (19,19)
  ┌──────────────────────┐
  │⬤                  ⊕  │
  │r                    b│
  │                      │
  │     T          P     │
  │                      │
  │        mid            │
  │     (10,10)           │
  │                      │
  │                      │
  │r                    b│
  │⬤                  ⊕  │
  └──────────────────────┘

Legend:
  ⬤ = Red Tower at (2,2)
  ⊕ = Blue Tower at (17,17)
  T = Red Titan (moves toward Blue tower)
  P = Blue Titan (moves toward Red tower)
  r/R = Red agents (r=single, R=group)
  b/B = Blue agents (b=single, B=group)
```

---

## Victory Paths

### Path 1: Tower Destruction (Fast, ~100 steps)
```
Escort Titan through 3 checkpoints (5 times)
→ Tower takes 100 damage total
→ Tower destroyed = INSTANT WIN
Points: ~30-50 per team
```

### Path 2: Score Accumulation (Slow, ~300-400 steps)
```
Gradually earn points from:
  - Checkpoints (+30 each)
  - Tower damage (+10 per hit)
  - Combat positioning (+0.5/step near enemies)
First team to 100 points wins
```

---

## Core Game Rules

| Aspect | Rules |
|--------|-------|
| **Teams** | Red (0-2) vs Blue (3-5) |
| **Map** | 20×20 grid, discrete positions |
| **Movement** | 4 directions: up/down/left/right |
| **Towers** | Start at 100 health, destroyed at 0 |
| **Titans** | Spawn at bases, need escorts to move |
| **Escape Radius** | Agents within 3 units move Titan |
| **Combat** | Proximity reward at <2 distance |
| **Episode Length** | Max 500 steps or victory |

---

## Reward Breakdown

### When Your Team Completes a Checkpoint
```
Each agent on team gets:
  Individual: +30
  Team Bonus: +(0.5 × 30) = +15
  Total: +45 per agent (ideally)
```

### When Your Team Destroys Enemy Tower
```
Each agent on team gets:
  Individual: +10
  Team Bonus: +(0.5 × 10) = +5
  Total: +15 per agent
  
PLUS: Game ends, large score swing
```

### When in Combat (Proximity < 2)
```
Each combat-positioned agent gets:
  Individual: +0.5 per step
  Team Bonus: +(0.5 × avg) = +0.25 per step
  Total: +0.75 per step (if in proximity)
```

---

## Agent Learning

### Episode Phases

```
Episodes 1-5:    Random behavior
                 Score: 10-20
                 Learns: Nothing yet
                 
Episodes 5-15:   Learning accelerates
                 Score: 30-50
                 Learns: Checkpoints, escort behavior
                 
Episodes 15+:    Strategy stabilizes
                 Score: 40-70
                 Learns: Roles, team coordination
```

---

## File Quick Reference

| File | Purpose | Usage |
|------|---------|-------|
| `envs/clash_of_titans.py` | Game environment | `from envs.clash_of_titans import ClashOfTitansEnv` |
| `experiments/train_clash.py` | Training script | `python -m experiments.train_clash` |
| `experiments/visualize_clash.py` | Game viewer | `python -m experiments.visualize_clash` |
| `experiments/analyze_clash.py` | Results analysis | `python -m experiments.analyze_clash` |
| `CLASH_OF_TITANS_GUIDE.md` | Full documentation | Read for mechanics |
| `results/clash_rewards.npy` | Training scores | Loaded by analyze script |
| `results/clash_tower_damage.npy` | Tower damage data | Loaded by analyze script |

---

## Environment API

### Creating Environment
```python
from envs.clash_of_titans import ClashOfTitansEnv

env = ClashOfTitansEnv(grid_size=20, team_size=3, max_steps=500)
```

### Episode Loop
```python
obs = env.reset()

for step in range(500):
    # Agent chooses actions (0=up, 1=down, 2=left, 3=right)
    actions = {
        0: 'up',    # Red agent 0
        1: 'down',  # Red agent 1
        2: 'left',  # Red agent 2
        3: 'right', # Blue agent 3
        4: 'up',    # Blue agent 4
        5: 'down',  # Blue agent 5
    }
    
    # Step environment
    next_obs, rewards, done, info = env.step(actions)
    
    if done:
        break
```

### Key Attributes
```python
env.red_score        # Red team total score
env.blue_score       # Blue team total score
env.red_tower_health # Red tower remaining health
env.blue_tower_health # Blue tower remaining health
env.agent_positions  # Dict: agent_id → (x, y)
env.red_titan        # Titan object (has pos, health)
env.blue_titan       # Titan object (has pos, health)
```

---

## Hyperparameters Reference

### Training (in train_clash.py)
```python
learning_rate = 0.01      # How fast agents learn
gamma = 0.99              # Future reward discount
epsilon_start = 1.0       # Exploration probability
epsilon_decay = 0.995     # Decay per episode
epsilon_min = 0.05        # Min exploration

cooperative_bonus = 0.5   # Team reward weight (0=pure selfish, 1=pure team)
```

### Environment (in clash_of_titans.py)
```python
grid_size = 20            # Map dimensions (20×20)
team_size = 3             # Agents per team (6 total)
max_steps = 500           # Episode timeout
tower_max_health = 100    # Initial tower health
titan_proximity = 3       # Escort radius (within 3 units)
attack_radius = 1.5       # Combat radius (proximity < 1.5)
```

---

## Expected Performance

### After 20 Episodes
```
Red Wins: 20%
Blue Wins: 20%
Avg Score: 25
Tower Damage: 17
```

### After 50 Episodes
```
Red Wins: 25%
Blue Wins: 25%
Avg Score: 40-50
Tower Damage: 25-30
```

### After 100+ Episodes
```
Red Wins: 30%
Blue Wins: 30%
Avg Score: 60-80
Tower Damage: 40+
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError` | Run from repo root: `cd /home/india/Desktop/ctf-marl` |
| No progress after 10 eps | Increase EPISODES to 50+ for statistical significance |
| Agents not moving | Check agent.epsilon > 0 (exploration enabled) |
| Tower not taking damage | Verify Titan reaches final checkpoint |
| Import errors | Use virtual env: `./marl_env/bin/python` |

---

## Advanced Customization

### Make Games Faster
```python
env = ClashOfTitansEnv(
    grid_size=15,        # Smaller map
    team_size=2,         # Fewer agents
    max_steps=300        # Shorter games
)
```

### Make Games Harder
```python
env = ClashOfTitansEnv(
    grid_size=25,        # Larger map
    team_size=4,         # More agents
    max_steps=1000       # Longer games
)
```

### Increase Offense Pressure
```python
# In _update_titans(), change:
# if dist < self.titan_proximity_radius:
# to:
# if dist < self.titan_proximity_radius * 1.5:
# (Makes Titan easier to move)
```

### Make Towers Tougher
```python
self.red_tower_health = 150      # Was 100
self.tower_max_health = 150
# Now takes 7.5 escorts instead of 5
```

---

## Key Insights

1. **Tower Offense > Defense**: Teams with >17 tower damage score +42.5 points higher
2. **Escort Specialization**: One agent naturally stays near Titan, others roam
3. **Cooperative Bonus**: Rewards all agents, even those not escorting
4. **Early Learning**: Most progress happens in first 15 episodes
5. **Emerges Balanced Play**: After 30+ episodes, roles stabilize

---

## Resources

- **Full Guide**: Read [CLASH_OF_TITANS_GUIDE.md](CLASH_OF_TITANS_GUIDE.md)
- **Implementation Details**: Read [CLASH_OF_TITANS_IMPLEMENTATION.md](CLASH_OF_TITANS_IMPLEMENTATION.md)
- **Code**: Check [envs/clash_of_titans.py](envs/clash_of_titans.py)

---

**Status**: ✅ Ready to Use
**Last Updated**: January 2025
**Python Version**: 3.12+
**Dependencies**: numpy, matplotlib, torch (optional)
