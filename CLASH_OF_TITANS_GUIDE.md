# Clash of Titans Game Mode - Complete Guide

## Overview

**Clash of Titans** is an advanced multi-agent game mode combining three classic game mechanics:

1. **Tower Defense**: Protect your base tower from enemy destruction
2. **Relay Racing**: Escort powerful units through checkpoints toward the enemy
3. **Team Deathmatch**: Score points through proximity-based tactical positioning

The game features:
- **6 agents** organized into **2 teams** (3 per team)
- **Competitive and Cooperative gameplay** with team reward sharing
- **Multiple victory conditions**: Tower destruction or score-based win
- **Dynamic combat** with movement, positioning, and escort mechanics

---

## Game Setup

### Environment
- **Grid Size**: 20x20 (customizable)
- **Teams**: Red (agents 0-2) vs Blue (agents 3-5)
- **Max Steps**: 500 (configurable)

### Initial Positions
- **Red Team**: Start near red tower at (2, 2)
- **Blue Team**: Start near blue tower at (17, 17)
- **Red Titan**: Spawns at red base, escorts toward blue tower
- **Blue Titan**: Spawns at blue base, escorts toward red tower

### Starting Resources
- **Red Tower Health**: 100
- **Blue Tower Health**: 100
- **Titan Health**: 30 each
- **Initial Scores**: 0-0

---

## Core Mechanics

### 1. Titan Escort System (Relay Mechanic)

Each team has a "Titan" unit - a powerful entity that can destroy towers.

#### How Escort Works
```
Agent is within 3 units of Titan → Titan moves toward next checkpoint
Titan reaches checkpoint → Team gets +30 points
Titan reaches final checkpoint → Tower takes 20 damage, team gets +50 points
```

#### Checkpoint Progression
**Red Team Checkpoints** (escorting Red Titan to Blue tower):
1. (5, 5) - First waypoint
2. (10, 10) - Middle waypoint  
3. (15, 15) - Final waypoint (at Blue tower)

**Blue Team Checkpoints** (escorting Blue Titan to Red tower):
1. (15, 15) - First waypoint
2. (10, 10) - Middle waypoint
3. (5, 5) - Final waypoint (at Red tower)

#### Strategy Implications
- **Offensive Role**: One agent should stay near the Titan to guide it
- **Defensive Role**: Others can intercept enemy Titans or defend the tower
- **Timing**: Faster escort = quicker victory via tower destruction

### 2. Tower Defense System

Towers are the primary objective - first to destroy enemy tower wins.

#### Tower Mechanics
- Each tower has **100 health**
- Takes **20 damage** each time enemy Titan completes relay (reaches final checkpoint)
- **Tower destroyed** at 0 health = instant team victory
- Visible tower status:
  - ⬤ (Red) / ⊕ (Blue) = Full health (100)
  - ● (Red) / ⊙ (Blue) = Half health (50)
  - ○ (Red) / ◯ (Blue) = Critical health (<25)

#### Tower Defense Strategy
- Position agents near tower to prevent Titan escorts
- Use numbers advantage (3v1 or 3v2 can overwhelm Titan + escort)
- Proximity combat rewards help discourage enemy advances

### 3. Proximity Combat System (Deathmatch)

Agents earn rewards for being near opponents (tactical positioning).

#### Combat Rules
- **Attack Radius**: 2 grid units
- **Reward**: +0.5 for each agent-opponent pair in proximity
- **No Health Loss**: Proximity-based (no agent elimination)
- **Strategic Value**: Encourages aggressive positioning and defense

#### Combat Applications
- Surround enemy Titan to prevent escorts
- Block checkpoint routes with agent clusters
- Defend your tower with 3-agent formations

### 4. Reward Structure

All rewards are calculated per agent and include cooperative bonuses.

#### Individual Rewards
- **Titan Checkpoint**: +30 points (to team)
- **Titan Final Checkpoint**: +50 points bonus (to team)
- **Proximity Combat**: +0.5 per step (for being near opponents)
- **Tower Destroyed**: +10 points (for destroying tower)

#### Cooperative Bonus
```
agent_reward = individual_reward + 0.5 * team_average_reward
```

This means:
- If one teammate escorts Titan (+30), all 3 team agents get +15 bonus
- Tower destruction bonus spreads across entire team
- Encourages cooperation without strict role assignment

#### Example Calculation
```
Red Team Scenario:
- Agent 0 escorts Titan to checkpoint: +30
- Agent 1 proxies enemy agent: +0.5
- Agent 2 inactive: 0

Team average = (30 + 0.5 + 0) / 3 = 10.17
Final rewards:
- Agent 0: 30 + 0.5*10.17 = 35.09
- Agent 1: 0.5 + 0.5*10.17 = 5.59
- Agent 2: 0 + 0.5*10.17 = 5.09
```

---

## Victory Conditions

### Primary Victory: Tower Destruction
- **Trigger**: Tower health drops to 0
- **Winner**: Team whose Titan successfully escorts 5 times (100 damage total)
- **Time**: Usually ~100-150 steps (fast-paced)

### Secondary Victory: Score-Based
- **Trigger**: Either team reaches 100 points
- **How**: Via checkpoint escorts, tower damage, combat positioning
- **Time**: Usually ~300-400 steps (if towers survive)

### Draw/Timeout
- **Trigger**: 500 steps elapsed with no victory
- **Result**: Drawn game (can measure as both teams' scores)

---

## Training the Environment

### Quick Start
```bash
# Train for 10 episodes
EPISODES=10 python -m experiments.train_clash

# Results stored in:
# - results/clash_rewards.npy
# - results/clash_tower_damage.npy
```

### Training Configuration
Located in [experiments/train_clash.py](experiments/train_clash.py):

```python
env = ClashOfTitansEnv(grid_size=20, team_size=3, max_steps=500)
agents = [IQLAgent(state_size=20, action_size=4) for _ in range(6)]

# Hyperparameters
learning_rate = 0.01
gamma = 0.99  # Discount factor
epsilon_decay = 0.995  # Exploration decay
```

### State Encoding (20 dimensions)
```
[0:6]   - Red agent positions (3 agents × 2 coords, normalized)
[6:12]  - Blue agent positions (3 agents × 2 coords, normalized)
[12:14] - Red tower (position normalized)
[14:16] - Red tower health + normalized
[16:18] - Blue tower (position normalized)
[18:20] - Blue tower health + normalized
[20:21] - Red Titan position + health (normalized)
[21:22] - Blue Titan position + health (normalized)
[22:23] - Checkpoint progress (red/blue, normalized 0-1)
[24:25] - Team scores (red/blue, normalized 0-1)
```

### Training Dynamics

**Learning Phases**:
1. **Exploration Phase (0-5 episodes)**: Agents explore randomly, learn basic rewards
2. **Role Specialization (5-15 episodes)**: Agents begin to differentiate (escorts vs defenders)
3. **Strategy Formation (15+ episodes)**: Coordinated team tactics emerge

**Expected Learning Curve**:
```
Episodes 1-5:   Scores 10-30 (random behavior)
Episodes 5-15:  Scores 30-60 (learning checkpoints)
Episodes 15+:   Scores 60-100+ (coordinated strategies)
```

---

## Visualization

### Running the Visualization
```bash
python -m experiments.visualize_clash
```

This shows 3 demo games with full ASCII visualization.

### Display Legend
```
Grid Symbols:
  ⬤ / ● / ○ = Red Tower (full/half/critical health)
  ⊕ / ⊙ / ◯ = Blue Tower (full/half/critical health)
  T / t = Red Titan (healthy/damaged)
  P / p = Blue Titan (healthy/damaged)
  r / R = Red agents (single/group)
  b / B = Blue agents (single/group)
  X = Combat zone (agents from both teams nearby)

Information Display:
  Red/Blue Tower Health (0-100)
  Team Scores (0-100+)
  Titan Checkpoint Progress (0-3)
```

### Visualization Features
- **Real-time updates**: Every 10 steps
- **Final state**: Shows victory condition and final scores
- **Post-game stats**: Win ratios, average scores, game length

---

## Game Strategies

### Offensive Strategy
**Goal**: Escort your Titan to destroy enemy tower

1. **Agent Roles**:
   - 1 Escort Agent: Stays within 3 units of Titan
   - 2 Offensive Agents: Advance with Titan, harass defenders

2. **Progression Path**:
   ```
   Start at (2,2) → CP1(5,5) → CP2(10,10) → CP3(15,15) → Enemy Tower
   ```

3. **Timing**:
   - Quick escorts (5 relays) = tower destroyed in ~100 steps
   - Slow escorts = defenders have time to organize

### Defensive Strategy
**Goal**: Prevent enemy Titan from reaching checkpoints

1. **Agent Roles**:
   - 2 Defense Agents: Block checkpoint routes
   - 1 Intercept Agent: Engage enemy Escort

2. **Defensive Positions**:
   ```
   Agents spread along enemy checkpoint routes
   Form clusters (R or B symbols) to block progress
   ```

3. **Counter-Offensive**:
   - After blocking, agents can switch to offensive mode
   - Use momentum to attack weakened enemy defense

### Balanced Strategy
**Goal**: Simultaneously attack and defend

1. **Split Forces** (2 Offensive, 1 Defensive):
   - 2 agents escort your Titan
   - 1 agent defends tower from enemy Titan

2. **Rotate Roles**:
   - After completing a relay, rotate defenders
   - Keeps all agents engaged and learning

3. **Conditional Tactics**:
   - If winning: Aggressive offense, weak defense
   - If losing: Strong defense, tactical offense

---

## Advanced Customization

### Adjusting Difficulty

#### Easier (Faster Games)
```python
env = ClashOfTitansEnv(
    grid_size=15,      # Smaller map
    team_size=2,       # Fewer agents (4 total)
    max_steps=300      # Shorter games
)
```

#### Harder (Slower Games)
```python
env = ClashOfTitansEnv(
    grid_size=25,      # Larger map
    team_size=4,       # More agents (8 total)
    max_steps=1000     # Extended games
)
```

### Adjusting Tower Health
In [envs/clash_of_titans.py](envs/clash_of_titans.py):
```python
self.red_tower_health = 150  # Tougher towers
self.tower_max_health = 150
```

### Adjusting Checkpoint Rewards
```python
# In step() method, find:
# "Checkpoint completed" section
# Change: reward += 30  (from 30 to any value)
# Change: reward += 50  (final checkpoint bonus)
```

### Adjusting Combat Radius
```python
self.attack_radius = 2.5  # Increase from 1.5 for longer-range deathmatch
```

---

## Performance Metrics

### Tracking Learning Progress

After training, analyze with:
```python
import numpy as np
import matplotlib.pyplot as plt

rewards = np.load('results/clash_rewards.npy')
tower_damage = np.load('results/clash_tower_damage.npy')

plt.plot(rewards)
plt.title('Score Progression')
plt.ylabel('Total Episode Score')
plt.xlabel('Episode')
plt.show()
```

### Key Metrics
- **Win Rate**: % of episodes where team destroys tower
- **Average Score**: Sum of team's total score across episodes
- **Tower Damage**: How much damage inflicted per episode (should increase)
- **Learning Curve**: Score should increase over episodes

### Expected Performance by Episode Range

| Episodes | Red Score | Blue Score | Tower Damage | Notes |
|----------|-----------|------------|--------------|-------|
| 1-5 | 10-20 | 10-20 | 0-10 | Random exploration |
| 5-15 | 20-40 | 20-40 | 10-30 | Learning begins |
| 15-30 | 40-70 | 40-70 | 30-50 | Strategy forming |
| 30+ | 70-100+ | 70-100+ | 50+ | Mature strategies |

---

## Troubleshooting

### Issue: Titans not moving
**Check**: Agents within 3 units of Titan?
- Ensure agents are positioned near Titan spawn area
- Check position values in observation

### Issue: No checkpoints being reached
**Solution**: Agents need consistent escorts
- Verify escort position updates in `_update_titans()`
- May indicate insufficient learning (run more episodes)

### Issue: Games taking too long
**Solution**: Lower `max_steps` or increase map size penalties
- Make tower health lower (80 instead of 100)
- Increase checkpoint rewards (40 instead of 30)

### Issue: Rewards not increasing
**Solution**: Check reward structure and state encoding
- Verify cooperative bonus calculation (0.5 * team_avg)
- Ensure agents are exploring (epsilon-greedy in action selection)

---

## File Structure

```
Clash of Titans Implementation:
├── envs/
│   └── clash_of_titans.py          # Main environment
├── experiments/
│   ├── train_clash.py              # Training script
│   └── visualize_clash.py           # Visualization tool
├── algorithms/
│   └── iql.py                      # IQL agents (for training)
└── results/
    ├── clash_rewards.npy           # Training results
    └── clash_tower_damage.npy       # Tower destruction metrics
```

---

## Next Steps

### Experiment Ideas
1. **Asymmetric Teams**: 4v2 (more attackers vs more defenders)
2. **Dynamic Towers**: Towers repair over time
3. **Multiple Titans**: Teams have 2-3 Titans to escort
4. **Power-Ups**: Spawn bonuses that boost agent capabilities
5. **Environmental Hazards**: Obstacles that block direct routes

### Advanced Features
- **Role Assignment**: Agents learn distinct attacker/defender roles
- **Communication**: Agents share position info (not in current version)
- **Fog of War**: Limited vision for each agent
- **Persistent Damage**: Agent health tracking across episodes

### Alternative Learning Algorithms
- **MADDPG**: Actor-Critic networks for continuous adaptation
- **PPO**: Policy Gradient for faster convergence
- **Q-learning Variants**: Dueling DQN for better value estimation

---

## References

### Game Design Inspiration
- **Clash Royale** (Supercell): Tower destruction mechanic
- **Tower Defense Games**: Strategic positioning and resource management
- **Deathmatch Games**: Proximity-based tactical engagement

### Multi-Agent Learning Concepts
- **IQL (Independent Q-Learning)**: Decentralized agent training
- **Cooperative Rewards**: Team bonuses encouraging coordination
- **Emergent Behavior**: Complex strategies from simple reward signals

---

## Summary

Clash of Titans combines classic game mechanics (Tower Defense + Relay + Deathmatch) into a challenging multi-agent learning environment. The game emphasizes:

- **Teamwork**: Cooperative reward structure
- **Strategy**: Multiple paths to victory (speed vs durability)
- **Tactics**: Real-time positioning and engagement
- **Learning**: Progressive skill development through RL

Start with basic training to understand mechanics, then experiment with different strategies, team compositions, and reward structures to discover emergent behaviors!
