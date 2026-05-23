# How to Create Game Simulations with Competitive & Cooperative Agents

## TL;DR - Quick Start

```bash
# Train agents with teamwork
EPISODES=100 ./marl_env/bin/python -m experiments.train_ctf_teams

# Watch agents play
./marl_env/bin/python -m experiments.visualize_ctf

# Compare strategies
./marl_env/bin/python -m experiments.analyze_strategies
```

---

## The Complete Architecture

### 1. **Game Environment** (`envs/ctf_env.py`)

The core game logic that defines:
- **Teams**: Red (agents 0-2) vs Blue (agents 3-5)
- **Bases**: Each team has a home base they defend
- **Objective**: Capture opponent's flag and bring it home
- **Mechanics**: Flag pickup, dropoff, defense mechanics
- **Rewards**: Individual + shared team rewards

Key game logic:
```python
class CTFEnv:
    # Flag management
    - flag_carriers tracking who carries which flag
    - _get_red_flag_pos() returns flag position
    - _handle_flag_pickups() manages pickup mechanics
    
    # Defense mechanics
    - Defenders near own base can tag flag carriers
    - Tagged carriers drop the flag
    
    # Reward shaping
    - +10 for capturing flag (big win)
    - +0.1 for carrying flag (progress)
    - +0.05 for being near opponent flag
    - +0.02 for defending base
    - +0.01 for being near teammate
```

### 2. **Individual Agents** (`algorithms/iql.py`)

Independent Q-Learning agents that:
- Learn individual policies via Q-tables
- Make decisions independently
- Can be trained with shared or individual rewards

```python
class IQLAgent:
    def choose_action(state):
        # Exploration-exploitation tradeoff
        # Pick best action with probability (1-epsilon)
        # Pick random action with probability epsilon
        
    def update(state, action, reward, next_state, done):
        # Q-learning update
        # Q(s,a) += lr * (reward + gamma*max(Q(s',a')) - Q(s,a))
```

### 3. **Training Loop** (`experiments/train_ctf_teams.py`)

Orchestrates team-based learning:

```python
for episode in episodes:
    # Reset environment
    obs = env.reset()
    
    while not done:
        # 1. INDIVIDUAL DECISION MAKING
        #    Each agent chooses action independently
        for agent in agents:
            action = agent.choose_action(state)
        
        # 2. ENVIRONMENT SIMULATION
        #    All actions executed simultaneously
        rewards = env.step(all_actions)
        
        # 3. COOPERATIVE LEARNING
        #    Share team rewards
        for agent in red_team:
            team_avg = average(rewards[red_team])
            combined = individual_reward + 0.5 * team_avg
            agent.update(state, action, combined, next_state, done)
        
        # 4. LEARNING RATE DECAY
        #    Reduce exploration over time
        for agent in agents:
            agent.decay_epsilon()
```

### 4. **Visualization** (`experiments/visualize_ctf.py`)

ASCII art game display showing:
- Agent positions (r/b for red/blue)
- Flag positions (R/B at bases or being carried)
- Scores and step counter
- Real-time game progression

### 5. **Analysis Tools** (`experiments/analyze_strategies.py`)

Measure emergent behaviors:
- **Cooperation quality**: How close teammates stay
- **Role specialization**: Who becomes offense/defense
- **Strategy comparison**: Different cooperation weights
- **Performance metrics**: Win rates, capture speeds

---

## How Cooperation & Competition Work Together

### The Key Insight: Aligned vs Opposing Goals

```
COOPERATIVE MECHANISMS:
┌─────────────────────────────────────────────┐
│ All teammates get BONUS when TEAM succeeds  │
│                                              │
│ Red Team Captures Flag:                      │
│   - Capturing agent: +10                    │
│   - Other red agents: +5 (shared bonus)     │
│                                              │
│ Result: Agents learn to HELP each other     │
└─────────────────────────────────────────────┘

COMPETITIVE MECHANISMS:
┌─────────────────────────────────────────────┐
│ Opposing teams have CONFLICTING objectives  │
│                                              │
│ Red attacks (wants Blue flag)                │
│ Blue defends (protects Red flag)             │
│                                              │
│ Result: Natural adversarial behavior        │
└─────────────────────────────────────────────┘

BALANCED TRAINING:
┌──────────────────────────────────────────────────┐
│ combined_reward = individual + 0.5 * team_avg    │
│                                                   │
│ 0.5 weight means:                                │
│ - Agents partially care about own reward        │
│ - Agents partially care about team success      │
│ - Balance between selfish and altruistic        │
│                                                  │
│ Can adjust: 0.0 (pure selfish)                  │
│            1.0 (pure teamwork)                  │
│            2.0 (extreme cooperation)            │
└──────────────────────────────────────────────────┘
```

### Why This Creates Realistic Gameplay

```
Emergent Strategy:
  Episode 1-10:  Random movement, no coordination
  Episode 11-30: Some agents move toward flags
  Episode 31-50: Teammates start moving together
  Episode 51+:   Clear attack vs defense roles
  
Example Emergence:
  • Agent 0 learns to stay near flag (defense)
  • Agent 1 learns to stay near flag (defense)
  • Agent 2 learns to move toward opponent (offense)
  
  Why?
  - Team gets bonus when flag captured
  - Agents that attack = more captures
  - Agents that defend = prevent opponent captures
  - Nash equilibrium: some specialize in each role
```

---

## Customization Guide

### Adjust Cooperation Level

```python
# In experiments/train_ctf_teams.py

# Pure competition (agents ignore teammates)
combined_reward = rewards[agent_id] + 0.0 * team_bonuses[agent_id]

# Balanced (default)
combined_reward = rewards[agent_id] + 0.5 * team_bonuses[agent_id]

# Heavy cooperation
combined_reward = rewards[agent_id] + 2.0 * team_bonuses[agent_id]
```

### Change Game Parameters

```python
# In experiments/train_ctf_teams.py main():

# Larger grid = harder exploration
env = CTFEnv(grid_size=20, team_size=3, max_steps=500)

# Fewer agents = faster training
env = CTFEnv(grid_size=15, team_size=2, max_steps=300)

# Shorter episodes = quicker convergence
env = CTFEnv(grid_size=15, team_size=3, max_steps=100)
```

### Modify Reward Structure

```python
# In envs/ctf_env.py _calculate_rewards():

# Make capture harder
if distance_to_base < 1.5:
    reward += 5.0  # Reduced from 10.0

# Increase exploration incentive
if distance_to_flag < 3:
    reward += 0.2  # Increased from 0.05

# Reduce defense bonus
if distance_to_base < 4:
    reward += 0.005  # Reduced from 0.02
```

---

## Expected Learning Progression

```
LEARNING CURVE:
Episode 0:      0% capture rate   (random actions)
Episode 10:     5-15% capture      (first explorations)
Episode 30:     20-40% capture     (strategy emerging)
Episode 50:     40-60% capture     (good coordination)
Episode 100:    60-80% capture     (optimized play)
Episode 200+:   80-95% capture     (near optimal)

WHAT AGENTS LEARN:
- How to navigate to objectives
- Team coordination (when cooperation_weight > 0)
- Risk assessment (vs defenders)
- When to attack vs defend
- Teammate recognition (implicit)
```

---

## Game Modes Included

See `experiments/game_modes.py` for these modes:

| Mode | Description | Use Case |
|------|-------------|----------|
| **Standard** | Classic flag capture | General multi-agent learning |
| **Asymmetric** | 2v4 teams | Test skill-based play |
| **King of Hill** | Control center point | Territorial mechanics |
| **Deathmatch** | Score by proximity | Direct conflict |
| **Relay** | Pass baton through checkpoints | Handoff mechanics |
| **Tower Defense** | Defend/attack tower | Asymmetric roles |

Usage:
```python
from experiments.game_modes import create_game_mode

env = create_game_mode('relay')
env = create_game_mode('tower')
```

---

## Measuring Success

### Metrics to Track

```python
# 1. Capture Rate (primary metric)
captures_per_episode = np.mean(rewards > 0)
# Target: >50% by episode 50

# 2. Efficiency (captures per step)
efficiency = total_captures / total_steps
# Target: Decreasing capture times

# 3. Cooperation Quality (teammate proximity)
avg_teammate_distance = ...
# Target: <5 grid units for good coordination

# 4. Role Specialization
offense_percentage = ...
defense_percentage = ...
# Target: Clear split (some agents specialize)

# 5. Team Balance (both teams win sometimes)
red_wins = sum(red_score > 0)
blue_wins = sum(blue_score > 0)
win_balance = abs(red_wins - blue_wins) / total_games
# Target: <0.1 (roughly equal)
```

---

## Common Issues & Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| Training too slow | Large grid/many agents | Reduce grid_size to 10, team_size to 2 |
| One team always wins | Asymmetric setup | Verify base symmetry, check spawn points |
| No cooperation | Low weight | Increase combined_reward coefficient to 1.0+ |
| Agents get stuck | Reward too sparse | Add intermediate rewards (near flag, etc.) |
| High variance in wins | Insufficient training | Run 200+ episodes |

---

## Next Level: Advanced Techniques

### 1. Communication between Agents

```python
class CommIQLAgent(IQLAgent):
    def get_message(self, state):
        # Learn to send messages to teammates
        return self.compute_message(state)
    
    def choose_action(self, state, teammate_messages):
        # Condition action on messages received
        return self.choose_with_communication(state, messages)
```

### 2. Hierarchical Learning

```python
# Team leader makes high-level decisions
team_strategy = team_leader.choose_strategy(state)

# Individual agents follow strategy
individual_action = agent.choose_action(state, team_strategy)
```

### 3. Curriculum Learning

```python
# Start with simple 2v2, small grid
env = CTFEnv(grid_size=10, team_size=2)
train(env, episodes=50)

# Increase difficulty gradually
env = CTFEnv(grid_size=15, team_size=3)
train(env, episodes=100, init_from_previous=True)

# Final challenge
env = CTFEnv(grid_size=20, team_size=4)
train(env, episodes=200, init_from_previous=True)
```

### 4. Self-Play Learning

```python
# Train red team against blue team
# Periodically update opponents

for generation in range(10):
    # Red team trains against current blue team
    train_red_team(blue_team)
    
    # Blue team trains against updated red team
    train_blue_team(red_team)
    
    # Evaluate new strategies
    evaluate_teams()
```

---

## Summary

You now have a **complete multi-agent game simulation framework** that implements:

✅ **Cooperative mechanics** (shared team rewards, proximity bonuses)  
✅ **Competitive mechanics** (opposing objectives, direct conflict)  
✅ **Emergent behaviors** (role specialization, strategies)  
✅ **Training systems** (IQL agents, reward shaping)  
✅ **Analysis tools** (visualization, metrics, strategy comparison)  
✅ **Multiple game modes** (CTF, King of Hill, Tower Defense, etc.)  

**To get started:**

```bash
# Basic training
EPISODES=100 ./marl_env/bin/python -m experiments.train_ctf_teams

# Watch gameplay
./marl_env/bin/python -m experiments.visualize_ctf

# Analyze strategies
./marl_env/bin/python -m experiments.analyze_strategies
```

🎮 **Now you can create any competitive & cooperative multi-agent game you imagine!**
