"""
Analyze and compare different multi-agent strategies.

Shows how to:
1. Train with different cooperation levels
2. Compare competitive vs cooperative agents
3. Measure emergent behaviors
4. Analyze strategy differences
"""

import os
import numpy as np
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from envs.ctf_env import CTFEnv
from algorithms.iql import IQLAgent


def encode_observation(obs, grid_size=15, num_agents=6):
    """Encode observation to state vector."""
    vec = []
    for i in range(num_agents):
        pos = obs['agent_positions'].get(i, (0, 0))
        vec.extend([pos[0], pos[1]])
    red_flag = obs['red_flag_position']
    blue_flag = obs['blue_flag_position']
    vec.extend([red_flag[0], red_flag[1], blue_flag[0], blue_flag[1]])
    return np.array(vec, dtype=np.float32)


def train_with_cooperation_weight(cooperation_weight, num_episodes=50):
    """
    Train agents with specific cooperation weight.
    
    cooperation_weight = 0.0: Fully individual (no teamwork)
    cooperation_weight = 1.0: Balanced individual and team rewards
    cooperation_weight = 2.0: Heavy emphasis on teamwork
    """
    env = CTFEnv(grid_size=15, team_size=3, max_steps=200)
    state_size = 2 * env.num_agents + 4
    agents = [IQLAgent(state_size=state_size, lr=0.01) for _ in range(env.num_agents)]
    
    action_map = {0: 'up', 1: 'down', 2: 'left', 3: 'right'}
    episode_rewards = []
    
    for ep in range(num_episodes):
        obs = env.reset()
        state = encode_observation(obs, grid_size=env.grid_size, num_agents=env.num_agents)
        done = False
        
        while not done:
            # Choose actions
            actions = {}
            actions_discrete = {}
            for agent_id in range(env.num_agents):
                discrete_action = agents[agent_id].choose_action(state)
                actions_discrete[agent_id] = discrete_action
                actions[agent_id] = action_map[discrete_action]
            
            # Step environment
            next_obs, rewards, done, _ = env.step(actions)
            next_state = encode_observation(next_obs, grid_size=env.grid_size, num_agents=env.num_agents)
            
            # Apply cooperation weight
            red_team = [0, 1, 2]
            blue_team = [3, 4, 5]
            
            for agent_id in range(env.num_agents):
                team = red_team if agent_id in red_team else blue_team
                team_reward = sum(rewards[a] for a in team) / len(team)
                combined_reward = rewards[agent_id] + cooperation_weight * team_reward
                agents[agent_id].update(state, actions_discrete[agent_id], combined_reward, next_state, done)
            
            state = next_state
        
        episode_rewards.append(env.red_score + env.blue_score)
        
        for agent in agents:
            agent.decay_epsilon()
    
    return np.array(episode_rewards)


def compare_strategies():
    """Compare different cooperation levels."""
    cooperation_levels = [0.0, 0.5, 1.0, 2.0]
    results = {}
    
    print("Comparing cooperation levels...")
    print("=" * 60)
    
    for coop in cooperation_levels:
        print(f"\nTraining with cooperation_weight={coop}...")
        rewards = train_with_cooperation_weight(coop, num_episodes=30)
        results[coop] = rewards
        
        print(f"  Average reward: {np.mean(rewards):.3f}")
        print(f"  Win rate (captures/episodes): {np.sum(rewards > 0) / len(rewards) * 100:.1f}%")
        print(f"  Best episode: {np.max(rewards):.1f}")
    
    # Plot comparison
    print("\n" + "=" * 60)
    print("Results Summary:")
    print("=" * 60)
    
    for coop in cooperation_levels:
        mean = np.mean(results[coop])
        win_rate = np.sum(results[coop] > 0) / len(results[coop])
        print(f"Cooperation {coop:.1f}: Mean={mean:.3f}, Win%={win_rate*100:.1f}%")
    
    if plt:
        plt.figure(figsize=(12, 5))
        
        for coop in cooperation_levels:
            plt.plot(results[coop], label=f'Coop={coop}', alpha=0.7)
        
        plt.xlabel('Episode')
        plt.ylabel('Captures per Episode')
        plt.title('Effect of Cooperation Weight on Team Performance')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('results/cooperation_comparison.png', dpi=150)
        print("\nPlot saved to: results/cooperation_comparison.png")


def analyze_team_behavior():
    """Analyze how agents distribute between offense and defense."""
    env = CTFEnv(grid_size=15, team_size=3, max_steps=200)
    state_size = 2 * env.num_agents + 4
    agents = [IQLAgent(state_size=state_size) for _ in range(env.num_agents)]
    
    action_map = {0: 'up', 1: 'down', 2: 'left', 3: 'right'}
    
    # Track agent roles
    offense_tendency = {i: 0.0 for i in range(6)}
    defense_tendency = {i: 0.0 for i in range(6)}
    
    print("\nAnalyzing agent role specialization...")
    
    for ep in range(20):
        obs = env.reset()
        state = encode_observation(obs, grid_size=env.grid_size, num_agents=env.num_agents)
        done = False
        
        while not done:
            actions = {}
            for agent_id in range(env.num_agents):
                discrete_action = agents[agent_id].choose_action(state)
                actions[agent_id] = action_map[discrete_action]
                
                # Track position relative to bases
                pos = env.agent_positions[agent_id]
                own_base = env.red_flag_base if agent_id < 3 else env.blue_flag_base
                opponent_base = env.blue_flag_base if agent_id < 3 else env.red_flag_base
                
                dist_to_own = env._distance(pos, own_base)
                dist_to_opponent = env._distance(pos, opponent_base)
                
                # Offensive tendency: closer to opponent base
                if dist_to_opponent < dist_to_own:
                    offense_tendency[agent_id] += 1
                else:
                    defense_tendency[agent_id] += 1
            
            next_obs, _, done, _ = env.step(actions)
            state = encode_observation(next_obs, grid_size=env.grid_size, num_agents=env.num_agents)
    
    print("\nAgent Role Distribution:")
    print("=" * 60)
    red_team = [0, 1, 2]
    blue_team = [3, 4, 5]
    
    for team_name, team_ids in [("RED", red_team), ("BLUE", blue_team)]:
        print(f"\n{team_name} TEAM:")
        for agent_id in team_ids:
            total = offense_tendency[agent_id] + defense_tendency[agent_id]
            if total == 0:
                continue
            offense_pct = offense_tendency[agent_id] / total * 100
            role = "OFFENSE" if offense_pct > 60 else ("DEFENSE" if offense_pct < 40 else "BALANCED")
            print(f"  Agent {agent_id}: {offense_pct:.0f}% offense -> {role}")


def measure_coordination_quality():
    """Measure how well agents coordinate (using teammate proximity)."""
    env = CTFEnv(grid_size=15, team_size=3, max_steps=200)
    state_size = 2 * env.num_agents + 4
    agents = [IQLAgent(state_size=state_size) for _ in range(env.num_agents)]
    
    action_map = {0: 'up', 1: 'down', 2: 'left', 3: 'right'}
    
    teammate_distances = []
    
    print("\nMeasuring coordination quality (teammate proximity)...")
    
    for ep in range(20):
        obs = env.reset()
        state = encode_observation(obs, grid_size=env.grid_size, num_agents=env.num_agents)
        done = False
        
        while not done:
            actions = {}
            for agent_id in range(env.num_agents):
                discrete_action = agents[agent_id].choose_action(state)
                actions[agent_id] = action_map[discrete_action]
            
            # Measure teammate distances
            red_team = [0, 1, 2]
            blue_team = [3, 4, 5]
            
            for team_ids in [red_team, blue_team]:
                for i, a1 in enumerate(team_ids):
                    for a2 in team_ids[i+1:]:
                        dist = env._distance(
                            env.agent_positions[a1],
                            env.agent_positions[a2]
                        )
                        teammate_distances.append(dist)
            
            next_obs, _, done, _ = env.step(actions)
            state = encode_observation(next_obs, grid_size=env.grid_size, num_agents=env.num_agents)
    
    mean_dist = np.mean(teammate_distances)
    std_dist = np.std(teammate_distances)
    
    print("\nCoordination Quality Metrics:")
    print("=" * 60)
    print(f"Average distance between teammates: {mean_dist:.2f} grid units")
    print(f"Std deviation: {std_dist:.2f}")
    print(f"Min distance: {np.min(teammate_distances):.2f}")
    print(f"Max distance: {np.max(teammate_distances):.2f}")
    
    if mean_dist < 3:
        print("\n✓ GOOD COORDINATION: Teams stay close together")
    elif mean_dist < 5:
        print("\n≈ MODERATE COORDINATION: Some spread")
    else:
        print("\n✗ POOR COORDINATION: Teams spread out")


def main():
    os.makedirs("results", exist_ok=True)
    
    print("\n" + "=" * 70)
    print("COMPETITIVE & COOPERATIVE MULTI-AGENT ANALYSIS")
    print("=" * 70)
    
    # Run analyses
    compare_strategies()
    print("\n" + "=" * 70)
    analyze_team_behavior()
    print("\n" + "=" * 70)
    measure_coordination_quality()
    
    print("\n" + "=" * 70)
    print("Analysis complete! Check results/ directory for plots.")
    print("=" * 70)


if __name__ == "__main__":
    main()
