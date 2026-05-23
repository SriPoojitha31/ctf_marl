"""
Clash of Titans Training Script

Trains 6 agents (3v3) in a hybrid game mode combining:
- Tower defense (destroy enemy tower)
- Relay mechanics (escort titans to checkpoints)
- Team deathmatch (proximity-based combat)
"""

import os
import numpy as np
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x

from envs.clash_of_titans import ClashOfTitansEnv
from algorithms.iql import IQLAgent


def encode_observation(obs, grid_size=20, num_agents=6):
    """Encode complex observation to state vector."""
    vec = []
    
    # Agent positions
    for i in range(num_agents):
        pos = obs['agent_positions'].get(i, (0, 0))
        vec.extend([pos[0] / grid_size, pos[1] / grid_size])
    
    # Tower positions and health
    red_tower = obs['red_tower_pos']
    blue_tower = obs['blue_tower_pos']
    vec.extend([
        red_tower[0] / grid_size,
        red_tower[1] / grid_size,
        obs['red_tower_health'] / 100,
        blue_tower[0] / grid_size,
        blue_tower[1] / grid_size,
        obs['blue_tower_health'] / 100,
    ])
    
    # Titan positions and health
    red_titan_pos = obs['red_titan_pos'] if obs['red_titan_pos'] else (0, 0)
    blue_titan_pos = obs['blue_titan_pos'] if obs['blue_titan_pos'] else (0, 0)
    vec.extend([
        red_titan_pos[0] / grid_size,
        red_titan_pos[1] / grid_size,
        obs['red_titan_health'] / 30,
        blue_titan_pos[0] / grid_size,
        blue_titan_pos[1] / grid_size,
        obs['blue_titan_health'] / 30,
    ])
    
    # Checkpoints progress
    vec.extend([
        obs['red_checkpoint'] / 3,
        obs['blue_checkpoint'] / 3,
    ])
    
    # Scores
    vec.extend([
        obs['red_score'] / 100,
        obs['blue_score'] / 100,
    ])
    
    return np.array(vec, dtype=np.float32)


def main():
    os.makedirs("results", exist_ok=True)
    
    print("\n" + "="*70)
    print("CLASH OF TITANS - Training 6 Agents (3v3)")
    print("="*70)
    print("\nGame Features:")
    print("  - Tower Defense: Destroy enemy tower (health: 100)")
    print("  - Relay Mechanics: Escort titans through checkpoints")
    print("  - Team Deathmatch: Proximity-based combat scoring")
    print("  - 6 Agents: Red team (0-2) vs Blue team (3-5)")
    print("  - Victory: First to 100 points OR destroy tower\n")
    
    env = ClashOfTitansEnv(grid_size=20, team_size=3, max_steps=500)
    
    # State size based on encoded obs
    state_size = 2*6 + 4 + 4 + 2 + 2  # = 20
    
    # Create agents with learning parameters
    agents = [
        IQLAgent(state_size=state_size, action_size=4, lr=0.01, gamma=0.95, epsilon=1.0)
        for _ in range(env.num_agents)
    ]
    
    action_map = {0: 'up', 1: 'down', 2: 'left', 3: 'right'}
    
    episodes = int(os.environ.get("EPISODES", "100"))
    red_wins = 0
    blue_wins = 0
    episode_rewards = []
    episode_tower_damage = []
    
    for ep in tqdm(range(episodes), desc="Episodes"):
        obs = env.reset()
        state = encode_observation(obs, grid_size=env.grid_size, num_agents=env.num_agents)
        done = False
        step = 0
        
        red_tower_damage = 0
        blue_tower_damage = 0
        
        while not done and step < env.max_steps:
            # All agents choose actions
            actions = {}
            actions_discrete = {}
            
            for agent_id in range(env.num_agents):
                discrete_action = agents[agent_id].choose_action(state)
                actions_discrete[agent_id] = discrete_action
                actions[agent_id] = action_map[discrete_action]
            
            # Environment step
            next_obs, rewards, done, _ = env.step(actions)
            next_state = encode_observation(next_obs, grid_size=env.grid_size, num_agents=env.num_agents)
            
            # Track tower damage
            if next_obs['red_tower_health'] < obs['red_tower_health']:
                red_tower_damage += obs['red_tower_health'] - next_obs['red_tower_health']
            if next_obs['blue_tower_health'] < obs['blue_tower_health']:
                blue_tower_damage += obs['blue_tower_health'] - next_obs['blue_tower_health']
            
            # Apply team-based cooperative rewards
            red_team = [0, 1, 2]
            blue_team = [3, 4, 5]
            
            team_bonuses = {}
            for agent_id in range(env.num_agents):
                team = red_team if agent_id in red_team else blue_team
                team_reward = sum(rewards[a] for a in team) / len(team)
                team_bonuses[agent_id] = team_reward
            
            # Update agents with cooperative + individual rewards
            for agent_id in range(env.num_agents):
                combined_reward = rewards[agent_id] + 0.5 * team_bonuses[agent_id]
                agents[agent_id].update(state, actions_discrete[agent_id], combined_reward, next_state, done)
            
            obs = next_obs
            state = next_state
            step += 1
        
        # Track outcomes
        total_score = env.red_score + env.blue_score
        episode_rewards.append(total_score)
        episode_tower_damage.append(red_tower_damage + blue_tower_damage)
        
        if env.red_score > env.blue_score:
            red_wins += 1
        elif env.blue_score > env.red_score:
            blue_wins += 1
        
        # Decay exploration
        for agent in agents:
            agent.decay_epsilon()
        
        # Progress report
        if (ep + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            red_pct = red_wins / (ep + 1) * 100
            blue_pct = blue_wins / (ep + 1) * 100
            avg_tower_dmg = np.mean(episode_tower_damage[-10:])
            
            tqdm.write(
                f"Episode {ep+1:3d}: "
                f"Red={red_wins:2d} ({red_pct:5.1f}%) | "
                f"Blue={blue_wins:2d} ({blue_pct:5.1f}%) | "
                f"AvgScore={avg_reward:.2f} | "
                f"TowerDmg={avg_tower_dmg:.2f}"
            )
    
    # Save results
    np.save("results/clash_rewards.npy", np.array(episode_rewards))
    np.save("results/clash_tower_damage.npy", np.array(episode_tower_damage))
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"Red Team Wins: {red_wins}/{episodes} ({red_wins/episodes*100:.1f}%)")
    print(f"Blue Team Wins: {blue_wins}/{episodes} ({blue_wins/episodes*100:.1f}%)")
    print(f"Average Score: {np.mean(episode_rewards):.2f}")
    print(f"Max Score: {np.max(episode_rewards):.2f}")
    print(f"Average Tower Damage: {np.mean(episode_tower_damage):.2f}")
    print("\nResults saved to:")
    print("  - results/clash_rewards.npy")
    print("  - results/clash_tower_damage.npy")


if __name__ == "__main__":
    main()
