"""
Team-based CTF training script with competitive and cooperative mechanics.

This script trains agents using team-based rewards:
- Cooperative: agents within same team get shared rewards
- Competitive: red team vs blue team compete for flag capture
"""

import os
import numpy as np
import torch
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x

from envs.ctf_env import CTFEnv
from algorithms.iql import IQLAgent


def encode_observation(obs, grid_size=15, num_agents=6):
    """Encode full observation to fixed-size state vector."""
    vec = []
    
    # Agent positions
    for i in range(num_agents):
        pos = obs['agent_positions'].get(i, (0, 0))
        vec.extend([pos[0], pos[1]])
    
    # Flag positions
    red_flag = obs['red_flag_position']
    blue_flag = obs['blue_flag_position']
    vec.extend([red_flag[0], red_flag[1], blue_flag[0], blue_flag[1]])
    
    return np.array(vec, dtype=np.float32)


def main():
    os.makedirs("results", exist_ok=True)
    
    env = CTFEnv(grid_size=15, team_size=3, max_steps=500)
    
    # State size: 6 agents * 2 coords + 2 flags * 2 coords = 16
    state_size = 2 * env.num_agents + 4
    
    # One IQL agent per player
    agents = [IQLAgent(state_size=state_size, lr=0.01, gamma=0.99) for _ in range(env.num_agents)]
    
    action_map = {0: 'up', 1: 'down', 2: 'left', 3: 'right'}
    
    episodes = int(os.environ.get("EPISODES", "100"))
    red_wins = 0
    blue_wins = 0
    episode_rewards = []
    
    for ep in tqdm(range(episodes), desc="Episodes"):
        obs = env.reset()
        state = encode_observation(obs, grid_size=env.grid_size, num_agents=env.num_agents)
        done = False
        step = 0
        
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
            
            # Apply team-based cooperative bonus
            team_bonuses = {}
            red_team = [0, 1, 2]
            blue_team = [3, 4, 5]
            
            # Cooperative reward within teams
            for agent_id in range(env.num_agents):
                team = red_team if agent_id in red_team else blue_team
                team_reward = sum(rewards[a] for a in team) / len(team)
                team_bonuses[agent_id] = team_reward
            
            # Update each agent with cooperative + individual reward
            for agent_id in range(env.num_agents):
                combined_reward = rewards[agent_id] + 0.5 * team_bonuses[agent_id]
                agents[agent_id].update(state, actions_discrete[agent_id], combined_reward, next_state, done)
            
            state = next_state
            step += 1
        
        # Track scores
        if env.red_score > 0:
            red_wins += 1
        elif env.blue_score > 0:
            blue_wins += 1
        
        episode_rewards.append(env.red_score + env.blue_score)
        
        # Decay exploration
        for agent in agents:
            agent.decay_epsilon()
        
        # Print progress
        if (ep + 1) % 10 == 0:
            win_rate = (red_wins + blue_wins) / (ep + 1) * 100
            tqdm.write(
                f"Episode {ep+1}: Red={red_wins}, Blue={blue_wins}, "
                f"Win Rate={win_rate:.1f}%, Avg Score={np.mean(episode_rewards[-10:]):.2f}"
            )
    
    # Save results
    np.save("results/ctf_team_rewards.npy", np.array(episode_rewards))
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Red Team Wins: {red_wins}/{episodes} ({red_wins/episodes*100:.1f}%)")
    print(f"Blue Team Wins: {blue_wins}/{episodes} ({blue_wins/episodes*100:.1f}%)")
    print(f"Average Episode Score: {np.mean(episode_rewards):.2f}")
    print(f"Max Episode Score: {np.max(episode_rewards):.2f}")
    print("Rewards saved to results/ctf_team_rewards.npy")


if __name__ == "__main__":
    main()
