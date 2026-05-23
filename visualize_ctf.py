"""
Visualization of CTF game with team-based mechanics.
Shows real-time or saved replay of agents playing.
"""

import numpy as np
import os
from envs.ctf_env import CTFEnv
from algorithms.iql import IQLAgent


def encode_observation(obs, grid_size=15, num_agents=6):
    """Encode full observation to fixed-size state vector."""
    vec = []
    for i in range(num_agents):
        pos = obs['agent_positions'].get(i, (0, 0))
        vec.extend([pos[0], pos[1]])
    red_flag = obs['red_flag_position']
    blue_flag = obs['blue_flag_position']
    vec.extend([red_flag[0], red_flag[1], blue_flag[0], blue_flag[1]])
    return np.array(vec, dtype=np.float32)


def print_game_state(env, step):
    """Print ASCII visualization of the game."""
    grid = [['.' for _ in range(env.grid_size)] for _ in range(env.grid_size)]
    
    # Place flags
    r_flag = env._get_red_flag_pos()
    b_flag = env._get_blue_flag_pos()
    grid[r_flag[1]][r_flag[0]] = 'R'
    grid[b_flag[1]][b_flag[0]] = 'B'
    
    # Place agents
    for agent_id, (x, y) in env.agent_positions.items():
        if agent_id in env.red_team:
            grid[y][x] = 'r' if grid[y][x] == '.' else '1'
        else:
            grid[y][x] = 'b' if grid[y][x] == '.' else '2'
    
    # Print
    print(f"\n{'='*60}")
    print(f"Step {step} | Red Score: {env.red_score} | Blue Score: {env.blue_score}")
    print(f"{'='*60}")
    
    for row in grid:
        print(''.join(row))
    
    print(f"\nLegend: R/B=Flags | r/b=Agents | 1/2=Stacked agents")
    print(f"Red base: {env.red_flag_base} | Blue base: {env.blue_flag_base}")
    if env.flag_carriers:
        print(f"Flag carriers: {env.flag_carriers}")


def play_episode(env, agents, action_map, visualize=True, max_steps=500):
    """
    Play one episode with trained agents.
    
    Args:
        env: CTFEnv instance
        agents: list of IQLAgent
        action_map: dict mapping action index to action string
        visualize: whether to print game state
        max_steps: maximum steps per episode
    
    Returns:
        game statistics
    """
    obs = env.reset()
    state = encode_observation(obs, grid_size=env.grid_size, num_agents=env.num_agents)
    done = False
    step = 0
    
    if visualize:
        print_game_state(env, 0)
    
    while not done and step < max_steps:
        # Get actions
        actions = {}
        for agent_id in range(env.num_agents):
            # Use greedy policy (no exploration)
            q_values = agents[agent_id].get_q_values(state)
            discrete_action = np.argmax(q_values)
            actions[agent_id] = action_map[discrete_action]
        
        # Step environment
        next_obs, rewards, done, _ = env.step(actions)
        next_state = encode_observation(next_obs, grid_size=env.grid_size, num_agents=env.num_agents)
        state = next_state
        step += 1
        
        if visualize and step % 5 == 0:
            print_game_state(env, step)
    
    if visualize:
        print_game_state(env, step)
        print(f"\n✓ GAME OVER!")
        if env.red_score > 0:
            print(f"🏆 RED TEAM WINS! (Score: {env.red_score})")
        elif env.blue_score > 0:
            print(f"🏆 BLUE TEAM WINS! (Score: {env.blue_score})")
        else:
            print(f"🤝 TIE - No team captured flag in {step} steps")
    
    return {
        'red_score': env.red_score,
        'blue_score': env.blue_score,
        'steps': step,
        'winner': 'red' if env.red_score > 0 else ('blue' if env.blue_score > 0 else 'tie')
    }


def main():
    # Initialize environment and agents
    env = CTFEnv(grid_size=15, team_size=3, max_steps=500)
    state_size = 2 * env.num_agents + 4
    agents = [IQLAgent(state_size=state_size) for _ in range(env.num_agents)]
    
    action_map = {0: 'up', 1: 'down', 2: 'left', 3: 'right'}
    
    print("\n" + "="*60)
    print("CTF Team-Based Game Visualization")
    print("="*60)
    print("\nGame Setup:")
    print(f"- Grid: {env.grid_size}x{env.grid_size}")
    print(f"- Red Team: Agents 0-2 (base at {env.red_flag_base})")
    print(f"- Blue Team: Agents 3-5 (base at {env.blue_flag_base})")
    print(f"- Max steps: {env.max_steps}")
    print("\nNote: Agents make random moves (no trained model loaded)")
    print("To use trained models, modify to load from results/ directory\n")
    
    # Play a few random episodes
    stats = []
    for ep in range(3):
        print(f"\n{'#'*60}")
        print(f"# Playing Episode {ep + 1}")
        print(f"{'#'*60}")
        result = play_episode(env, agents, action_map, visualize=True, max_steps=500)
        stats.append(result)
    
    # Print summary
    print("\n" + "="*60)
    print("EPISODE SUMMARY")
    print("="*60)
    red_wins = sum(1 for s in stats if s['winner'] == 'red')
    blue_wins = sum(1 for s in stats if s['winner'] == 'blue')
    ties = sum(1 for s in stats if s['winner'] == 'tie')
    
    print(f"Red Wins: {red_wins}")
    print(f"Blue Wins: {blue_wins}")
    print(f"Ties: {ties}")
    print(f"Average Steps: {np.mean([s['steps'] for s in stats]):.0f}")


if __name__ == "__main__":
    main()
