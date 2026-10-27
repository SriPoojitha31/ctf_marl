"""
Clash of Titans Visualization

Shows real-time or replay of the game with:
- Towers and health bars
- Titans and escort groups
- Agent positions and team colors
- Combat effects
"""

import numpy as np
from envs.clash_of_titans import ClashOfTitansEnv
from algorithms.iql import IQLAgent


def encode_observation(obs, grid_size=20, num_agents=6):
    """Encode observation to state vector."""
    vec = []
    for i in range(num_agents):
        pos = obs['agent_positions'].get(i, (0, 0))
        vec.extend([pos[0] / grid_size, pos[1] / grid_size])
    red_tower = obs['red_tower_pos']
    blue_tower = obs['blue_tower_pos']
    vec.extend([
        red_tower[0] / grid_size, red_tower[1] / grid_size,
        obs['red_tower_health'] / 100,
        blue_tower[0] / grid_size, blue_tower[1] / grid_size,
        obs['blue_tower_health'] / 100,
    ])
    red_titan_pos = obs['red_titan_pos'] if obs['red_titan_pos'] else (0, 0)
    blue_titan_pos = obs['blue_titan_pos'] if obs['blue_titan_pos'] else (0, 0)
    vec.extend([
        red_titan_pos[0] / grid_size, red_titan_pos[1] / grid_size,
        obs['red_titan_health'] / 30,
        blue_titan_pos[0] / grid_size, blue_titan_pos[1] / grid_size,
        obs['blue_titan_health'] / 30,
        obs['red_checkpoint'] / 3, obs['blue_checkpoint'] / 3,
        obs['red_score'] / 100, obs['blue_score'] / 100,
    ])
    return np.array(vec, dtype=np.float32)


def print_game_state(env, step, title=""):
    """Print ASCII visualization of Clash of Titans."""
    grid = [['.' for _ in range(env.grid_size)] for _ in range(env.grid_size)]
    
    # Draw towers
    rt = env.red_tower_base
    bt = env.blue_tower_base
    grid[rt[1]][rt[0]] = '⬤' if env.red_tower_health > 50 else '●' if env.red_tower_health > 25 else '○'
    grid[bt[1]][bt[0]] = '⊕' if env.blue_tower_health > 50 else '⊙' if env.blue_tower_health > 25 else '◯'
    
    # Draw titans
    if env.red_titan:
        rt_pos = (int(env.red_titan.pos[0]), int(env.red_titan.pos[1]))
        if 0 <= rt_pos[1] < env.grid_size and 0 <= rt_pos[0] < env.grid_size:
            grid[rt_pos[1]][rt_pos[0]] = 'T' if env.red_titan.health > 15 else 't'
    
    if env.blue_titan:
        bt_pos = (int(env.blue_titan.pos[0]), int(env.blue_titan.pos[1]))
        if 0 <= bt_pos[1] < env.grid_size and 0 <= bt_pos[0] < env.grid_size:
            grid[bt_pos[1]][bt_pos[0]] = 'P' if env.blue_titan.health > 15 else 'p'
    
    # Draw agents
    agent_count = {}
    for agent_id, (x, y) in env.agent_positions.items():
        x, y = int(x), int(y)
        if 0 <= y < env.grid_size and 0 <= x < env.grid_size:
            team = 'red' if agent_id in env.red_team else 'blue'
            key = (x, y)
            if key not in agent_count:
                agent_count[key] = {'red': 0, 'blue': 0}
            agent_count[key][team] += 1
    
    for (x, y), counts in agent_count.items():
        if counts['red'] > 0 and counts['blue'] > 0:
            grid[y][x] = 'X'  # Combat
        elif counts['red'] > 1:
            grid[y][x] = 'R'
        elif counts['red'] == 1:
            grid[y][x] = 'r'
        elif counts['blue'] > 1:
            grid[y][x] = 'B'
        elif counts['blue'] == 1:
            grid[y][x] = 'b'
    
    # Print display
    print(f"\n{'='*70}")
    print(f"Step {step:3d} | {title}")
    print(f"Red Tower: {int(env.red_tower_health):3d}/100 HP | Blue Tower: {int(env.blue_tower_health):3d}/100 HP")
    print(f"Red Score: {env.red_score:3d} | Blue Score: {env.blue_score:3d}")
    print(f"Red Titan CP: {env.red_titan_checkpoint}/3 | Blue Titan CP: {env.blue_titan_checkpoint}/3")
    print(f"{'='*70}")
    
    for row in grid:
        print(''.join(row))
    
    print("\nLegend:")
    print("  ⬤/●/○ = Red Tower (full/half/critical)")
    print("  ⊕/⊙/◯ = Blue Tower (full/half/critical)")
    print("  T/t = Red Titan (healthy/damaged)")
    print("  P/p = Blue Titan (healthy/damaged)")
    print("  r/R = Red agents (single/group)")
    print("  b/B = Blue agents (single/group)")
    print("  X = Combat zone")


def play_episode(env, agents, action_map, visualize=True, max_steps=500):
    """Play one Clash of Titans episode."""
    obs = env.reset()
    state = encode_observation(obs, grid_size=env.grid_size, num_agents=env.num_agents)
    done = False
    step = 0
    
    if visualize:
        print_game_state(env, 0, "GAME START")
    
    while not done and step < max_steps:
        # Get actions (greedy, no exploration)
        actions = {}
        for agent_id in range(env.num_agents):
            q_values = agents[agent_id].get_q_values(state)
            discrete_action = np.argmax(q_values)
            actions[agent_id] = action_map[discrete_action]
        
        # Step environment
        next_obs, rewards, done, _ = env.step(actions)
        next_state = encode_observation(next_obs, grid_size=env.grid_size, num_agents=env.num_agents)
        state = next_state
        step += 1
        
        if visualize and (step % 10 == 0 or done):
            title = ""
            if env.red_tower_health <= 0:
                title = "🔵 BLUE TEAM DESTROYED RED TOWER!"
            elif env.blue_tower_health <= 0:
                title = "🔴 RED TEAM DESTROYED BLUE TOWER!"
            elif env.red_score >= 100:
                title = "🔴 RED TEAM REACHES 100 POINTS!"
            elif env.blue_score >= 100:
                title = "🔵 BLUE TEAM REACHES 100 POINTS!"
            else:
                title = f"Game in progress..."
            
            print_game_state(env, step, title)
    
    # Final result
    if visualize:
        print("\n" + "="*70)
        if env.red_tower_health <= 0:
            print("🔵 GAME OVER - BLUE TEAM WINS! (Tower Destruction)")
        elif env.blue_tower_health <= 0:
            print("🔴 GAME OVER - RED TEAM WINS! (Tower Destruction)")
        elif env.red_score > env.blue_score:
            print(f"🔴 GAME OVER - RED TEAM WINS! ({env.red_score} points)")
        elif env.blue_score > env.red_score:
            print(f"🔵 GAME OVER - BLUE TEAM WINS! ({env.blue_score} points)")
        else:
            print(f"🤝 GAME OVER - TIE! ({env.red_score} vs {env.blue_score})")
        print("="*70)
    
    return {
        'red_score': env.red_score,
        'blue_score': env.blue_score,
        'winner': 'red' if env.red_score > env.blue_score else ('blue' if env.blue_score > env.red_score else 'tie'),
        'steps': step,
        'red_tower_hp': env.red_tower_health,
        'blue_tower_hp': env.blue_tower_health,
    }


def main():
    env = ClashOfTitansEnv(grid_size=20, team_size=3, max_steps=500)
    state_size = 2*6 + 4 + 4 + 2 + 2
    agents = [IQLAgent(state_size=state_size, action_size=4) for _ in range(env.num_agents)]
    
    action_map = {0: 'up', 1: 'down', 2: 'left', 3: 'right'}
    
    print("\n" + "="*70)
    print("CLASH OF TITANS - GAME VISUALIZATION")
    print("="*70)
    print("\nGame Setup:")
    print("  Red Tower (⬇): Position (2, 2) - Health: 100")
    print("  Blue Tower (⬇): Position (17, 17) - Health: 100")
    print("  Red Titan (T): Escorts toward Blue tower")
    print("  Blue Titan (P): Escorts toward Red tower")
    print("  Red Team: Agents 0-2 (r/R symbols)")
    print("  Blue Team: Agents 3-5 (b/B symbols)")
    print("\nObjectives:")
    print("  1. Escort titan through 3 checkpoints to enemy tower")
    print("  2. Destroy enemy tower (20 damage per successful escort)")
    print("  3. Combat/tagging for extra points")
    print("  4. First team to 100 points OR destroy tower wins!\n")
    
    # Play 3 demo games
    stats = []
    for game_num in range(3):
        print(f"\n{'#'*70}")
        print(f"# GAME {game_num + 1}")
        print(f"{'#'*70}")
        
        result = play_episode(env, agents, action_map, visualize=True, max_steps=500)
        stats.append(result)
    
    # Summary
    print("\n" + "="*70)
    print("SESSION SUMMARY")
    print("="*70)
    red_wins = sum(1 for s in stats if s['winner'] == 'red')
    blue_wins = sum(1 for s in stats if s['winner'] == 'blue')
    
    print(f"Red Wins: {red_wins}")
    print(f"Blue Wins: {blue_wins}")
    print(f"Avg Final Score: {np.mean([s['red_score'] + s['blue_score'] for s in stats]):.1f}")
    print(f"Avg Game Length: {np.mean([s['steps'] for s in stats]):.0f} steps")


if __name__ == "__main__":
    main()
