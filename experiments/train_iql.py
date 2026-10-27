import os
from pathlib import Path

import numpy as np

from algorithms.iql import IQLAgent
from envs.ctf_env import CTFEnv

try:
    import imageio
except Exception:
    imageio = None


ACTION_MAP = {0: "up", 1: "down", 2: "left", 3: "right"}


def obs_to_agent_positions(obs, num_agents):
    return [tuple(obs["agent_positions"][i]) for i in range(num_agents)]


def render_obs_to_image(obs_dict, red_team, size=20):
    canvas = np.ones((size, size, 3), dtype=np.uint8) * 255
    for (ox, oy) in obs_dict.get("obstacles", []):
        canvas[oy, ox] = (160, 160, 160)
    for aid, (x, y) in obs_dict["agent_positions"].items():
        canvas[y, x] = (200, 50, 50) if aid in red_team else (50, 50, 200)
    rx, ry = obs_dict["red_flag_position"]
    bx, by = obs_dict["blue_flag_position"]
    canvas[ry, rx] = (120, 30, 30)
    canvas[by, bx] = (30, 30, 120)
    return canvas


def run(episodes=50, max_steps=100):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    env = CTFEnv(grid_size=20, team_size=5, max_steps=max_steps)
    num_agents = env.num_agents
    agents = [IQLAgent(grid_size=env.grid_size, action_dim=4, epsilon=1.0) for _ in range(num_agents)]

    out_dir = results_dir / "iql_frames"
    out_dir.mkdir(parents=True, exist_ok=True)

    rewards_log = []
    frame_count = 0
    for ep in range(episodes):
        obs = env.reset()
        pos_list = obs_to_agent_positions(obs, num_agents)
        done = False
        ep_reward = 0.0
        step = 0

        while not done and step < max_steps:
            actions = {}
            action_indices = [0] * num_agents
            for aid in range(num_agents):
                enemy_flag = obs["blue_flag_position"] if aid in env.red_team else obs["red_flag_position"]
                action_idx = agents[aid].get_action(pos_list[aid], enemy_flag)
                actions[aid] = ACTION_MAP[action_idx]
                action_indices[aid] = action_idx

            next_obs, rewards, done, _ = env.step(actions)
            next_pos_list = obs_to_agent_positions(next_obs, num_agents)

            for aid in range(num_agents):
                enemy_flag = obs["blue_flag_position"] if aid in env.red_team else obs["red_flag_position"]
                next_enemy_flag = next_obs["blue_flag_position"] if aid in env.red_team else next_obs["red_flag_position"]
                agents[aid].update(pos_list[aid], enemy_flag, action_indices[aid], rewards[aid], next_pos_list[aid], next_enemy_flag, done)

            if imageio is not None:
                frame = render_obs_to_image(next_obs, env.red_team, size=env.grid_size)
                imageio.imwrite(str(out_dir / f"iql_frame_{frame_count:05d}.png"), frame)
                frame_count += 1

            ep_reward += sum(rewards.values())
            obs = next_obs
            pos_list = next_pos_list
            step += 1

        rewards_log.append(ep_reward)
        for agent in agents:
            agent.decay_epsilon()

    rewards = np.array(rewards_log, dtype=np.float32)
    np.save(results_dir / "iql_rewards.npy", rewards)

    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 3))
        plt.plot(rewards, marker="o")
        plt.title("IQL Training Rewards")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(results_dir / "iql_rewards_plot.png")
        plt.close()
    except Exception as exc:
        print("Could not save IQL reward plot:", exc)

    print(f"IQL training complete, frames saved: {frame_count}")
    print("Rewards saved to results/iql_rewards.npy")


if __name__ == "__main__":
    run(
        episodes=int(os.environ.get("EPISODES", "50")),
        max_steps=int(os.environ.get("MAX_STEPS", "100")),
    )
