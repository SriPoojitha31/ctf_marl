import os
import numpy as np
from pathlib import Path

from algorithms.maddpg import MADDPG
from experiments.moba_render import render_moba_frame
from envs.moba_ctf_env import MobaCTFEnv

try:
    import imageio
except Exception:
    imageio = None


ACTION_MAP = {0: "up", 1: "down", 2: "left", 3: "right", 4: "stay"}


def obs_to_agent_states(obs_dict, num_agents, grid_size, max_steps):
    """Encode the MOBA-style observation into per-agent vectors."""
    agent_states = []
    lane_rows = obs_dict["lanes"]
    tower_lookup = {(tower["team"], tower["lane"], tower["tier"]): tower for tower in obs_dict["towers"]}
    role_ids = {"Tank": 0.0, "Fighter": 1.0, "Assassin": 2.0, "Marksman": 3.0, "Support": 4.0}

    for aid in range(num_agents):
        ax, ay = obs_dict["agent_positions"][aid]
        role = obs_dict["agent_roles"][aid]
        hp = obs_dict["agent_health"][aid]
        respawn = obs_dict["respawn_timers"].get(aid, 0)
        own_team = "red" if aid in obs_dict["red_team"] else "blue"
        enemy_team = "blue" if own_team == "red" else "red"
        own_core = obs_dict["red_core_pos"] if own_team == "red" else obs_dict["blue_core_pos"]
        enemy_core = obs_dict["blue_core_pos"] if own_team == "red" else obs_dict["red_core_pos"]
        own_core_hp = obs_dict["red_core_hp"] if own_team == "red" else obs_dict["blue_core_hp"]
        enemy_core_hp = obs_dict["blue_core_hp"] if own_team == "red" else obs_dict["red_core_hp"]

        allies = [idx for idx in (obs_dict["red_team"] if own_team == "red" else obs_dict["blue_team"]) if idx != aid]
        enemies = obs_dict["blue_team"] if own_team == "red" else obs_dict["red_team"]

        nearest_ally = min(
            ((obs_dict["agent_positions"][ally][0] - ax, obs_dict["agent_positions"][ally][1] - ay) for ally in allies),
            key=lambda delta: abs(delta[0]) + abs(delta[1]),
            default=(0, 0),
        )
        nearest_enemy = min(
            ((obs_dict["agent_positions"][enemy][0] - ax, obs_dict["agent_positions"][enemy][1] - ay) for enemy in enemies),
            key=lambda delta: abs(delta[0]) + abs(delta[1]),
            default=(0, 0),
        )

        lane_name = min(lane_rows, key=lambda lane: abs(lane_rows[lane] - ay))
        lane_value = {"top": 0.0, "mid": 0.5, "bot": 1.0}[lane_name]
        enemy_outer = tower_lookup.get((enemy_team, lane_name, "outer"))
        enemy_inner = tower_lookup.get((enemy_team, lane_name, "inner"))

        state = np.array([
            ax / grid_size,
            ay / grid_size,
            hp / 180.0,
            respawn / 3.0,
            role_ids[role] / 4.0,
            own_core[0] / grid_size,
            own_core[1] / grid_size,
            enemy_core[0] / grid_size,
            enemy_core[1] / grid_size,
            own_core_hp / 250.0,
            enemy_core_hp / 250.0,
            lane_value,
            nearest_ally[0] / grid_size,
            nearest_ally[1] / grid_size,
            nearest_enemy[0] / grid_size,
            nearest_enemy[1] / grid_size,
            (enemy_outer["hp"] if enemy_outer else 0.0) / 120.0,
            (enemy_inner["hp"] if enemy_inner else 0.0) / 120.0,
            obs_dict["step"] / max_steps,
            obs_dict["red_score"],
            obs_dict["blue_score"],
        ], dtype=np.float32)
        agent_states.append(state)

    return np.stack(agent_states, axis=0)


def render_obs_to_image(obs_dict, size=15):
    return render_moba_frame(obs_dict, grid_size=size)


def run(smoke_episodes=3, max_steps=100):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    env = MobaCTFEnv(grid_size=15, team_size=5, max_steps=max_steps)
    num_agents = env.num_agents
    state_dim = 21
    action_dim = 5

    agent = MADDPG(num_agents=num_agents, state_dim=state_dim, action_dim=action_dim)

    out_dir = results_dir / "frames"
    out_dir.mkdir(parents=True, exist_ok=True)

    rewards_history = []
    checkpoints_dir = results_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    frame_count = 0
    for ep in range(smoke_episodes):
        obs = env.reset()
        agent_states = obs_to_agent_states(obs, num_agents, env.grid_size, max_steps)

        done = False
        step = 0
        last_rewards = np.zeros((num_agents,), dtype=np.float32)
        while not done and step < max_steps:
            actions = {}
            action_indices = np.zeros((num_agents,), dtype=np.int64)
            for aid in range(num_agents):
                action_idx = agent.select_action(aid, agent_states[aid], explore=True)
                actions[aid] = ACTION_MAP[action_idx]
                action_indices[aid] = action_idx

            next_obs, rewards_dict, done, _ = env.step(actions)
            rewards = np.array([rewards_dict[i] for i in range(num_agents)], dtype=np.float32)
            next_agent_states = obs_to_agent_states(next_obs, num_agents, env.grid_size, max_steps)
            done_mask = np.full((num_agents,), float(done), dtype=np.float32)

            agent.store(agent_states, action_indices, rewards, next_agent_states, done_mask)
            agent.update(batch_size=16)

            if imageio is not None:
                img = render_obs_to_image(next_obs, size=env.grid_size)
                imageio.imwrite(str(out_dir / f"frame_maddpg_{frame_count:04d}.png"), img)
                frame_count += 1

            agent_states = next_agent_states
            last_rewards = rewards
            step += 1

        rewards_history.append(float(last_rewards.sum()))

        if (ep + 1) % 5 == 0:
            ckpt_path = checkpoints_dir / f"maddpg_ckpt_ep{ep + 1}.pt"
            agent.save(str(ckpt_path))

        print(
            f"Episode {ep}: frames={frame_count} "
            f"red_core={env.red_core_hp:.1f} blue_core={env.blue_core_hp:.1f}"
        )

    print("Done smoke run. Total frames:", frame_count)
    np.save(results_dir / "maddpg_rewards.npy", np.array(rewards_history, dtype=np.float32))

    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 3))
        plt.plot(rewards_history, marker="o")
        plt.title("MADDPG Training Reward over Episodes")
        plt.xlabel("Episode")
        plt.ylabel("Sum Reward")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(results_dir / "maddpg_rewards.png")
        plt.savefig(results_dir / "maddpg_rewards_plot.png")
        plt.close()
    except Exception as exc:
        print("Could not save rewards plot:", exc)

    try:
        if imageio is None or frame_count == 0:
            raise RuntimeError("Skipping video assembly because frame export is unavailable.")
        writer = imageio.get_writer(results_dir / "clash_demo.mp4", fps=10)
        for frame_idx in range(frame_count):
            frame = imageio.imread(out_dir / f"frame_maddpg_{frame_idx:04d}.png")
            writer.append_data(frame)
        writer.close()
        print("Video assembled to results/clash_demo.mp4")
    except Exception as exc:
        print("Video assembly failed:", exc)


if __name__ == "__main__":
    run(smoke_episodes=int(os.environ.get("EPISODES", "2")), max_steps=int(os.environ.get("MAX_STEPS", "60")))
