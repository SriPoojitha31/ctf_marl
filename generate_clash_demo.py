import os
from pathlib import Path

import imageio
import numpy as np

from envs.moba_ctf_env import MobaCTFEnv
from experiments.moba_render import render_moba_frame


def _lerp(a, b, t):
    return a + (b - a) * t


def _interp_pos(p1, p2, t):
    x = int(round(_lerp(p1[0], p2[0], t)))
    y = int(round(_lerp(p1[1], p2[1], t)))
    return (x, y)


def _interp_dict(d1, d2, t):
    return {k: _interp_pos(d1[k], d2[k], t) for k in d1}


def _interp_hp(h1, h2, t):
    return {k: float(_lerp(h1[k], h2[k], t)) for k in h1}


def _make_towers(base_towers, hp_map):
    towers = []
    for tower in base_towers:
        key = (tower["team"], tower["lane"], tower["tier"])
        item = dict(tower)
        item["hp"] = hp_map.get(key, tower["max_hp"])
        towers.append(item)
    return towers


def _phase(progress, start, end):
    if progress <= start:
        return 0.0
    if progress >= end:
        return 1.0
    return (progress - start) / (end - start)


def snapshot(base_obs, progress):
    start_pos = {
        0: (1, 3), 1: (1, 7), 2: (1, 11), 3: (2, 4), 4: (2, 10),
        5: (13, 3), 6: (13, 7), 7: (13, 11), 8: (12, 4), 9: (12, 10),
    }
    clash_pos = {
        0: (5, 3), 1: (6, 7), 2: (6, 11), 3: (5, 7), 4: (4, 10),
        5: (9, 3), 6: (8, 7), 7: (8, 11), 8: (9, 7), 9: (10, 10),
    }
    push_pos = {
        0: (8, 3), 1: (9, 7), 2: (9, 11), 3: (8, 7), 4: (7, 10),
        5: (11, 3), 6: (6, 7), 7: (11, 11), 8: (11, 7), 9: (12, 10),
    }
    final_pos = {
        0: (11, 3), 1: (12, 7), 2: (11, 11), 3: (11, 7), 4: (10, 10),
        5: (13, 3), 6: (3, 7), 7: (13, 11), 8: (12, 7), 9: (12, 10),
    }

    start_hp = {i: 180.0 if i in (0, 5) else 130.0 if i in (1, 6) else 95.0 if i in (2, 7) else 90.0 if i in (3, 8) else 110.0 for i in range(10)}
    clash_hp = {**start_hp, 0: 120.0, 1: 95.0, 3: 60.0, 5: 125.0, 6: 85.0, 8: 55.0}
    push_hp = {**start_hp, 0: 90.0, 1: 110.0, 2: 70.0, 3: 40.0, 4: 90.0, 5: 60.0, 6: 30.0, 7: 95.0, 8: 35.0, 9: 70.0}
    final_hp = {**start_hp, 0: 60.0, 1: 75.0, 2: 55.0, 3: 30.0, 4: 85.0, 5: 0.0, 6: 0.0, 7: 35.0, 8: 0.0, 9: 45.0}

    base_towers = base_obs["towers"]
    clash_towers = {key: 20.0 for key in [(t["team"], t["lane"], t["tier"]) for t in base_towers]}
    push_towers = dict(clash_towers)
    push_towers[("blue", "top", "outer")] = 0.0
    push_towers[("blue", "mid", "outer")] = 0.0
    push_towers[("red", "mid", "outer")] = 0.0
    final_towers = dict(push_towers)
    final_towers[("blue", "top", "inner")] = 0.0
    final_towers[("blue", "mid", "inner")] = 0.0

    if progress < 0.25:
        t = _phase(progress, 0.0, 0.25)
        positions = _interp_dict(start_pos, clash_pos, t)
        health = _interp_hp(start_hp, clash_hp, t)
        respawn = {}
        red_core_hp = 80.0
        blue_core_hp = 80.0
        red_score = 0
        blue_score = 0
        towers = _make_towers(base_towers, clash_towers)
    elif progress < 0.7:
        t = _phase(progress, 0.25, 0.7)
        positions = _interp_dict(clash_pos, push_pos, t)
        health = _interp_hp(clash_hp, push_hp, t)
        respawn = {8: 1} if 0.32 < progress < 0.42 else ({6: 1} if 0.48 < progress < 0.56 else {})
        red_core_hp = 80.0
        blue_core_hp = 80.0
        red_score = 0
        blue_score = 0
        towers = _make_towers(base_towers, push_towers if progress > 0.5 else clash_towers)
    else:
        t = _phase(progress, 0.7, 1.0)
        positions = _interp_dict(push_pos, final_pos, t)
        health = _interp_hp(push_hp, final_hp, t)
        respawn = {5: 2, 8: 2} if progress < 0.82 else ({6: 2} if progress < 0.9 else {})
        red_core_hp = 80.0
        blue_core_hp = _lerp(80.0, 0.0, t)
        red_score = 1 if progress > 0.96 else 0
        blue_score = 0
        towers = _make_towers(base_towers, final_towers if progress > 0.82 else push_towers)

    obs = dict(base_obs)
    obs["agent_positions"] = positions
    obs["agent_health"] = health
    obs["respawn_timers"] = respawn
    obs["red_core_hp"] = red_core_hp
    obs["blue_core_hp"] = blue_core_hp
    obs["red_score"] = red_score
    obs["blue_score"] = blue_score
    obs["towers"] = towers
    return obs


def main():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    results_dir = Path("results")
    frames_dir = results_dir / "demo_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    env = MobaCTFEnv(grid_size=15, team_size=5, max_steps=140)
    env.static_obstacles = {
        (5, 5), (5, 9), (9, 5), (9, 9),
        (6, 5), (6, 9), (8, 5), (8, 9),
    }
    env.dynamic_obstacles = []
    base_obs = env.reset()
    for tower in base_obs["towers"]:
        tower["hp"] = 20.0
        tower["max_hp"] = 20.0

    total_steps = 140
    frames = []
    best_fight_frame = None
    best_fight_value = -1

    for frame_idx in range(total_steps):
        progress = frame_idx / (total_steps - 1)
        obs = snapshot(base_obs, progress)
        obs["step"] = frame_idx
        frame = render_moba_frame(obs, env.grid_size)

        fight_value = 0
        for rid in obs["red_team"]:
            for bid in obs["blue_team"]:
                rx, ry = obs["agent_positions"][rid]
                bx, by = obs["agent_positions"][bid]
                if abs(rx - bx) + abs(ry - by) <= 3:
                    fight_value += 1
        if fight_value > best_fight_value:
            best_fight_value = fight_value
            best_fight_frame = frame

        imageio.imwrite(frames_dir / f"clash_demo_{frame_idx:04d}.png", frame)
        frames.append(frame)

    opening_frame = frames[0]
    final_frame = frames[-1]

    hold_open = [opening_frame] * 15
    hold_final = [final_frame] * 20
    export_frames = hold_open + frames + hold_final

    imageio.mimsave(results_dir / "clash_demo.gif", export_frames, fps=6)
    writer = imageio.get_writer(results_dir / "clash_demo.mp4", fps=6)
    for frame in export_frames:
        writer.append_data(frame)
    writer.close()

    imageio.imwrite(results_dir / "clash_demo_cover.png", final_frame)
    imageio.imwrite(results_dir / "clash_demo_opening.png", opening_frame)
    imageio.imwrite(results_dir / "clash_demo_fight.png", best_fight_frame if best_fight_frame is not None else frames[len(frames) // 2])
    imageio.imwrite(results_dir / "clash_demo_final.png", final_frame)

    print(f"Saved {len(export_frames)} demo frames to {frames_dir}")
    print("Saved results/clash_demo.gif")
    print("Saved results/clash_demo.mp4")
    print("Saved results/clash_demo_cover.png")
    print("Saved results/clash_demo_opening.png")
    print("Saved results/clash_demo_fight.png")
    print("Saved results/clash_demo_final.png")


if __name__ == "__main__":
    main()
