"""
Record Clash of Titans episodes and save as images and (optionally) MP4.

Usage:
  ./marl_env/bin/python -m experiments.record_clash --episodes 1 --steps 300

The script will:
- Run `episodes` episodes
- Save frames to `results/frames/` (png)
- Try to assemble `results/clash_demo.mp4` using imageio (if available)

Note: If ffmpeg is installed, imageio will use it. If not, you'll have PNG frames you can stitch with ffmpeg:
  ffmpeg -framerate 12 -i results/frames/frame_%04d.png -c:v libx264 -pix_fmt yuv420p results/clash_demo.mp4
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from envs.clash_of_titans import ClashOfTitansEnv


def draw_grid(env):
    size = env.grid_size
    grid = np.zeros((size, size, 3), dtype=np.uint8) + 255

    # Draw towers
    rx, ry = env.red_tower_base
    bx, by = env.blue_tower_base

    def paint(x, y, color):
        if 0 <= x < size and 0 <= y < size:
            grid[y, x] = color

    # color: red tower (light red), blue tower (light blue)
    paint(rx, ry, np.array([255, 120, 120], dtype=np.uint8))
    paint(bx, by, np.array([120, 160, 255], dtype=np.uint8))

    # Draw titans
    if env.red_titan:
        tx, ty = int(env.red_titan.pos[0]), int(env.red_titan.pos[1])
        paint(tx, ty, np.array([200, 40, 40], dtype=np.uint8))
    if env.blue_titan:
        tx, ty = int(env.blue_titan.pos[0]), int(env.blue_titan.pos[1])
        paint(tx, ty, np.array([40, 80, 200], dtype=np.uint8))

    # Draw agents (red and blue)
    for aid, pos in env.agent_positions.items():
        x, y = int(pos[0]), int(pos[1])
        if aid in env.red_team:
            paint(x, y, np.array([220, 60, 60], dtype=np.uint8))
        else:
            paint(x, y, np.array([80, 120, 220], dtype=np.uint8))

    return grid


def record_episode(env, out_dir, episode_idx=0, max_steps=500):
    os.makedirs(out_dir, exist_ok=True)
    obs = env.reset()
    done = False
    step = 0
    saved = 0

    while not done and step < max_steps:
        # Simple random policy (replace with trained agents if available)
        actions = {}
        for agent_id in range(env.num_agents):
            a = np.random.randint(0, 4)
            # map to directions expected by env (strings)
            actions[agent_id] = {0: 'up', 1: 'down', 2: 'left', 3: 'right'}[a]

        next_obs, rewards, done, info = env.step(actions)

        img = draw_grid(env)
        fname = os.path.join(out_dir, f"frame_{episode_idx:02d}_{saved:04d}.png")
        plt.imsave(fname, img)
        saved += 1

        step += 1
        obs = next_obs

    return saved


def assemble_video(frame_pattern, out_video, fps=12):
    try:
        import imageio
        import imageio_ffmpeg
        writer = imageio.get_writer(out_video, fps=fps)
        import glob
        files = sorted(glob.glob(frame_pattern))
        for f in files:
            img = imageio.imread(f)
            writer.append_data(img)
        writer.close()
        return True
    except Exception as e:
        print("Could not assemble mp4 via imageio/ffmpeg:", e)
        print("You can assemble with ffmpeg manually:")
        print("  ffmpeg -framerate 12 -i results/frames/frame_%02d_%%04d.png -c:v libx264 -pix_fmt yuv420p results/clash_demo.mp4")
        return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=1)
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--out', type=str, default='results/frames')
    parser.add_argument('--video', type=str, default='results/clash_demo.mp4')
    args = parser.parse_args()

    env = ClashOfTitansEnv(grid_size=20, team_size=3, max_steps=args.steps)
    total_saved = 0
    for ep in range(args.episodes):
        saved = record_episode(env, args.out, episode_idx=ep, max_steps=args.steps)
        total_saved += saved
        print(f"Episode {ep} saved frames: {saved}")

    print(f"Total frames saved: {total_saved}")

    pattern = os.path.join(args.out, 'frame_*.png')
    ok = assemble_video(pattern, args.video, fps=12)
    if ok:
        print(f"Video written to {args.video}")
    else:
        print("Video assembly not completed. Frames are available in results/frames/")
