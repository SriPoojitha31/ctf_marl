# CTF-MARL: Multi-Agent Reinforcement Learning for Grid-Based Team Games

This repository studies competitive and cooperative multi-agent reinforcement learning with capture-the-flag and MOBA-inspired environments. It includes baseline IQL training, a MADDPG pipeline, analysis scripts, visual outputs, and paper-facing result artifacts saved under `results/`.

## Environments

- `envs/ctf_env.py` - Enhanced capture-the-flag environment with roles, towers, obstacles, health, and respawn
- `envs/moba_ctf_env.py` - Simplified 5v5 MOBA-style environment with lanes, towers, cores, combat, and shaping rewards
- `envs/clash_of_titans.py` - Hybrid escort and tower mode used by the Clash analysis scripts

## Algorithms

- `algorithms/iql.py` - Tabular IQL agent supporting both positional and vector observations
- `algorithms/maddpg.py` - Centralized-critic MADDPG for discrete multi-agent control

## Main Commands

Use the provided virtual environment:

```bash
./marl_env/bin/python test_env.py
EPISODES=5 MAX_STEPS=100 ./marl_env/bin/python -m experiments.train_iql
EPISODES=5 MAX_STEPS=60 ./marl_env/bin/python -m experiments.train_maddpg
./marl_env/bin/python experiments/plot_iql.py
./marl_env/bin/python experiments/plot_maddpg.py
./marl_env/bin/python -m experiments.analyze_clash
./marl_env/bin/python -m experiments.generate_submission_summary
```

## Saved Outputs

The project writes its paper-facing artifacts to `results/`:

- `results/iql_rewards.npy`
- `results/iql_rewards_plot.png`
- `results/maddpg_rewards.npy`
- `results/maddpg_rewards_plot.png`
- `results/maddpg_rewards.png`
- `results/checkpoints/`
- `results/frames/` and `results/iql_frames/`
- `results/clash_rewards.npy`
- `results/clash_tower_damage.npy`
- `results/clash_analysis.png`
- `results/submission_metrics.json`
- `results/submission_summary.md`

## Notes

- The training and plotting scripts are written to work in headless environments.
- `ffmpeg` video assembly may fail on systems with restricted `snap/apparmor` policies; the raw frames are still saved.
- The repository already contains several walkthrough documents for the Clash and CTF variants if you need more narrative detail for a report.
