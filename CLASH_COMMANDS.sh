#!/bin/bash
# Clash of Titans - Quick Command Reference
# Copy this file or use individual commands below

# Enter the project directory
cd /home/india/Desktop/ctf-marl

# ============================================================================
# TRAINING COMMANDS
# ============================================================================

# Quick Test (5 episodes, ~0.2 sec)
# EPISODES=5 ./marl_env/bin/python -m experiments.train_clash

# Short Training (20 episodes, ~0.8 sec) - RECOMMENDED
# EPISODES=20 ./marl_env/bin/python -m experiments.train_clash

# Normal Training (50 episodes, ~2 sec)
# EPISODES=50 ./marl_env/bin/python -m experiments.train_clash

# Extended Training (100 episodes, ~4 sec)
# EPISODES=100 ./marl_env/bin/python -m experiments.train_clash

# ============================================================================
# VISUALIZATION COMMANDS
# ============================================================================

# View 3 demo games with agent behavior
# ./marl_env/bin/python -m experiments.visualize_clash

# ============================================================================
# ANALYSIS COMMANDS
# ============================================================================

# Analyze training results and generate plots
# ./marl_env/bin/python -m experiments.analyze_clash

# ============================================================================
# COMPLETE WORKFLOW (Copy all 3 commands)
# ============================================================================

# 1. Train for 30 episodes
EPISODES=30 ./marl_env/bin/python -m experiments.train_clash

# 2. Watch gameplay
./marl_env/bin/python -m experiments.visualize_clash

# 3. Analyze results
./marl_env/bin/python -m experiments.analyze_clash

# ============================================================================
# QUICK DIAGNOSTICS
# ============================================================================

# Check if files exist
echo "Checking Clash of Titans files..."
ls -la envs/clash_of_titans.py
ls -la experiments/train_clash.py
ls -la experiments/visualize_clash.py
ls -la experiments/analyze_clash.py
echo "✓ All files present"

# Check if venv works
echo "Checking Python environment..."
./marl_env/bin/python --version
./marl_env/bin/python -c "import numpy, torch, matplotlib; print('✓ All dependencies installed')"

# Run single episode test
echo "Running single episode test..."
EPISODES=1 ./marl_env/bin/python -m experiments.train_clash

# ============================================================================
# PERFORMANCE TUNING
# ============================================================================

# Faster games (smaller map, fewer agents)
# - Edit experiments/train_clash.py
# - Change: env = ClashOfTitansEnv(grid_size=15, team_size=2, max_steps=300)
# - Run: EPISODES=50 ./marl_env/bin/python -m experiments.train_clash

# Harder games (larger map, more agents)
# - Edit experiments/train_clash.py
# - Change: env = ClashOfTitansEnv(grid_size=25, team_size=4, max_steps=1000)
# - Run: EPISODES=100 ./marl_env/bin/python -m experiments.train_clash

# More competitive (reduce team bonus)
# - Edit experiments/train_clash.py
# - Change: cooperative_bonus_weight = 0.2  (from 0.5)
# - Run: EPISODES=50 ./marl_env/bin/python -m experiments.train_clash

# More cooperative (increase team bonus)
# - Edit experiments/train_clash.py
# - Change: cooperative_bonus_weight = 1.0  (from 0.5)
# - Run: EPISODES=50 ./marl_env/bin/python -m experiments.train_clash

# ============================================================================
# RESULTS VIEWING
# ============================================================================

# View training data
python3 << 'EOF'
import numpy as np
rewards = np.load('results/clash_rewards.npy')
tower_damage = np.load('results/clash_tower_damage.npy')
print(f"Episodes trained: {len(rewards)}")
print(f"Average score: {rewards.mean():.2f}")
print(f"Max score: {rewards.max():.2f}")
print(f"Average tower damage: {tower_damage.mean():.2f}")
EOF

# View the analysis plot
# Linux/Mac: open results/clash_analysis.png
# Or use: eog results/clash_analysis.png (Eye of GNOME)

# ============================================================================
# DOCUMENTATION
# ============================================================================

# Read the comprehensive guide
# cat CLASH_OF_TITANS_GUIDE.md

# Read quick reference
# cat CLASH_QUICK_REFERENCE.md

# Read implementation details
# cat CLASH_OF_TITANS_IMPLEMENTATION.md

# Read getting started guide
# cat README_CLASH_OF_TITANS.md

# ============================================================================
echo "✅ Clash of Titans Command Reference ready!"
echo ""
echo "Start with: EPISODES=20 ./marl_env/bin/python -m experiments.train_clash"
echo "Then view: ./marl_env/bin/python -m experiments.visualize_clash"
echo "Finally: ./marl_env/bin/python -m experiments.analyze_clash"
