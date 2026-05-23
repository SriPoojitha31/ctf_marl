import os

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

try:
    import matplotlib.pyplot as plt
    has_matplotlib = True
except ImportError:
    has_matplotlib = False


try:
    rewards = np.load("results/iql_rewards.npy")

    if has_matplotlib:
        plt.figure(figsize=(10, 6))
        plt.plot(rewards)
        plt.title("IQL Training Rewards")
        plt.xlabel("Training Episode (X-axis)")
        plt.ylabel("Reward (Y-axis)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("results/iql_rewards_plot.png")
        plt.close()
        print("Saved results/iql_rewards_plot.png")
    else:
        print("Matplotlib not installed. Saving plot to PNG skipped.")
        print(f"Mean reward: {np.mean(rewards):.2f}, Max reward: {np.max(rewards):.2f}")
except FileNotFoundError:
    print("No results file found. Run train_iql.py first.")
