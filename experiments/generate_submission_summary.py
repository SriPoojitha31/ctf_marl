import json
from pathlib import Path

import numpy as np


RESULTS_DIR = Path("results")


def load_array(path: Path):
    if path.exists():
        return np.load(path)
    return None


def summarize_array(name: str, arr):
    if arr is None or len(arr) == 0:
        return None
    return {
        "name": name,
        "episodes": int(len(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "last": float(arr[-1]),
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    iql = summarize_array("iql_rewards", load_array(RESULTS_DIR / "iql_rewards.npy"))
    maddpg = summarize_array("maddpg_rewards", load_array(RESULTS_DIR / "maddpg_rewards.npy"))
    clash_rewards = summarize_array("clash_rewards", load_array(RESULTS_DIR / "clash_rewards.npy"))
    clash_tower = summarize_array("clash_tower_damage", load_array(RESULTS_DIR / "clash_tower_damage.npy"))

    metrics = {
        "available_outputs": {
            "iql_rewards": iql is not None,
            "maddpg_rewards": maddpg is not None,
            "clash_rewards": clash_rewards is not None,
            "clash_tower_damage": clash_tower is not None,
            "clash_analysis_png": (RESULTS_DIR / "clash_analysis.png").exists(),
            "cooperation_comparison_png": (RESULTS_DIR / "cooperation_comparison.png").exists(),
            "maddpg_plot_png": (RESULTS_DIR / "maddpg_rewards_plot.png").exists(),
            "iql_plot_png": (RESULTS_DIR / "iql_rewards_plot.png").exists(),
        },
        "metrics": {
            "iql_rewards": iql,
            "maddpg_rewards": maddpg,
            "clash_rewards": clash_rewards,
            "clash_tower_damage": clash_tower,
        },
    }

    with open(RESULTS_DIR / "submission_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    lines = [
        "# Submission Summary",
        "",
        "This file summarizes the currently generated experimental outputs for paper/report submission.",
        "",
    ]

    for section in [iql, maddpg, clash_rewards, clash_tower]:
        if section is None:
            continue
        lines.extend(
            [
                f"## {section['name']}",
                f"- episodes: {section['episodes']}",
                f"- mean: {section['mean']:.3f}",
                f"- std: {section['std']:.3f}",
                f"- min: {section['min']:.3f}",
                f"- max: {section['max']:.3f}",
                f"- last: {section['last']:.3f}",
                "",
            ]
        )

    lines.extend(
        [
            "## Output Files",
            "",
            f"- iql_rewards_plot.png: {(RESULTS_DIR / 'iql_rewards_plot.png').exists()}",
            f"- maddpg_rewards_plot.png: {(RESULTS_DIR / 'maddpg_rewards_plot.png').exists()}",
            f"- clash_analysis.png: {(RESULTS_DIR / 'clash_analysis.png').exists()}",
            f"- cooperation_comparison.png: {(RESULTS_DIR / 'cooperation_comparison.png').exists()}",
            "",
        ]
    )

    (RESULTS_DIR / "submission_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("Saved results/submission_metrics.json")
    print("Saved results/submission_summary.md")


if __name__ == "__main__":
    main()
