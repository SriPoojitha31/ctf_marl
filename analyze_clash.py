"""
Clash of Titans - Learning Analysis

Analyzes training results to show:
- Score progression over episodes
- Team performance comparison
- Tower damage scaling
- Strategy emergence
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def analyze_clash_training():
    """Analyze and plot Clash of Titans training results."""
    
    # Load results
    rewards_file = Path('results/clash_rewards.npy')
    tower_damage_file = Path('results/clash_tower_damage.npy')
    
    if not rewards_file.exists() or not tower_damage_file.exists():
        print("Error: Training results not found!")
        print("Run: EPISODES=30 python -m experiments.train_clash")
        return
    
    rewards = np.load(rewards_file)
    tower_damage = np.load(tower_damage_file)
    
    print("\n" + "="*70)
    print("CLASH OF TITANS - TRAINING ANALYSIS")
    print("="*70)
    
    # Basic Statistics
    print(f"\nDataset Size: {len(rewards)} episodes")
    print(f"Reward Range: [{rewards.min():.1f}, {rewards.max():.1f}]")
    print(f"Tower Damage Range: [{tower_damage.min():.1f}, {tower_damage.max():.1f}]")
    
    # Progression Analysis
    print("\n" + "-"*70)
    print("LEARNING PROGRESSION")
    print("-"*70)
    
    if len(rewards) >= 10:
        early = rewards[:5]
        mid = rewards[5:len(rewards)//2] if len(rewards) > 10 else rewards[5:]
        late = rewards[len(rewards)//2:] if len(rewards) > 10 else rewards[5:]
        
        print(f"\nEarly Phase (Ep 1-5):")
        print(f"  Avg Score: {early.mean():.2f} ± {early.std():.2f}")
        print(f"  Avg Tower Damage: {tower_damage[:5].mean():.2f}")
        print(f"  Max Score: {early.max():.2f}")
        
        if len(mid) > 0:
            print(f"\nMid Phase (Ep 6-{len(rewards)//2}):")
            print(f"  Avg Score: {mid.mean():.2f} ± {mid.std():.2f}")
            print(f"  Avg Tower Damage: {tower_damage[5:len(rewards)//2].mean():.2f}")
            print(f"  Max Score: {mid.max():.2f}")
        
        print(f"\nLate Phase (Ep {len(rewards)//2+1}-{len(rewards)}):")
        print(f"  Avg Score: {late.mean():.2f} ± {late.std():.2f}")
        print(f"  Avg Tower Damage: {tower_damage[len(rewards)//2:].mean():.2f}")
        print(f"  Max Score: {late.max():.2f}")
    
    # Training Metrics
    print("\n" + "-"*70)
    print("OVERALL METRICS")
    print("-"*70)
    print(f"Average Episode Score: {rewards.mean():.2f}")
    print(f"Score Std Dev: {rewards.std():.2f}")
    print(f"Average Tower Damage: {tower_damage.mean():.2f}")
    print(f"Total Tower Damage: {tower_damage.sum():.0f}")
    
    # Improvement Rate
    if len(rewards) > 1:
        improvement = (rewards[-1] - rewards[0])
        print(f"Score Improvement: {improvement:+.2f} (from first to last episode)")
    
    # Plot Results
    print("\n" + "-"*70)
    print("GENERATING PLOTS...")
    print("-"*70)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Clash of Titans - Training Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Score over Episodes
    ax1 = axes[0, 0]
    episodes = np.arange(1, len(rewards) + 1)
    ax1.plot(episodes, rewards, 'b-', linewidth=1, label='Episode Score', alpha=0.7)
    
    # Add moving average
    if len(rewards) > 5:
        window = max(1, len(rewards) // 10)
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax1.plot(np.arange(window, len(rewards) + 1), moving_avg, 'r--', 
                linewidth=2, label=f'Moving Avg (window={window})')
    
    ax1.set_xlabel('Episode', fontweight='bold')
    ax1.set_ylabel('Total Score', fontweight='bold')
    ax1.set_title('Score Progression')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Tower Damage over Episodes
    ax2 = axes[0, 1]
    ax2.plot(episodes, tower_damage, 'g-', linewidth=1, label='Tower Damage', alpha=0.7)
    
    if len(tower_damage) > 5:
        window = max(1, len(tower_damage) // 10)
        moving_avg = np.convolve(tower_damage, np.ones(window)/window, mode='valid')
        ax2.plot(np.arange(window, len(tower_damage) + 1), moving_avg, 'r--',
                linewidth=2, label=f'Moving Avg (window={window})')
    
    ax2.set_xlabel('Episode', fontweight='bold')
    ax2.set_ylabel('Total Tower Damage', fontweight='bold')
    ax2.set_title('Tower Damage Progression')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Score Distribution (Histogram)
    ax3 = axes[1, 0]
    ax3.hist(rewards, bins=20, color='blue', alpha=0.7, edgecolor='black')
    ax3.axvline(rewards.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {rewards.mean():.1f}')
    ax3.set_xlabel('Score', fontweight='bold')
    ax3.set_ylabel('Frequency', fontweight='bold')
    ax3.set_title('Score Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Correlation between Score and Tower Damage
    ax4 = axes[1, 1]
    ax4.scatter(tower_damage, rewards, alpha=0.6, s=50, color='purple')
    
    # Add trend line
    if len(tower_damage) > 1:
        z = np.polyfit(tower_damage, rewards, 1)
        p = np.poly1d(z)
        ax4.plot(tower_damage, p(tower_damage), "r--", linewidth=2, label='Trend')
    
    correlation = np.corrcoef(tower_damage, rewards)[0, 1]
    ax4.set_xlabel('Tower Damage', fontweight='bold')
    ax4.set_ylabel('Total Score', fontweight='bold')
    ax4.set_title(f'Score vs Tower Damage (r={correlation:.2f})')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_file = 'results/clash_analysis.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"✓ Plot saved to {plot_file}")
    
    # Additional Analysis
    print("\n" + "-"*70)
    print("STRATEGY INSIGHTS")
    print("-"*70)
    
    # Check if tower damage correlates with score
    high_damage_mask = tower_damage > tower_damage.mean()
    if high_damage_mask.sum() > 0:
        high_damage_scores = rewards[high_damage_mask].mean()
        low_damage_scores = rewards[~high_damage_mask].mean()
        
        print(f"\nTower Offense Effectiveness:")
        print(f"  High Damage Episodes (>{tower_damage.mean():.1f}): Avg Score = {high_damage_scores:.2f}")
        print(f"  Low Damage Episodes (<{tower_damage.mean():.1f}): Avg Score = {low_damage_scores:.2f}")
        print(f"  Difference: {abs(high_damage_scores - low_damage_scores):+.2f}")
    
    # Estimate learning rate
    if len(rewards) > 2:
        first_half_avg = rewards[:len(rewards)//2].mean()
        second_half_avg = rewards[len(rewards)//2:].mean()
        improvement_pct = ((second_half_avg - first_half_avg) / first_half_avg * 100) if first_half_avg != 0 else 0
        
        print(f"\nLearning Rate:")
        print(f"  First Half Avg: {first_half_avg:.2f}")
        print(f"  Second Half Avg: {second_half_avg:.2f}")
        print(f"  Improvement: {improvement_pct:+.1f}%")
    
    print("\n" + "="*70)
    print("Analysis complete! View results/clash_analysis.png for visualizations.")
    print("="*70 + "\n")


if __name__ == "__main__":
    analyze_clash_training()
