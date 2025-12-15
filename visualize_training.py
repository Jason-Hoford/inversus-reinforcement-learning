"""
Visualize INVERSUS RL Training Progress

This script reads training log CSV files and generates comprehensive visualizations
showing how the agent's performance evolves over training.

Usage:
    python visualize_training.py <path_to_training_dir>
    
Example:
    python visualize_training.py runs/final_verification
"""

import os
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_training_data(log_dir):
    """Load training data from CSV file."""
    csv_path = os.path.join(log_dir, "training_log.csv")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Training log not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} training checkpoints from {csv_path}")
    return df


def create_visualizations(df, output_dir):
    """Create comprehensive training visualizations."""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Figure 1: Main Training Metrics (2x2 grid)
    fig1, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig1.suptitle('INVERSUS RL Training Progress', fontsize=16, fontweight='bold')
    
    # Plot 1: Win Rate over Steps
    axes[0, 0].plot(df['step'], df['win_rate'] * 100, linewidth=2, color='#2ecc71')
    axes[0, 0].fill_between(df['step'], 0, df['win_rate'] * 100, alpha=0.3, color='#2ecc71')
    axes[0, 0].set_xlabel('Training Steps', fontsize=12)
    axes[0, 0].set_ylabel('Win Rate (%)', fontsize=12)
    axes[0, 0].set_title('Win Rate vs Dummy Opponent', fontsize=13, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% (Random)')
    axes[0, 0].legend()
    
    # Plot 2: Average Reward over Steps
    axes[0, 1].plot(df['step'], df['avg_reward'], linewidth=2, color='#3498db')
    axes[0, 1].fill_between(df['step'], df['avg_reward'], 0, 
                            where=(df['avg_reward'] >= 0), alpha=0.3, color='#2ecc71', 
                            interpolate=True, label='Positive')
    axes[0, 1].fill_between(df['step'], df['avg_reward'], 0, 
                            where=(df['avg_reward'] < 0), alpha=0.3, color='#e74c3c', 
                            interpolate=True, label='Negative')
    axes[0, 1].set_xlabel('Training Steps', fontsize=12)
    axes[0, 1].set_ylabel('Average Reward', fontsize=12)
    axes[0, 1].set_title('Average Episode Reward', fontsize=13, fontweight='bold')
    axes[0, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # Plot 3: Average Episode Length over Steps
    axes[1, 0].plot(df['step'], df['avg_ep_len'], linewidth=2, color='#9b59b6')
    axes[1, 0].fill_between(df['step'], 0, df['avg_ep_len'], alpha=0.3, color='#9b59b6')
    axes[1, 0].set_xlabel('Training Steps', fontsize=12)
    axes[1, 0].set_ylabel('Average Episode Length', fontsize=12)
    axes[1, 0].set_title('Episode Length (Lower = Faster Wins)', fontsize=13, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Learning Metrics (Entropy)
    axes[1, 1].plot(df['step'], df['entropy'], linewidth=2, color='#e67e22', label='Entropy')
    axes[1, 1].set_xlabel('Training Steps', fontsize=12)
    axes[1, 1].set_ylabel('Policy Entropy', fontsize=12)
    axes[1, 1].set_title('Exploration (Entropy)', fontsize=13, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    plt.tight_layout()
    
    # Save main figure
    main_path = os.path.join(output_dir, 'training_overview.png')
    plt.savefig(main_path, dpi=300, bbox_inches='tight')
    print(f"[DONE] Saved main overview: {main_path}")
    
    # Figure 2: Loss Curves
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    fig2.suptitle('Training Loss Curves', fontsize=16, fontweight='bold')
    
    # Policy Loss
    axes2[0].plot(df['step'], df['policy_loss'], linewidth=2, color='#e74c3c')
    axes2[0].set_xlabel('Training Steps', fontsize=12)
    axes2[0].set_ylabel('Policy Loss', fontsize=12)
    axes2[0].set_title('Policy Loss (Lower = Better)', fontsize=13, fontweight='bold')
    axes2[0].grid(True, alpha=0.3)
    
    # Value Loss
    axes2[1].plot(df['step'], df['value_loss'], linewidth=2, color='#3498db')
    axes2[1].set_xlabel('Training Steps', fontsize=12)
    axes2[1].set_ylabel('Value Loss', fontsize=12)
    axes2[1].set_title('Value Loss (Lower = Better)', fontsize=13, fontweight='bold')
    axes2[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save loss figure
    loss_path = os.path.join(output_dir, 'training_losses.png')
    plt.savefig(loss_path, dpi=300, bbox_inches='tight')
    print(f"[DONE] Saved loss curves: {loss_path}")
    
    # Figure 3: Performance Summary
    fig3, ax = plt.subplots(figsize=(10, 6))
    
    # Create summary table
    final_metrics = df.iloc[-1]
    summary_text = f"""
    TRAINING SUMMARY
    ════════════════════════════════════════
    
    Total Training Steps:     {int(final_metrics['step']):,}
    Total Episodes:           {int(final_metrics['episode']):,}
    
    FINAL PERFORMANCE:
    ────────────────────────────────────────
    Win Rate:                 {final_metrics['win_rate']*100:.1f}%
    Average Reward:           {final_metrics['avg_reward']:.3f}
    Average Episode Length:   {final_metrics['avg_ep_len']:.1f}
    
    LEARNING PROGRESS:
    ────────────────────────────────────────
    Initial Win Rate:         {df.iloc[0]['win_rate']*100:.1f}%
    Final Win Rate:           {final_metrics['win_rate']*100:.1f}%
    Improvement:              {(final_metrics['win_rate'] - df.iloc[0]['win_rate'])*100:+.1f}%
    
    Initial Avg Reward:       {df.iloc[0]['avg_reward']:.3f}
    Final Avg Reward:         {final_metrics['avg_reward']:.3f}
    Improvement:              {final_metrics['avg_reward'] - df.iloc[0]['avg_reward']:+.3f}
    
    FINAL LOSSES:
    ────────────────────────────────────────
    Policy Loss:              {final_metrics['policy_loss']:.6f}
    Value Loss:               {final_metrics['value_loss']:.6f}
    Entropy:                  {final_metrics['entropy']:.6f}
    """
    
    ax.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
            verticalalignment='center', transform=ax.transAxes)
    ax.axis('off')
    
    plt.tight_layout()
    
    # Save summary
    summary_path = os.path.join(output_dir, 'training_summary.png')
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    print(f"[DONE] Saved summary: {summary_path}")
    
    # Print summary to console
    print("\n" + "="*50)
    print("TRAINING RESULTS SUMMARY")
    print("="*50)
    print(f"Final Win Rate: {final_metrics['win_rate']*100:.1f}%")
    print(f"Final Avg Reward: {final_metrics['avg_reward']:.3f}")
    print(f"Episodes Completed: {int(final_metrics['episode']):,}")
    print(f"Improvement: {(final_metrics['win_rate'] - df.iloc[0]['win_rate'])*100:+.1f}% win rate")
    print("="*50 + "\n")
    
    return fig1, fig2, fig3


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Visualize INVERSUS RL training progress",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python visualize_training.py runs/final_verification
  python visualize_training.py runs/my_training --output plots
        """
    )
    parser.add_argument('log_dir', type=str,
                       help='Directory containing training_log.csv')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output directory for plots (default: <log_dir>/visualizations)')
    parser.add_argument('--show', action='store_true',
                       help='Display plots interactively')
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.isdir(args.log_dir):
        print(f"Error: Directory not found: {args.log_dir}")
        sys.exit(1)
    
    # Determine output directory
    if args.output:
        output_dir = args.output
    else:
        output_dir = os.path.join(args.log_dir, 'visualizations')
    
    print(f"\n{'='*60}")
    print(f"INVERSUS RL Training Visualization")
    print(f"{'='*60}\n")
    
    # Load data
    try:
        df = load_training_data(args.log_dir)
    except Exception as e:
        print(f"Error loading training data: {e}")
        sys.exit(1)
    
    # Create visualizations
    print(f"\nGenerating visualizations...")
    try:
        create_visualizations(df, output_dir)
    except Exception as e:
        print(f"Error creating visualizations: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print(f"\n[DONE] All visualizations saved to: {output_dir}\n")
    
    # Show plots if requested
    if args.show:
        print("Displaying plots...")
        plt.show()
    else:
        print("Tip: Use --show flag to display plots interactively")


if __name__ == "__main__":
    main()
