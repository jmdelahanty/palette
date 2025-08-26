#!/usr/bin/env python3
"""
Group-Based Trial Preference Plotter

Plots trial-by-trial preferences with fish grouped into two categories:
- Group 1: Fish IDs 0-5
- Group 2: Fish IDs 6-11

Each trial shows all fish preferences as points, color-coded by group.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path
from scipy import stats
from rich.console import Console

console = Console()
sns.set_style("whitegrid")

def load_and_group_data(csv_path):
    """Load preference data and assign groups."""
    df = pd.read_csv(csv_path)
    
    # Assign groups based on fish ID
    df['group'] = df['roi_id'].apply(lambda x: 1 if x <= 5 else 2)
    
    console.print(f"[green]Loaded {len(df)} records[/green]")
    console.print(f"[cyan]Group 1 (IDs 0-5): {df[df['group']==1]['roi_id'].nunique()} fish[/cyan]")
    console.print(f"[yellow]Group 2 (IDs 6-11): {df[df['group']==2]['roi_id'].nunique()} fish[/yellow]")
    
    return df

def plot_trials_by_group(df, save_dir=None):
    """Plot each trial with all fish preferences, color-coded by group."""
    
    # Get unique trials
    trials = sorted(df['trial_number'].unique())
    n_trials = len(trials)
    
    # Set up figure with subplots for each trial
    n_cols = min(5, n_trials)
    n_rows = int(np.ceil(n_trials / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    if n_trials == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]
    
    # Colors for groups
    group_colors = {1: '#2E86AB', 2: '#A23B72'}  # Blue for group 1, Purple for group 2
    
    for idx, trial_num in enumerate(trials):
        ax = axes[idx]
        trial_df = df[df['trial_number'] == trial_num]
        
        # Plot each group
        for group in [1, 2]:
            group_data = trial_df[trial_df['group'] == group]
            
            # Add jitter for better visibility
            x_positions = np.random.normal(group, 0.1, size=len(group_data))
            
            ax.scatter(x_positions, group_data['top_proportion'], 
                      color=group_colors[group], alpha=0.6, s=100,
                      edgecolor='black', linewidth=1,
                      label=f'Group {group}')
            
            # Add group mean as horizontal line
            group_mean = group_data['top_proportion'].mean()
            ax.hlines(group_mean, group - 0.3, group + 0.3, 
                     color=group_colors[group], linewidth=3, alpha=0.8)
        
        # Add 50% reference line
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        
        # Formatting
        ax.set_xlim([0.5, 2.5])
        ax.set_ylim([0, 1])
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Group 1\n(IDs 0-5)', 'Group 2\n(IDs 6-11)'])
        ax.set_ylabel('Top Proportion')
        ax.set_title(f'Trial {trial_num}', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add legend to first plot
        if idx == 0:
            ax.legend(loc='upper right')
        
        # Add stimulus type if available
        if 'trial_type' in trial_df.columns:
            stim_type = trial_df['trial_type'].iloc[0]
            ax.text(0.5, 0.95, f'({stim_type})', transform=ax.transAxes,
                   ha='center', fontsize=9, style='italic')
    
    # Hide unused subplots
    for idx in range(n_trials, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Trial-by-Trial Preferences by Group', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_dir:
        save_path = Path(save_dir) / 'trials_by_group.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        console.print(f"[green]✓ Saved trials plot to {save_path}[/green]")
    
    plt.show()

def plot_group_comparison_summary(df, save_dir=None):
    """Create summary plots comparing the two groups."""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Colors for groups
    group_colors = {1: '#2E86AB', 2: '#A23B72'}
    
    # 1. Overall distribution by group
    ax = axes[0, 0]
    for group in [1, 2]:
        group_data = df[df['group'] == group]['top_proportion']
        ax.hist(group_data, bins=20, alpha=0.5, color=group_colors[group],
               edgecolor='black', linewidth=1, label=f'Group {group}')
    ax.axvline(x=0.5, color='black', linestyle='--', alpha=0.5, linewidth=2)
    ax.set_xlabel('Top Proportion')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Preferences', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Box plot comparison
    ax = axes[0, 1]
    group_data_list = [df[df['group'] == g]['top_proportion'].values for g in [1, 2]]
    bp = ax.boxplot(group_data_list, patch_artist=True, labels=['Group 1', 'Group 2'])
    for patch, group in zip(bp['boxes'], [1, 2]):
        patch.set_facecolor(group_colors[group])
        patch.set_alpha(0.7)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.set_ylabel('Top Proportion')
    ax.set_title('Group Comparison', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add statistical test
    statistic, p_value = stats.ttest_ind(group_data_list[0], group_data_list[1])
    ax.text(0.5, 0.95, f'p = {p_value:.4f}', transform=ax.transAxes,
           ha='center', va='top', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 3. Individual fish means by group
    ax = axes[0, 2]
    fish_means = df.groupby(['group', 'roi_id'])['top_proportion'].mean().reset_index()
    
    for group in [1, 2]:
        group_fish = fish_means[fish_means['group'] == group]
        x_pos = [group] * len(group_fish)
        x_jittered = np.random.normal(x_pos, 0.05)
        ax.scatter(x_jittered, group_fish['top_proportion'], 
                  color=group_colors[group], s=150, alpha=0.7,
                  edgecolor='black', linewidth=1)
        
        # Add group mean
        group_mean = group_fish['top_proportion'].mean()
        ax.hlines(group_mean, group - 0.2, group + 0.2,
                 color='black', linewidth=3, alpha=0.8)
    
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.set_xlim([0.5, 2.5])
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Group 1', 'Group 2'])
    ax.set_ylabel('Mean Top Proportion')
    ax.set_title('Individual Fish Means', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 4. Preferences over trials
    ax = axes[1, 0]
    trial_means = df.groupby(['trial_number', 'group'])['top_proportion'].mean().reset_index()
    
    for group in [1, 2]:
        group_trials = trial_means[trial_means['group'] == group]
        ax.plot(group_trials['trial_number'], group_trials['top_proportion'],
               marker='o', color=group_colors[group], linewidth=2, 
               markersize=8, alpha=0.8, label=f'Group {group}')
    
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.set_xlabel('Trial Number')
    ax.set_ylabel('Mean Top Proportion')
    ax.set_title('Group Preferences Over Trials', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. Preference by stimulus type (if available)
    ax = axes[1, 1]
    if 'trial_type' in df.columns:
        stim_means = df.groupby(['trial_type', 'group'])['top_proportion'].mean().reset_index()
        
        stim_types = stim_means['trial_type'].unique()
        x_positions = np.arange(len(stim_types))
        width = 0.35
        
        for i, group in enumerate([1, 2]):
            group_stim = stim_means[stim_means['group'] == group]
            offsets = x_positions + (i - 0.5) * width
            ax.bar(offsets, group_stim['top_proportion'], width,
                  color=group_colors[group], alpha=0.7, 
                  edgecolor='black', linewidth=1, label=f'Group {group}')
        
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(stim_types)
        ax.set_xlabel('Stimulus Type')
        ax.set_ylabel('Mean Top Proportion')
        ax.set_title('Preferences by Stimulus', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    else:
        ax.text(0.5, 0.5, 'No stimulus type data', ha='center', va='center',
               transform=ax.transAxes, fontsize=12, color='gray')
        ax.set_title('Preferences by Stimulus', fontweight='bold')
    
    # 6. Scatter plot: Mean vs Variability
    ax = axes[1, 2]
    fish_stats = df.groupby(['group', 'roi_id'])['top_proportion'].agg(['mean', 'std']).reset_index()
    
    for group in [1, 2]:
        group_stats = fish_stats[fish_stats['group'] == group]
        ax.scatter(group_stats['mean'], group_stats['std'],
                  color=group_colors[group], s=100, alpha=0.7,
                  edgecolor='black', linewidth=1, label=f'Group {group}')
        
        # Add fish labels
        for _, row in group_stats.iterrows():
            ax.annotate(f"{int(row['roi_id'])}", 
                       (row['mean'], row['std']),
                       fontsize=8, ha='center', va='center')
    
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.set_xlabel('Mean Top Proportion')
    ax.set_ylabel('Std Dev')
    ax.set_title('Consistency vs Preference', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Group Comparison Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_dir:
        save_path = Path(save_dir) / 'group_comparison.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        console.print(f"[green]✓ Saved comparison plot to {save_path}[/green]")
    
    plt.show()

def plot_group_trajectories(df, save_dir=None):
    """Plot individual fish trajectories colored by group."""
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Colors for groups
    group_colors = {1: '#2E86AB', 2: '#A23B72'}
    
    # Plot each fish's trajectory
    for roi_id in sorted(df['roi_id'].unique()):
        fish_df = df[df['roi_id'] == roi_id].sort_values('trial_number')
        group = fish_df['group'].iloc[0]
        
        ax.plot(fish_df['trial_number'], fish_df['top_proportion'],
               marker='o', color=group_colors[group], linewidth=1.5,
               markersize=6, alpha=0.6, label=f'Fish {roi_id} (G{group})')
    
    # Add group means
    for group in [1, 2]:
        group_means = df[df['group'] == group].groupby('trial_number')['top_proportion'].mean()
        ax.plot(group_means.index, group_means.values,
               color=group_colors[group], linewidth=3, alpha=0.9,
               linestyle='--', label=f'Group {group} mean')
    
    # Add reference line
    ax.axhline(y=0.5, color='black', linestyle=':', alpha=0.5, linewidth=1)
    
    ax.set_xlabel('Trial Number', fontweight='bold')
    ax.set_ylabel('Top Proportion', fontweight='bold')
    ax.set_title('Individual Fish Trajectories by Group', fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
    
    plt.tight_layout()
    
    if save_dir:
        save_path = Path(save_dir) / 'group_trajectories.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        console.print(f"[green]✓ Saved trajectories plot to {save_path}[/green]")
    
    plt.show()

def print_group_statistics(df):
    """Print detailed statistics for each group."""
    console.print("\n[bold cyan]Group Statistics:[/bold cyan]")
    
    for group in [1, 2]:
        group_data = df[df['group'] == group]
        console.print(f"\n[bold]Group {group} (IDs {0 if group==1 else 6}-{5 if group==1 else 11}):[/bold]")
        console.print(f"  Overall mean: {group_data['top_proportion'].mean():.3f}")
        console.print(f"  Overall std: {group_data['top_proportion'].std():.3f}")
        console.print(f"  Median: {group_data['top_proportion'].median():.3f}")
        console.print(f"  Range: [{group_data['top_proportion'].min():.3f}, {group_data['top_proportion'].max():.3f}]")
        
        # Per-fish in group
        fish_means = group_data.groupby('roi_id')['top_proportion'].mean().sort_values(ascending=False)
        console.print(f"  Top fish: ID {fish_means.index[0]} ({fish_means.iloc[0]:.3f})")
        console.print(f"  Bottom fish: ID {fish_means.index[-1]} ({fish_means.iloc[-1]:.3f})")
    
    # Statistical comparison
    group1_data = df[df['group'] == 1]['top_proportion']
    group2_data = df[df['group'] == 2]['top_proportion']
    
    # T-test
    t_stat, t_p = stats.ttest_ind(group1_data, group2_data)
    console.print(f"\n[bold]Group Comparison:[/bold]")
    console.print(f"  T-test: t={t_stat:.3f}, p={t_p:.4f}")
    
    # Mann-Whitney U test (non-parametric alternative)
    u_stat, u_p = stats.mannwhitneyu(group1_data, group2_data, alternative='two-sided')
    console.print(f"  Mann-Whitney U: U={u_stat:.1f}, p={u_p:.4f}")
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt((group1_data.std()**2 + group2_data.std()**2) / 2)
    cohens_d = (group1_data.mean() - group2_data.mean()) / pooled_std
    console.print(f"  Cohen's d: {cohens_d:.3f}")
    
    if abs(cohens_d) < 0.2:
        effect_size = "negligible"
    elif abs(cohens_d) < 0.5:
        effect_size = "small"
    elif abs(cohens_d) < 0.8:
        effect_size = "medium"
    else:
        effect_size = "large"
    console.print(f"  Effect size: {effect_size}")

def save_group_summaries(df, output_dir):
    """Save group-specific summary CSV files."""
    output_path = Path(output_dir)
    
    # Save group means per trial
    group_trial_means = df.groupby(['trial_number', 'group'])['top_proportion'].agg(['mean', 'std', 'count']).reset_index()
    group_trial_means.to_csv(output_path / 'group_trial_means.csv', index=False)
    console.print(f"[green]✓ Saved group trial means[/green]")
    
    # Save individual fish summaries with group labels
    fish_summary = df.groupby(['group', 'roi_id'])['top_proportion'].agg(['mean', 'std', 'min', 'max', 'count']).reset_index()
    fish_summary.to_csv(output_path / 'fish_summary_by_group.csv', index=False)
    console.print(f"[green]✓ Saved fish summaries by group[/green]")

def main():
    parser = argparse.ArgumentParser(
        description='Plot trial preferences with group-based analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('csv_path', help='Path to CSV file with preference data')
    parser.add_argument('--save-dir', type=str, help='Directory to save plots and summaries')
    parser.add_argument('--no-trials', action='store_true', 
                       help='Skip individual trial plots')
    parser.add_argument('--no-summary', action='store_true',
                       help='Skip summary comparison plots')
    parser.add_argument('--no-trajectories', action='store_true',
                       help='Skip trajectory plots')
    
    args = parser.parse_args()
    
    # Create save directory if specified
    if args.save_dir:
        save_path = Path(args.save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        console.print(f"[cyan]Saving outputs to {save_path}[/cyan]")
    
    # Load and process data
    df = load_and_group_data(args.csv_path)
    
    # Print statistics
    print_group_statistics(df)
    
    # Create plots
    if not args.no_trials:
        console.print("\n[cyan]Creating trial-by-trial plots...[/cyan]")
        plot_trials_by_group(df, args.save_dir)
    
    if not args.no_summary:
        console.print("\n[cyan]Creating group comparison plots...[/cyan]")
        plot_group_comparison_summary(df, args.save_dir)
    
    if not args.no_trajectories:
        console.print("\n[cyan]Creating trajectory plots...[/cyan]")
        plot_group_trajectories(df, args.save_dir)
    
    # Save summaries if directory specified
    if args.save_dir:
        console.print("\n[cyan]Saving summary CSV files...[/cyan]")
        save_group_summaries(df, args.save_dir)
    
    console.print("\n[green]✓ Analysis complete![/green]")

if __name__ == '__main__':
    main()