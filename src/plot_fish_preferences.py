#!/usr/bin/env python3
"""
Per-Fish Trial Preference Plotter

Visualizes trial-by-trial quadrant preferences for each fish from the analyzer output.
Creates individual plots showing how each fish's top-quadrant preference varies across trials.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path
from rich.console import Console

console = Console()
sns.set_palette("husl")

def load_preference_data(csv_path):
    """Load the trial preference data from CSV."""
    df = pd.read_csv(csv_path)
    console.print(f"[green]Loaded {len(df)} trial records for {df['roi_id'].nunique()} fish[/green]")
    return df

def plot_individual_fish_preferences(df, save_dir=None):
    """Create individual plots for each fish showing trial-by-trial preferences."""
    n_fish = df['roi_id'].nunique()
    fish_ids = sorted(df['roi_id'].unique())
    
    # Create figure with subplots for each fish
    n_cols = min(3, n_fish)
    n_rows = int(np.ceil(n_fish / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
    if n_fish == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]
    
    stimulus_colors = {
        'grating': '#FF6B6B',
        'black': '#4ECDC4',
        'white': '#95E77E',
        'unknown': '#FFE66D'
    }
    
    for idx, fish_id in enumerate(fish_ids):
        ax = axes[idx]
        fish_df = df[df['roi_id'] == fish_id].sort_values('trial_number')
        
        # Plot bars colored by stimulus type
        for trial_type in fish_df['trial_type'].unique():
            type_df = fish_df[fish_df['trial_type'] == trial_type]
            ax.bar(type_df['trial_number'], type_df['top_proportion'],
                  color=stimulus_colors.get(trial_type, 'gray'),
                  alpha=0.7, label=trial_type, width=0.8)
        
        # Add 50% line
        ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, linewidth=1)
        
        # Add mean line
        mean_pref = fish_df['top_proportion'].mean()
        ax.axhline(y=mean_pref, color='red', linestyle='-', alpha=0.7, 
                  linewidth=2, label=f'Mean: {mean_pref:.2%}')
        
        # Formatting
        ax.set_xlabel('Trial Number')
        ax.set_ylabel('Top Proportion')
        ax.set_title(f'Fish {fish_id}', fontweight='bold')
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add legend only to first plot
        if idx == 0:
            ax.legend(loc='upper right', fontsize=9)
        
        # Add stats text
        std_pref = fish_df['top_proportion'].std()
        stats_text = f'μ={mean_pref:.2f}\nσ={std_pref:.2f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Hide unused subplots
    for idx in range(n_fish, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Trial-by-Trial Top Quadrant Preference', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_dir:
        save_path = Path(save_dir) / 'individual_fish_preferences.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        console.print(f"[green]✓ Saved individual plot to {save_path}[/green]")
    
    plt.show()

def plot_fish_preference_matrix(df, save_dir=None):
    """Create a heatmap showing all fish x trials preference matrix."""
    # Pivot data to create matrix
    pivot_df = df.pivot_table(values='top_proportion', 
                              index='roi_id', 
                              columns='trial_number',
                              aggfunc='mean')
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create heatmap
    im = ax.imshow(pivot_df, cmap='RdBu_r', vmin=0, vmax=1, aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Top Proportion', rotation=270, labelpad=20)
    
    # Set ticks
    ax.set_xticks(range(len(pivot_df.columns)))
    ax.set_xticklabels(pivot_df.columns)
    ax.set_yticks(range(len(pivot_df.index)))
    ax.set_yticklabels([f'Fish {i}' for i in pivot_df.index])
    
    # Labels
    ax.set_xlabel('Trial Number', fontweight='bold')
    ax.set_ylabel('Fish ID', fontweight='bold')
    ax.set_title('Fish Preference Matrix (All Fish × All Trials)', fontweight='bold')
    
    # Add grid
    ax.set_xticks(np.arange(-0.5, len(pivot_df.columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(pivot_df.index), 1), minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=0.5)
    
    # Add text annotations for values
    if len(pivot_df.index) <= 12 and len(pivot_df.columns) <= 20:
        for i in range(len(pivot_df.index)):
            for j in range(len(pivot_df.columns)):
                value = pivot_df.iloc[i, j]
                if not np.isnan(value):
                    color = 'white' if value < 0.3 or value > 0.7 else 'black'
                    ax.text(j, i, f'{value:.2f}', ha='center', va='center',
                           color=color, fontsize=8)
    
    plt.tight_layout()
    
    if save_dir:
        save_path = Path(save_dir) / 'preference_matrix.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        console.print(f"[green]✓ Saved matrix plot to {save_path}[/green]")
    
    plt.show()

def plot_preference_distributions(df, save_dir=None):
    """Plot distributions of preferences across fish."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Overall distribution
    ax = axes[0]
    ax.hist(df['top_proportion'], bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    ax.axvline(x=0.5, color='red', linestyle='--', linewidth=2, alpha=0.7, label='No preference')
    ax.axvline(x=df['top_proportion'].mean(), color='orange', linestyle='-', 
              linewidth=2, alpha=0.7, label=f'Mean: {df["top_proportion"].mean():.2f}')
    ax.set_xlabel('Top Proportion')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of All Trial Preferences', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Per-fish means
    ax = axes[1]
    fish_means = df.groupby('roi_id')['top_proportion'].mean().sort_values()
    colors = ['#e74c3c' if x < 0.5 else '#3498db' for x in fish_means.values]
    
    bars = ax.barh(range(len(fish_means)), fish_means.values, color=colors, alpha=0.7)
    ax.axvline(x=0.5, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_yticks(range(len(fish_means)))
    ax.set_yticklabels([f'Fish {i}' for i in fish_means.index])
    ax.set_xlabel('Mean Top Proportion')
    ax.set_title('Fish Ranked by Preference', fontweight='bold')
    ax.set_xlim([0, 1])
    ax.grid(True, alpha=0.3, axis='x')
    
    # 3. Preference by stimulus type
    ax = axes[2]
    stimulus_data = df.groupby('trial_type')['top_proportion'].apply(list).to_dict()
    
    positions = []
    labels = []
    data_to_plot = []
    colors_to_use = []
    
    stimulus_colors = {
        'grating': '#FF6B6B',
        'black': '#4ECDC4',
        'white': '#95E77E',
        'unknown': '#FFE66D'
    }
    
    pos = 1
    for stim_type, values in stimulus_data.items():
        data_to_plot.append(values)
        positions.append(pos)
        labels.append(stim_type)
        colors_to_use.append(stimulus_colors.get(stim_type, 'gray'))
        pos += 1
    
    bp = ax.boxplot(data_to_plot, positions=positions, patch_artist=True, widths=0.6)
    for patch, color in zip(bp['boxes'], colors_to_use):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.axhline(y=0.5, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Top Proportion')
    ax.set_xlabel('Stimulus Type')
    ax.set_title('Preference by Stimulus', fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Quadrant Preference Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_dir:
        save_path = Path(save_dir) / 'preference_distributions.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        console.print(f"[green]✓ Saved distributions plot to {save_path}[/green]")
    
    plt.show()

def plot_preference_trajectories(df, save_dir=None):
    """Plot preference trajectories over trials for all fish."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Get unique fish and assign colors
    fish_ids = sorted(df['roi_id'].unique())
    colors = plt.cm.tab20(np.linspace(0, 1, len(fish_ids)))
    
    # Plot each fish's trajectory
    for idx, fish_id in enumerate(fish_ids):
        fish_df = df[df['roi_id'] == fish_id].sort_values('trial_number')
        ax.plot(fish_df['trial_number'], fish_df['top_proportion'],
               marker='o', label=f'Fish {fish_id}', 
               color=colors[idx], linewidth=1.5, markersize=5, alpha=0.7)
    
    # Add reference lines
    ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, linewidth=1)
    ax.axhline(y=df['top_proportion'].mean(), color='red', 
              linestyle='-', alpha=0.3, linewidth=2,
              label=f'Population mean: {df["top_proportion"].mean():.2f}')
    
    # Formatting
    ax.set_xlabel('Trial Number', fontweight='bold')
    ax.set_ylabel('Top Proportion', fontweight='bold')
    ax.set_title('Individual Fish Preference Trajectories', fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    
    # Legend
    if len(fish_ids) <= 12:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1)
    else:
        ax.text(0.99, 0.99, f'{len(fish_ids)} fish', transform=ax.transAxes,
               ha='right', va='top', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_dir:
        save_path = Path(save_dir) / 'preference_trajectories.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        console.print(f"[green]✓ Saved trajectories plot to {save_path}[/green]")
    
    plt.show()

def print_summary_statistics(df):
    """Print summary statistics for the preference data."""
    console.print("\n[bold cyan]Summary Statistics:[/bold cyan]")
    
    # Overall stats
    console.print("\n[bold]Overall:[/bold]")
    console.print(f"  Mean preference: {df['top_proportion'].mean():.3f}")
    console.print(f"  Std deviation: {df['top_proportion'].std():.3f}")
    console.print(f"  Median: {df['top_proportion'].median():.3f}")
    
    # Per-fish stats
    fish_stats = df.groupby('roi_id')['top_proportion'].agg(['mean', 'std', 'min', 'max'])
    console.print("\n[bold]Per-fish summary:[/bold]")
    console.print(f"  Fish with top preference: {fish_stats['mean'].idxmax()} ({fish_stats['mean'].max():.3f})")
    console.print(f"  Fish with bottom preference: {fish_stats['mean'].idxmin()} ({fish_stats['mean'].min():.3f})")
    console.print(f"  Most consistent fish: {fish_stats['std'].idxmin()} (σ={fish_stats['std'].min():.3f})")
    console.print(f"  Most variable fish: {fish_stats['std'].idxmax()} (σ={fish_stats['std'].max():.3f})")
    
    # By stimulus type
    if 'trial_type' in df.columns:
        console.print("\n[bold]By stimulus type:[/bold]")
        stim_stats = df.groupby('trial_type')['top_proportion'].agg(['mean', 'std', 'count'])
        for stim_type, row in stim_stats.iterrows():
            console.print(f"  {stim_type}: {row['mean']:.3f} ± {row['std']:.3f} (n={int(row['count'])})")

def main():
    parser = argparse.ArgumentParser(
        description='Plot per-fish trial preference scores',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('csv_path', help='Path to CSV file with preference data')
    parser.add_argument('--save-dir', type=str, help='Directory to save plots')
    parser.add_argument('--no-individual', action='store_true', 
                       help='Skip individual fish plots')
    parser.add_argument('--no-matrix', action='store_true',
                       help='Skip preference matrix plot')
    parser.add_argument('--no-distributions', action='store_true',
                       help='Skip distribution plots')
    parser.add_argument('--no-trajectories', action='store_true',
                       help='Skip trajectory plots')
    
    args = parser.parse_args()
    
    # Create save directory if specified
    if args.save_dir:
        save_path = Path(args.save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        console.print(f"[cyan]Saving plots to {save_path}[/cyan]")
    
    # Load data
    df = load_preference_data(args.csv_path)
    
    # Print summary statistics
    print_summary_statistics(df)
    
    # Create plots
    if not args.no_individual:
        console.print("\n[cyan]Creating individual fish plots...[/cyan]")
        plot_individual_fish_preferences(df, args.save_dir)
    
    if not args.no_matrix:
        console.print("\n[cyan]Creating preference matrix...[/cyan]")
        plot_fish_preference_matrix(df, args.save_dir)
    
    if not args.no_distributions:
        console.print("\n[cyan]Creating distribution plots...[/cyan]")
        plot_preference_distributions(df, args.save_dir)
    
    if not args.no_trajectories:
        console.print("\n[cyan]Creating trajectory plots...[/cyan]")
        plot_preference_trajectories(df, args.save_dir)
    
    console.print("\n[green]✓ All plots complete![/green]")

if __name__ == '__main__':
    main()