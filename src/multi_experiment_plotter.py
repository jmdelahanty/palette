#!/usr/bin/env python3
"""
Multi-Experiment Quadrant Preference Analyzer

Combines and plots quadrant preference data from multiple experiments with:
- Custom time windows
- Fish exclusion lists
- Cross-experiment averaging
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path
from typing import List, Optional, Tuple
from rich.console import Console

console = Console()
sns.set_style("whitegrid")

def load_and_filter_experiment(
    csv_path: str,
    exclude_fish: List[int],
    time_range: Tuple[float, float],
    experiment_id: int
) -> pd.DataFrame:
    """
    Load and filter data from one experiment.
    
    Args:
        csv_path: Path to CSV file
        exclude_fish: List of fish IDs to exclude
        time_range: (start_minute, end_minute) to include
        experiment_id: Experiment identifier for tracking
    
    Returns:
        Filtered DataFrame with experiment ID added
    """
    console.print(f"[cyan]Loading Experiment {experiment_id}: {csv_path}[/cyan]")
    
    # Load data
    df = pd.read_csv(csv_path)
    
    # Standardize time column name
    if 'minute' in df.columns:
        df['time_minutes'] = df['minute']
    elif 'time_minutes' not in df.columns:
        console.print("[yellow]Warning: No time column found, using index[/yellow]")
        df['time_minutes'] = df.index
    
    # Add experiment ID
    df['experiment'] = experiment_id
    
    # Filter by time range
    start_min, end_min = time_range
    df = df[(df['time_minutes'] >= start_min) & (df['time_minutes'] <= end_min)]
    console.print(f"  Time filter: {start_min}-{end_min} minutes → {len(df)} records")
    
    # Exclude specified fish
    initial_count = len(df)
    df = df[~df['roi_id'].isin(exclude_fish)]
    console.print(f"  Excluded fish {exclude_fish} → removed {initial_count - len(df)} records")
    
    # Add group if not present
    if 'group' not in df.columns:
        df['group'] = df['roi_id'].apply(lambda x: 1 if x <= 5 else 2)
    
    # Ensure we have top_proportion column
    if 'top_proportion' not in df.columns and 'roi_q1_proportion' in df.columns:
        df['top_proportion'] = df['roi_q1_proportion']
    
    console.print(f"  Final: {len(df)} records, {df['roi_id'].nunique()} fish")
    
    return df


def combine_experiments(
    exp1_data: pd.DataFrame,
    exp2_data: pd.DataFrame
) -> pd.DataFrame:
    """
    Combine and average data across two experiments.
    
    Returns:
        Combined DataFrame with averaged values per fish per time point
    """
    # Combine the dataframes
    combined = pd.concat([exp1_data, exp2_data], ignore_index=True)
    
    # For averaging across experiments, group by roi_id and time
    # This assumes the same fish IDs represent equivalent subjects across experiments
    averaged = combined.groupby(['roi_id', 'time_minutes', 'group']).agg({
        'top_proportion': 'mean',
        'experiment': lambda x: list(x)  # Keep track of which experiments contributed
    }).reset_index()
    
    # Add a count of how many experiments contributed to each point
    averaged['n_experiments'] = averaged['experiment'].apply(len)
    
    console.print(f"\n[green]Combined data: {len(averaged)} averaged data points[/green]")
    console.print(f"[green]Fish included: {sorted(averaged['roi_id'].unique())}[/green]")
    
    return averaged


def plot_averaged_preference(
    df: pd.DataFrame,
    save_dir: Optional[str] = None,
    show_individual: bool = False,
    file_format: str = 'svg'
):
    """
    Plot the averaged quadrant preference data.
    
    Args:
        df: Combined/averaged DataFrame
        save_dir: Directory to save plots
        show_individual: Whether to show individual fish lines
        file_format: Output format ('svg' or 'png')
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Colors for groups
    group_colors = {1: '#2E86AB', 2: '#A23B72'}
    
    # Get unique time points
    unique_times = sorted(df['time_minutes'].unique())
    
    # Plot individual fish if requested (now defaults to False)
    if show_individual:
        for group in [1, 2]:
            group_df = df[df['group'] == group]
            for roi_id in sorted(group_df['roi_id'].unique()):
                fish_data = group_df[group_df['roi_id'] == roi_id].sort_values('time_minutes')
                ax.plot(fish_data['time_minutes'], fish_data['top_proportion'],
                       color=group_colors[group], alpha=0.3, linewidth=1,
                       marker='o', markersize=3)
    
    # Calculate and plot group means with error bars
    for group in [1, 2]:
        group_df = df[df['group'] == group]
        
        means = []
        sems = []
        times = []
        
        for time_point in unique_times:
            time_data = group_df[group_df['time_minutes'] == time_point]
            if len(time_data) > 0:
                means.append(time_data['top_proportion'].mean())
                sems.append(time_data['top_proportion'].sem())
                times.append(time_point)
        
        # Plot mean line
        ax.plot(times, means,
               color=group_colors[group], linewidth=3, alpha=0.9,
               marker='o', markersize=8, markeredgecolor='black', markeredgewidth=1.5,
               label=f'Group {group} (n={group_df["roi_id"].nunique()} fish)')
        
        # Add error bars
        ax.errorbar(times, means, yerr=sems,
                   color=group_colors[group], alpha=0.7,
                   capsize=5, capthick=2, elinewidth=2,
                   fmt='none')
    
    # Add reference line at 50%
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1.5,
              label='No preference (50%)')
    
    # Formatting
    ax.set_xlabel('Time (minutes)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Proportion of Time in Top Quadrant', fontweight='bold', fontsize=12)
    ax.set_title('Averaged Quadrant Preference Across Experiments\n(Minutes 6-10, Selected Fish)',
                fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10, framealpha=0.95)
    
    # Set x-axis ticks to whole minutes only
    min_time = int(np.floor(df['time_minutes'].min()))
    max_time = int(np.ceil(df['time_minutes'].max()))
    ax.set_xticks(range(min_time, max_time + 1))
    ax.set_xticklabels(range(min_time, max_time + 1))
    
    # Set y-axis limits
    ax.set_ylim([0, 1])
    
    # Add experiment info as text
    n_exp1 = df[df['experiment'].apply(lambda x: 1 in x if isinstance(x, list) else x == 1)]['roi_id'].nunique()
    n_exp2 = df[df['experiment'].apply(lambda x: 2 in x if isinstance(x, list) else x == 2)]['roi_id'].nunique()
    
    info_text = f'Exp 1: {n_exp1} fish | Exp 2: {n_exp2} fish'
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save in requested format
        if file_format.lower() == 'svg':
            save_file = save_path / 'averaged_preference_plot.svg'
            plt.savefig(save_file, format='svg', bbox_inches='tight')
            console.print(f"[green]✓ Saved plot to {save_file} (SVG format)[/green]")
        else:
            save_file = save_path / f'averaged_preference_plot.{file_format}'
            plt.savefig(save_file, dpi=150, bbox_inches='tight')
            console.print(f"[green]✓ Saved plot to {save_file}[/green]")
    
    plt.show()


def print_statistics(df: pd.DataFrame):
    """Print summary statistics for the combined data."""
    console.print("\n[bold cyan]Summary Statistics:[/bold cyan]")
    
    for group in [1, 2]:
        group_df = df[df['group'] == group]
        if len(group_df) == 0:
            continue
            
        console.print(f"\n[bold]Group {group}:[/bold]")
        console.print(f"  Fish included: {sorted(group_df['roi_id'].unique())}")
        console.print(f"  Mean top proportion: {group_df['top_proportion'].mean():.3f} ± {group_df['top_proportion'].std():.3f}")
        console.print(f"  Data points: {len(group_df)}")
        
        # Check experiment representation
        exp_coverage = group_df.groupby('roi_id')['n_experiments'].mean()
        console.print(f"  Average experiments per fish: {exp_coverage.mean():.1f}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze and plot quadrant preferences across multiple experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default settings
  python multi_experiment_plotter.py exp1.csv exp2.csv
  
  # Custom time range and exclusions
  python multi_experiment_plotter.py exp1.csv exp2.csv \\
    --time-range 6 10 \\
    --exclude-exp1 3 4 5 11 \\
    --exclude-exp2 0 2 3 5 11
        """
    )
    
    parser.add_argument('exp1_csv', help='Path to first experiment CSV file')
    parser.add_argument('exp2_csv', help='Path to second experiment CSV file')
    
    parser.add_argument('--time-range', nargs=2, type=float, default=[6, 10],
                       help='Time range to analyze (start end, in minutes)')
    
    parser.add_argument('--exclude-exp1', nargs='+', type=int, default=[3, 4, 5, 11],
                       help='Fish IDs to exclude from experiment 1')
    
    parser.add_argument('--exclude-exp2', nargs='+', type=int, default=[0, 2, 3, 5, 11],
                       help='Fish IDs to exclude from experiment 2')
    
    parser.add_argument('--save-dir', type=str,
                       help='Directory to save plots and output')
    
    parser.add_argument('--save-combined-csv', action='store_true',
                       help='Save the combined/averaged data to CSV')
    
    parser.add_argument('--no-individual', action='store_true', default=True,
                       help='Hide individual fish lines in plot (default: True)')
    
    parser.add_argument('--show-individual', action='store_true',
                       help='Show individual fish trajectories')
    
    parser.add_argument('--format', type=str, default='svg',
                       choices=['svg', 'png', 'pdf'],
                       help='Output file format (default: svg)')
    
    args = parser.parse_args()
    
    console.print("[bold]Multi-Experiment Quadrant Preference Analysis[/bold]\n")
    
    # Load and filter each experiment
    exp1_data = load_and_filter_experiment(
        args.exp1_csv,
        args.exclude_exp1,
        tuple(args.time_range),
        experiment_id=1
    )
    
    exp2_data = load_and_filter_experiment(
        args.exp2_csv,
        args.exclude_exp2,
        tuple(args.time_range),
        experiment_id=2
    )
    
    # Combine and average
    combined_data = combine_experiments(exp1_data, exp2_data)
    
    # Save combined data if requested
    if args.save_combined_csv and args.save_dir:
        save_path = Path(args.save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        combined_file = save_path / 'combined_averaged_data.csv'
        combined_data.to_csv(combined_file, index=False)
        console.print(f"[green]✓ Saved combined data to {combined_file}[/green]")
    
    # Print statistics
    print_statistics(combined_data)
    
    # Create plot
    console.print("\n[cyan]Creating plot...[/cyan]")
    plot_averaged_preference(
        combined_data,
        save_dir=args.save_dir,
        show_individual=args.show_individual,  # Now defaults to False
        file_format=args.format
    )
    
    console.print("\n[green]✓ Analysis complete![/green]")


if __name__ == '__main__':
    main()