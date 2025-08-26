#!/usr/bin/env python3
"""
Minute-by-Minute Group Preference Plotter

Aggregates trial preferences into minute bins and plots them with group color coding.
Shows temporal progression of preferences for both groups across the experiment.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path
from scipy import stats
from rich.console import Console
from scipy.ndimage import uniform_filter1d

console = Console()
sns.set_style("whitegrid")

def load_and_process_data(csv_path, fps=60.0):
    """Load preference data and convert to minute-based bins."""
    df = pd.read_csv(csv_path)
    
    # Assign groups based on fish ID (if not already present)
    if 'group' not in df.columns:
        df['group'] = df['roi_id'].apply(lambda x: 1 if x <= 5 else 2)
    
    # Convert to time_minutes if not already present
    if 'time_minutes' not in df.columns:
        if 'minute' in df.columns:
            # Data already has minute column (from minute extractor)
            df['time_minutes'] = df['minute']
        elif 'start_frame' in df.columns:
            # Trial-based data - convert frames to minutes
            df['time_minutes'] = df['start_frame'] / (fps * 60)
        elif 'trial_number' in df.columns:
            # Estimate based on trial number and assuming ~10s trials
            df['time_minutes'] = df['trial_number'] * 0.5  # Rough estimate
        else:
            # If we can't determine time, use row index as proxy
            console.print("[yellow]Warning: No time information found, using index as proxy[/yellow]")
            df['time_minutes'] = df.index
    
    console.print(f"[green]Loaded {len(df)} records[/green]")
    console.print(f"[cyan]Time range: {df['time_minutes'].min():.1f} - {df['time_minutes'].max():.1f} minutes[/cyan]")
    console.print(f"[yellow]Group 1 (IDs 0-5): {df[df['group']==1]['roi_id'].nunique()} fish[/yellow]")
    console.print(f"[magenta]Group 2 (IDs 6-11): {df[df['group']==2]['roi_id'].nunique()} fish[/magenta]")
    
    return df

def bin_data_by_minute(df, bin_size=1.0):
    """Aggregate data into minute bins."""
    # Create time bins
    max_time = df['time_minutes'].max()
    min_time = df['time_minutes'].min()
    bins = np.arange(min_time, max_time + bin_size, bin_size)
    
    # Assign bins
    df['time_bin'] = pd.cut(df['time_minutes'], bins=bins, 
                            labels=bins[:-1], include_lowest=True)
    
    # Aggregate by bin and group
    binned_data = []
    for time_bin in bins[:-1]:
        bin_mask = (df['time_minutes'] >= time_bin) & (df['time_minutes'] < time_bin + bin_size)
        bin_df = df[bin_mask]
        
        if len(bin_df) > 0:
            for group in [1, 2]:
                group_df = bin_df[bin_df['group'] == group]
                if len(group_df) > 0:
                    binned_data.append({
                        'minute': time_bin + bin_size/2,  # Center of bin
                        'group': group,
                        'mean_preference': group_df['top_proportion'].mean(),
                        'std_preference': group_df['top_proportion'].std(),
                        'sem_preference': group_df['top_proportion'].sem(),
                        'n_samples': len(group_df),
                        'fish_ids': group_df['roi_id'].unique().tolist()
                    })
    
    return pd.DataFrame(binned_data)

def plot_minute_by_minute_scatter(df, binned_df, save_dir=None, interpolate=False):
    """Create minute-by-minute scatter plot with all individual points and error bars."""
    fig, ax1 = plt.subplots(figsize=(16, 8))
    
    # Colors for groups
    group_colors = {1: '#2E86AB', 2: '#A23B72'}
    
    # Main plot - Plot individual points with jitter
    for group in [1, 2]:
        group_data = df[df['group'] == group]
        
        # Add small jitter to time for visibility
        time_jitter = np.random.normal(0, 0.02, size=len(group_data))
        
        ax1.scatter(group_data['time_minutes'] + time_jitter, 
                  group_data['top_proportion'],
                  color=group_colors[group], alpha=0.4, s=30,
                  edgecolor='none', label=f'Group {group} (IDs {0 if group==1 else 6}-{5 if group==1 else 11})')
    
    # Calculate means at actual data points instead of using binned data
    unique_times = sorted(df['time_minutes'].unique())
    
    for group in [1, 2]:
        group_df = df[df['group'] == group]
        
        # Calculate mean and SEM at each actual time point
        means = []
        sems = []
        times = []
        
        for time_point in unique_times:
            time_data = group_df[np.abs(group_df['time_minutes'] - time_point) < 0.01]  # Small tolerance for floating point
            if len(time_data) > 0:
                means.append(time_data['top_proportion'].mean())
                sems.append(time_data['top_proportion'].sem())
                times.append(time_point)
        
        # Plot means with connecting lines
        if len(times) > 0:
            ax1.plot(times, means,
                   color=group_colors[group], linewidth=3, alpha=0.9,
                   marker='o', markersize=8, markeredgecolor='black', markeredgewidth=1,
                   linestyle='-')
            
            # Add error bars
            ax1.errorbar(times, means, yerr=sems,
                       color=group_colors[group], alpha=0.7,
                       capsize=5, capthick=2, elinewidth=2,
                       fmt='none')
    
    # Add 50% reference line
    ax1.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, linewidth=1.5)
    
    # Formatting for main plot
    ax1.set_xlabel('Time (minutes)', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Top Proportion', fontweight='bold', fontsize=12)
    ax1.set_title('Preferences Over Time', fontweight='bold', fontsize=14)
    ax1.set_ylim([0, 1])
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=10)
    
    # Add minute markers
    max_time = df['time_minutes'].max()
    minute_ticks = np.arange(0, np.ceil(max_time) + 1, 1)
    ax1.set_xticks(minute_ticks)
    ax1.set_xticklabels([f'{int(m)}' for m in minute_ticks])
    
    plt.tight_layout()
    
    # Print trial timing information to console
    unique_times_list = sorted(df['time_minutes'].unique())
    console.print("\n[bold cyan]Data Summary:[/bold cyan]")
    console.print(f"Data points at: {[f'{t:.1f}' for t in unique_times_list]} minutes")
    console.print(f"Total time points: {len(unique_times_list)}")
    
    if save_dir:
        save_path = Path(save_dir) / 'minute_by_minute_scatter.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        console.print(f"[green]✓ Saved scatter plot to {save_path}[/green]")
    
    plt.show()

def plot_minute_by_minute_bars(binned_df, save_dir=None):
    """Create minute-by-minute bar plot showing group means."""
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Colors for groups
    group_colors = {1: '#2E86AB', 2: '#A23B72'}
    
    # Get unique minutes
    minutes = sorted(binned_df['minute'].unique())
    x_positions = np.arange(len(minutes))
    width = 0.35
    
    # Plot bars for each group
    for i, group in enumerate([1, 2]):
        group_data = binned_df[binned_df['group'] == group]
        
        # Align data with minutes
        means = []
        errors = []
        for minute in minutes:
            minute_data = group_data[group_data['minute'] == minute]
            if len(minute_data) > 0:
                means.append(minute_data['mean_preference'].iloc[0])
                errors.append(minute_data['sem_preference'].iloc[0])
            else:
                means.append(np.nan)
                errors.append(0)
        
        # Plot bars
        offset = (i - 0.5) * width
        bars = ax.bar(x_positions + offset, means, width,
                     yerr=errors, capsize=3,
                     color=group_colors[group], alpha=0.7,
                     edgecolor='black', linewidth=1,
                     label=f'Group {group}')
    
    # Add 50% reference line
    ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, linewidth=1.5)
    
    # Formatting
    ax.set_xlabel('Time (minutes)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Mean Top Proportion', fontweight='bold', fontsize=12)
    ax.set_title('Minute-by-Minute Group Preferences', fontweight='bold', fontsize=14)
    ax.set_ylim([0, 1])
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f'{m:.1f}' for m in minutes], rotation=45, ha='right')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_dir:
        save_path = Path(save_dir) / 'minute_by_minute_bars.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        console.print(f"[green]✓ Saved bar plot to {save_path}[/green]")
    
    plt.show()

def plot_minute_by_minute_continuous(binned_df, save_dir=None):
    """Create continuous line plot with confidence bands."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), 
                                   gridspec_kw={'height_ratios': [3, 1]})
    
    # Colors for groups
    group_colors = {1: '#2E86AB', 2: '#A23B72'}
    
    # Main plot - preferences over time
    for group in [1, 2]:
        group_data = binned_df[binned_df['group'] == group].sort_values('minute')
        
        # Plot mean line
        ax1.plot(group_data['minute'], group_data['mean_preference'],
                color=group_colors[group], linewidth=2.5, 
                label=f'Group {group}', marker='o', markersize=6)
        
        # Add confidence band (mean ± SEM)
        lower = group_data['mean_preference'] - group_data['sem_preference']
        upper = group_data['mean_preference'] + group_data['sem_preference']
        ax1.fill_between(group_data['minute'], lower, upper,
                        color=group_colors[group], alpha=0.2)
    
    ax1.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, linewidth=1.5)
    ax1.set_ylabel('Mean Top Proportion', fontweight='bold', fontsize=12)
    ax1.set_title('Group Preferences Over Time (Minute Bins)', fontweight='bold', fontsize=14)
    ax1.set_ylim([0, 1])
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Difference plot
    # Align data for both groups
    minutes = sorted(binned_df['minute'].unique())
    group1_means = []
    group2_means = []
    
    for minute in minutes:
        g1_data = binned_df[(binned_df['group'] == 1) & (binned_df['minute'] == minute)]
        g2_data = binned_df[(binned_df['group'] == 2) & (binned_df['minute'] == minute)]
        
        if len(g1_data) > 0 and len(g2_data) > 0:
            group1_means.append(g1_data['mean_preference'].iloc[0])
            group2_means.append(g2_data['mean_preference'].iloc[0])
        else:
            group1_means.append(np.nan)
            group2_means.append(np.nan)
    
    differences = np.array(group1_means) - np.array(group2_means)
    
    ax2.plot(minutes, differences, 'k-', linewidth=2, marker='s', markersize=5)
    ax2.fill_between(minutes, 0, differences, 
                     where=(differences >= 0), alpha=0.3, color=group_colors[1],
                     label='Group 1 > Group 2')
    ax2.fill_between(minutes, 0, differences, 
                     where=(differences < 0), alpha=0.3, color=group_colors[2],
                     label='Group 2 > Group 1')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
    ax2.set_xlabel('Time (minutes)', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Difference\n(G1 - G2)', fontweight='bold', fontsize=10)
    ax2.set_ylim([-0.5, 0.5])
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', fontsize=9)
    
    plt.tight_layout()
    
    if save_dir:
        save_path = Path(save_dir) / 'minute_by_minute_continuous.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        console.print(f"[green]✓ Saved continuous plot to {save_path}[/green]")
    
    plt.show()

def plot_individual_fish_minutes(df, save_dir=None):
    """Plot each fish individually over minutes, grouped by color."""
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    # Colors for groups
    group_colors = {1: '#2E86AB', 2: '#A23B72'}
    
    # Bin size for smoothing
    bin_size = 0.5  # 30-second bins
    
    for group_idx, group in enumerate([1, 2]):
        ax = axes[group_idx]
        group_df = df[df['group'] == group]
        fish_ids = sorted(group_df['roi_id'].unique())
        
        # Create lighter versions of group color for individual fish
        base_color = np.array([int(group_colors[group][i:i+2], 16)/255 for i in (1, 3, 5)])
        
        for fish_idx, fish_id in enumerate(fish_ids):
            fish_data = group_df[group_df['roi_id'] == fish_id].sort_values('time_minutes')
            
            # Bin the data for this fish
            max_time = fish_data['time_minutes'].max()
            min_time = fish_data['time_minutes'].min()
            bins = np.arange(min_time, max_time + bin_size, bin_size)
            
            binned_means = []
            binned_times = []
            for i in range(len(bins) - 1):
                bin_mask = (fish_data['time_minutes'] >= bins[i]) & (fish_data['time_minutes'] < bins[i+1])
                bin_data = fish_data[bin_mask]
                if len(bin_data) > 0:
                    binned_means.append(bin_data['top_proportion'].mean())
                    binned_times.append((bins[i] + bins[i+1]) / 2)
            
            # Vary alpha for different fish
            alpha = 0.3 + (fish_idx / len(fish_ids)) * 0.5
            
            ax.plot(binned_times, binned_means,
                   color=group_colors[group], alpha=alpha,
                   linewidth=1.5, label=f'Fish {fish_id}')
        
        # Add group mean as thick line
        group_times = []
        group_means = []
        max_time = group_df['time_minutes'].max()
        min_time = group_df['time_minutes'].min()
        bins = np.arange(min_time, max_time + bin_size, bin_size)
        
        for i in range(len(bins) - 1):
            bin_mask = (group_df['time_minutes'] >= bins[i]) & (group_df['time_minutes'] < bins[i+1])
            bin_data = group_df[bin_mask]
            if len(bin_data) > 0:
                group_means.append(bin_data['top_proportion'].mean())
                group_times.append((bins[i] + bins[i+1]) / 2)
        
        ax.plot(group_times, group_means, color=group_colors[group], 
               linewidth=4, alpha=0.9, label=f'Group {group} Mean')
        
        ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, linewidth=1)
        ax.set_ylabel('Top Proportion', fontweight='bold')
        ax.set_title(f'Group {group} (Fish IDs {fish_ids[0]}-{fish_ids[-1]})', fontweight='bold')
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', ncol=3, fontsize=8)
    
    axes[1].set_xlabel('Time (minutes)', fontweight='bold')
    
    plt.suptitle('Individual Fish Trajectories by Group (Minute Resolution)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_dir:
        save_path = Path(save_dir) / 'individual_fish_minutes.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        console.print(f"[green]✓ Saved individual fish plot to {save_path}[/green]")
    
    plt.show()

def print_temporal_statistics(binned_df):
    """Print statistics about temporal patterns."""
    console.print("\n[bold cyan]Temporal Statistics:[/bold cyan]")
    
    # Overall temporal trends
    for group in [1, 2]:
        group_data = binned_df[binned_df['group'] == group].sort_values('minute')
        
        # Calculate correlation with time
        if len(group_data) > 2:
            corr, p_value = stats.pearsonr(group_data['minute'], group_data['mean_preference'])
            console.print(f"\n[bold]Group {group} temporal trend:[/bold]")
            console.print(f"  Correlation with time: r={corr:.3f}, p={p_value:.4f}")
            
            if abs(corr) < 0.3:
                trend = "weak/no"
            elif abs(corr) < 0.6:
                trend = "moderate"
            else:
                trend = "strong"
            
            if corr > 0.1:
                direction = "increasing"
            elif corr < -0.1:
                direction = "decreasing"
            else:
                direction = "stable"
            
            console.print(f"  Trend: {trend} {direction}")
            console.print(f"  Start preference: {group_data.iloc[0]['mean_preference']:.3f}")
            console.print(f"  End preference: {group_data.iloc[-1]['mean_preference']:.3f}")
            console.print(f"  Change: {group_data.iloc[-1]['mean_preference'] - group_data.iloc[0]['mean_preference']:.3f}")

def save_minute_summaries(binned_df, output_dir):
    """Save minute-binned summaries to CSV."""
    output_path = Path(output_dir)
    
    # Save the binned data
    binned_df.to_csv(output_path / 'minute_binned_preferences.csv', index=False)
    console.print(f"[green]✓ Saved minute-binned data[/green]")
    
    # Create and save a pivot table
    pivot_df = binned_df.pivot_table(values='mean_preference', 
                                     index='minute', 
                                     columns='group',
                                     aggfunc='mean')
    pivot_df.columns = [f'group_{g}_mean' for g in pivot_df.columns]
    pivot_df['difference'] = pivot_df['group_1_mean'] - pivot_df['group_2_mean']
    pivot_df.to_csv(output_path / 'minute_comparison.csv')
    console.print(f"[green]✓ Saved minute comparison table[/green]")

def main():
    parser = argparse.ArgumentParser(
        description='Plot minute-by-minute preferences with group analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('csv_path', help='Path to CSV file with preference data')
    parser.add_argument('--save-dir', type=str, help='Directory to save plots and summaries')
    parser.add_argument('--bin-size', type=float, default=1.0,
                       help='Bin size in minutes (default: 1.0)')
    parser.add_argument('--fps', type=float, default=60.0,
                       help='Video frame rate for time conversion (default: 60)')
    parser.add_argument('--no-scatter', action='store_true',
                       help='Skip scatter plot')
    parser.add_argument('--no-bars', action='store_true',
                       help='Skip bar plot')
    parser.add_argument('--no-continuous', action='store_true',
                       help='Skip continuous line plot')
    parser.add_argument('--no-individual', action='store_true',
                       help='Skip individual fish plot')
    
    args = parser.parse_args()
    
    # Create save directory if specified
    if args.save_dir:
        save_path = Path(args.save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        console.print(f"[cyan]Saving outputs to {save_path}[/cyan]")
    
    # Load and process data
    df = load_and_process_data(args.csv_path, args.fps)
    
    # Bin data by minute
    console.print(f"\n[cyan]Binning data into {args.bin_size}-minute intervals...[/cyan]")
    binned_df = bin_data_by_minute(df, args.bin_size)
    
    # Print statistics
    print_temporal_statistics(binned_df)
    
    # Create plots
    if not args.no_scatter:
        console.print("\n[cyan]Creating scatter plot...[/cyan]")
        plot_minute_by_minute_scatter(df, binned_df, args.save_dir)
    
    if not args.no_bars:
        console.print("\n[cyan]Creating bar plot...[/cyan]")
        plot_minute_by_minute_bars(binned_df, args.save_dir)
    
    if not args.no_continuous:
        console.print("\n[cyan]Creating continuous plot...[/cyan]")
        plot_minute_by_minute_continuous(binned_df, args.save_dir)
    
    if not args.no_individual:
        console.print("\n[cyan]Creating individual fish plot...[/cyan]")
        plot_individual_fish_minutes(df, args.save_dir)
    
    # Save summaries if directory specified
    if args.save_dir:
        console.print("\n[cyan]Saving minute summaries...[/cyan]")
        save_minute_summaries(binned_df, args.save_dir)
    
    console.print("\n[green]✓ Analysis complete![/green]")

if __name__ == '__main__':
    main()