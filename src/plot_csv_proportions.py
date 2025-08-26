#!/usr/bin/env python3
"""
CSV Trial Proportion Plotter

Standalone script to visualize fish quadrant proportion data from CSV files.
Works with CSV output from multi_roi_grating_analyzer.py --save-proportions
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path
from typing import Optional
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_palette("husl")
plt.style.use('seaborn-v0_8-darkgrid')


def load_and_validate_csv(csv_path: str) -> pd.DataFrame:
    """Load CSV and validate required columns."""
    df = pd.read_csv(csv_path)
    
    required_cols = ['roi_id', 'trial_number', 'trial_type', 'top_proportion']
    missing = [col for col in required_cols if col not in df.columns]
    
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    print(f"Loaded {len(df)} records from {len(df['roi_id'].unique())} fish")
    print(f"Trial types: {df['trial_type'].unique()}")
    
    if 'orientation' in df.columns:
        print(f"Orientations found: {sorted(df['orientation'].dropna().unique())}")
    
    return df


def plot_proportion_by_stimulus(df: pd.DataFrame, save_path: Optional[str] = None):
    """Plot top proportion distributions by stimulus type."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    stimulus_colors = {
        'grating': '#FF6B6B',
        'black': '#4ECDC4',
        'white': '#95E77E',
        'unknown': '#FFE66D'
    }
    
    # 1. Boxplot by stimulus
    ax = axes[0]
    trial_types = df['trial_type'].unique()
    for i, trial_type in enumerate(trial_types):
        data = df[df['trial_type'] == trial_type]['top_proportion']
        bp = ax.boxplot([data], positions=[i], widths=0.6,
                       patch_artist=True, showmeans=True)
        bp['boxes'][0].set_facecolor(stimulus_colors.get(trial_type, 'gray'))
        bp['boxes'][0].set_alpha(0.7)
    
    ax.set_xticks(range(len(trial_types)))
    ax.set_xticklabels(trial_types)
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
    ax.set_ylabel('Top Proportion')
    ax.set_xlabel('Stimulus Type')
    ax.set_title('Distribution by Stimulus Type')
    ax.set_ylim([0, 1])
    
    # 2. Individual fish means
    ax = axes[1]
    fish_means = df.groupby(['roi_id', 'trial_type'])['top_proportion'].mean().reset_index()
    fish_pivot = fish_means.pivot(index='roi_id', columns='trial_type', values='top_proportion')
    
    x = np.arange(len(fish_pivot.index))
    width = 0.25
    
    for i, trial_type in enumerate(fish_pivot.columns):
        offset = (i - len(fish_pivot.columns)/2) * width + width/2
        ax.bar(x + offset, fish_pivot[trial_type], width, 
               label=trial_type, color=stimulus_colors.get(trial_type, 'gray'), alpha=0.7)
    
    ax.set_xlabel('Fish ID')
    ax.set_ylabel('Mean Top Proportion')
    ax.set_title('Individual Fish Responses')
    ax.set_xticks(x)
    ax.set_xticklabels(fish_pivot.index)
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
    ax.legend()
    ax.set_ylim([0, 1])
    
    # 3. Time series
    ax = axes[2]
    if 'time_minutes' in df.columns:
        for trial_type in df['trial_type'].unique():
            type_df = df[df['trial_type'] == trial_type]
            mean_by_trial = type_df.groupby('trial_number')['top_proportion'].mean()
            mean_time = type_df.groupby('trial_number')['time_minutes'].mean()
            ax.plot(mean_time, mean_by_trial, marker='o', 
                   label=trial_type, color=stimulus_colors.get(trial_type, 'gray'),
                   linewidth=2, markersize=6)
        ax.set_xlabel('Time (minutes)')
    else:
        for trial_type in df['trial_type'].unique():
            type_df = df[df['trial_type'] == trial_type]
            mean_by_trial = type_df.groupby('trial_number')['top_proportion'].mean()
            ax.plot(mean_by_trial.index, mean_by_trial.values, marker='o',
                   label=trial_type, color=stimulus_colors.get(trial_type, 'gray'),
                   linewidth=2, markersize=6)
        ax.set_xlabel('Trial Number')
    
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
    ax.set_ylabel('Mean Top Proportion')
    ax.set_title('Temporal Changes')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    plt.suptitle('Quadrant Proportion Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved plot to {save_path}")
    
    plt.show()


def plot_orientation_analysis(df: pd.DataFrame, save_path: Optional[str] = None):
    """Plot proportion analysis by grating orientation."""
    if 'orientation' not in df.columns:
        print("No orientation data in CSV")
        return
    
    # Filter to grating trials with orientation
    grating_df = df[(df['trial_type'] == 'grating') & (df['orientation'].notna())].copy()
    
    if grating_df.empty:
        print("No grating trials with orientation data")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Color map for orientations
    unique_orientations = sorted(grating_df['orientation'].unique())
    colors = plt.cm.hsv(np.linspace(0, 1, len(unique_orientations) + 1)[:-1])
    orientation_colors = dict(zip(unique_orientations, colors))
    
    # 1. Top proportion by orientation
    ax = axes[0, 0]
    for orientation in unique_orientations:
        data = grating_df[grating_df['orientation'] == orientation]['top_proportion']
        bp = ax.boxplot([data], positions=[orientation], widths=10,
                       patch_artist=True, showmeans=True)
        bp['boxes'][0].set_facecolor(orientation_colors[orientation])
        bp['boxes'][0].set_alpha(0.7)
    
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Grating Orientation (degrees)')
    ax.set_ylabel('Top Proportion')
    ax.set_title('Distribution by Orientation')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    
    # 2. Polar plot
    ax = plt.subplot(2, 2, 2, projection='polar')
    orientation_means = grating_df.groupby('orientation')['top_proportion'].agg(['mean', 'std'])
    
    for orientation in unique_orientations:
        if orientation in orientation_means.index:
            angle_rad = np.deg2rad(orientation)
            mean_val = orientation_means.loc[orientation, 'mean']
            std_val = orientation_means.loc[orientation, 'std']
            
            # Plot as deviation from 0.5 (no preference)
            deviation = mean_val - 0.5
            ax.bar(angle_rad, abs(deviation), width=np.deg2rad(15),
                  bottom=0.5 if deviation > 0 else 0.5 + deviation,
                  color=orientation_colors[orientation], alpha=0.7,
                  edgecolor='black', linewidth=1)
    
    ax.set_theta_zero_location('E')
    ax.set_theta_direction(-1)
    ax.set_ylim([0, 1])
    ax.set_title('Directional Preference (0.5 = no preference)', pad=20)
    
    # 3. Individual fish tuning curves
    ax = axes[1, 0]
    fish_orientation_means = grating_df.groupby(['roi_id', 'orientation'])['top_proportion'].mean().reset_index()
    
    for roi_id in sorted(fish_orientation_means['roi_id'].unique())[:10]:  # Limit to 10 fish
        fish_data = fish_orientation_means[fish_orientation_means['roi_id'] == roi_id]
        ax.plot(fish_data['orientation'], fish_data['top_proportion'],
               marker='o', alpha=0.5, label=f'Fish {int(roi_id)}')
    
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Grating Orientation (degrees)')
    ax.set_ylabel('Top Proportion')
    ax.set_title('Individual Fish Tuning')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    if len(fish_orientation_means['roi_id'].unique()) <= 10:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # 4. Heatmap
    ax = axes[1, 1]
    pivot_data = grating_df.pivot_table(values='top_proportion',
                                        index='roi_id',
                                        columns='orientation',
                                        aggfunc='mean')
    
    im = ax.imshow(pivot_data, cmap='RdBu_r', vmin=0, vmax=1, aspect='auto')
    ax.set_xlabel('Orientation (degrees)')
    ax.set_ylabel('Fish ID')
    ax.set_title('All Fish × All Orientations')
    ax.set_xticks(range(len(pivot_data.columns)))
    ax.set_xticklabels([f'{o:.0f}°' for o in pivot_data.columns])
    ax.set_yticks(range(len(pivot_data.index)))
    ax.set_yticklabels(pivot_data.index)
    plt.colorbar(im, ax=ax, label='Top Proportion')
    
    plt.suptitle('Orientation-Specific Quadrant Preferences', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved orientation plot to {save_path}")
    
    plt.show()


def calculate_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate summary statistics."""
    stats = []
    
    # Overall statistics
    stats.append({
        'Category': 'Overall',
        'Subcategory': 'All trials',
        'Mean': df['top_proportion'].mean(),
        'Std': df['top_proportion'].std(),
        'Median': df['top_proportion'].median(),
        'N': len(df)
    })
    
    # By stimulus type
    for trial_type in df['trial_type'].unique():
        type_df = df[df['trial_type'] == trial_type]
        stats.append({
            'Category': 'Stimulus',
            'Subcategory': trial_type,
            'Mean': type_df['top_proportion'].mean(),
            'Std': type_df['top_proportion'].std(),
            'Median': type_df['top_proportion'].median(),
            'N': len(type_df)
        })
    
    # By fish
    for roi_id in sorted(df['roi_id'].unique()):
        fish_df = df[df['roi_id'] == roi_id]
        stats.append({
            'Category': 'Fish',
            'Subcategory': f'Fish {int(roi_id)}',
            'Mean': fish_df['top_proportion'].mean(),
            'Std': fish_df['top_proportion'].std(),
            'Median': fish_df['top_proportion'].median(),
            'N': len(fish_df)
        })
    
    # By orientation if available
    if 'orientation' in df.columns:
        grating_df = df[(df['trial_type'] == 'grating') & (df['orientation'].notna())]
        for orientation in sorted(grating_df['orientation'].unique()):
            orient_df = grating_df[grating_df['orientation'] == orientation]
            stats.append({
                'Category': 'Orientation',
                'Subcategory': f'{orientation:.0f}°',
                'Mean': orient_df['top_proportion'].mean(),
                'Std': orient_df['top_proportion'].std(),
                'Median': orient_df['top_proportion'].median(),
                'N': len(orient_df)
            })
    
    return pd.DataFrame(stats)


def plot_individual_fish_analysis(df: pd.DataFrame, save_dir: Optional[str] = None):
    """Create individual analysis plots for each fish."""
    fish_ids = sorted(df['roi_id'].unique())
    n_fish = len(fish_ids)
    
    # Create directory if saving
    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
    
    for fish_id in fish_ids:
        fish_df = df[df['roi_id'] == fish_id]
        
        # Determine subplot layout based on available data
        has_orientation = 'orientation' in df.columns and fish_df['orientation'].notna().any()
        n_plots = 4 if has_orientation else 3
        
        fig, axes = plt.subplots(1, n_plots, figsize=(n_plots * 4, 4))
        
        # 1. Time series of proportion
        ax = axes[0]
        for trial_type in fish_df['trial_type'].unique():
            type_df = fish_df[fish_df['trial_type'] == trial_type]
            type_df = type_df.sort_values('trial_number')
            
            color = {'grating': '#FF6B6B', 'black': '#4ECDC4', 
                    'white': '#95E77E'}.get(trial_type, 'gray')
            
            if 'time_minutes' in df.columns:
                x_vals = type_df['time_minutes']
                x_label = 'Time (minutes)'
            else:
                x_vals = type_df['trial_number']
                x_label = 'Trial Number'
            
            ax.plot(x_vals, type_df['top_proportion'], marker='o',
                   label=trial_type, color=color, alpha=0.7, linewidth=2)
        
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
        ax.set_xlabel(x_label)
        ax.set_ylabel('Top Proportion')
        ax.set_title(f'Fish {int(fish_id)} - Time Series')
        ax.legend()
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
        
        # 2. Distribution by stimulus
        ax = axes[1]
        stimulus_types = fish_df['trial_type'].unique()
        positions = range(len(stimulus_types))
        
        for i, stim_type in enumerate(stimulus_types):
            data = fish_df[fish_df['trial_type'] == stim_type]['top_proportion']
            color = {'grating': '#FF6B6B', 'black': '#4ECDC4', 
                    'white': '#95E77E'}.get(stim_type, 'gray')
            
            # Violin plot with individual points
            parts = ax.violinplot([data], positions=[i], widths=0.5,
                                 showmeans=True, showmedians=True)
            for pc in parts['bodies']:
                pc.set_facecolor(color)
                pc.set_alpha(0.6)
            
            # Add individual points
            ax.scatter(np.full(len(data), i) + np.random.normal(0, 0.02, len(data)),
                      data, alpha=0.4, color='black', s=20)
        
        ax.set_xticks(positions)
        ax.set_xticklabels(stimulus_types)
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
        ax.set_ylabel('Top Proportion')
        ax.set_title('By Stimulus Type')
        ax.set_ylim([0, 1])
        
        # 3. Histogram of all proportions
        ax = axes[2]
        ax.hist(fish_df['top_proportion'], bins=20, range=(0, 1),
               color='steelblue', alpha=0.7, edgecolor='black')
        ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.5)
        ax.axvline(x=fish_df['top_proportion'].mean(), color='green',
                  linestyle='-', alpha=0.7, linewidth=2,
                  label=f'Mean: {fish_df["top_proportion"].mean():.2f}')
        ax.set_xlabel('Top Proportion')
        ax.set_ylabel('Count')
        ax.set_title('Overall Distribution')
        ax.legend()
        
        # 4. Orientation tuning (if available)
        if has_orientation:
            ax = axes[3]
            grating_df = fish_df[(fish_df['trial_type'] == 'grating') & 
                                 (fish_df['orientation'].notna())]
            
            if not grating_df.empty:
                orientation_means = grating_df.groupby('orientation')['top_proportion'].mean()
                orientation_stds = grating_df.groupby('orientation')['top_proportion'].std()
                
                ax.errorbar(orientation_means.index, orientation_means.values,
                          yerr=orientation_stds.values, marker='o', capsize=5,
                          linewidth=2, markersize=8, color='darkred')
                ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
                ax.set_xlabel('Orientation (degrees)')
                ax.set_ylabel('Mean Top Proportion')
                ax.set_title('Orientation Tuning')
                ax.set_ylim([0, 1])
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No orientation data', ha='center', va='center',
                       transform=ax.transAxes, fontsize=12, color='gray')
        
        # Add overall statistics text
        stats_text = f"Mean: {fish_df['top_proportion'].mean():.3f}\n"
        stats_text += f"Std: {fish_df['top_proportion'].std():.3f}\n"
        stats_text += f"N trials: {len(fish_df)}"
        
        fig.text(0.99, 0.99, stats_text, transform=fig.transFigure,
                ha='right', va='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle(f'Fish {int(fish_id)} Individual Analysis', 
                    fontsize=12, fontweight='bold')
        plt.tight_layout()
        
        if save_dir:
            save_file = save_path / f'fish_{int(fish_id)}_analysis.png'
            plt.savefig(save_file, dpi=150, bbox_inches='tight')
            print(f"✓ Saved Fish {int(fish_id)} plot to {save_file}")
            plt.close()
        else:
            plt.show()
    
    print(f"✓ Completed analysis for {n_fish} fish")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize fish quadrant proportion data from CSV',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('csv_path', help='Path to CSV file with proportion data')
    parser.add_argument('--plot-stimulus', action='store_true',
                       help='Create plots grouped by stimulus type')
    parser.add_argument('--plot-orientation', action='store_true',
                       help='Create plots for orientation analysis')
    parser.add_argument('--plot-individual', action='store_true',
                       help='Create individual plots for each fish')
    parser.add_argument('--save-stimulus', type=str,
                       help='Save stimulus plot to file')
    parser.add_argument('--save-orientation', type=str,
                       help='Save orientation plot to file')
    parser.add_argument('--save-individual', type=str,
                       help='Directory to save individual fish plots')
    parser.add_argument('--save-stats', type=str,
                       help='Save summary statistics to CSV')
    parser.add_argument('--print-stats', action='store_true',
                       help='Print summary statistics')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.csv_path}...")
    df = load_and_validate_csv(args.csv_path)
    
    # Calculate and optionally print/save statistics
    if args.print_stats or args.save_stats:
        stats_df = calculate_statistics(df)
        
        if args.print_stats:
            print("\n=== Summary Statistics ===")
            print(stats_df.to_string(index=False))
            print("\n=== Preference Summary ===")
            print(f"Fish preferring top (mean > 0.55): {sum(stats_df[(stats_df['Category']=='Fish') & (stats_df['Mean'] > 0.55)]['Subcategory'].notna())}")
            print(f"Fish preferring bottom (mean < 0.45): {sum(stats_df[(stats_df['Category']=='Fish') & (stats_df['Mean'] < 0.45)]['Subcategory'].notna())}")
            print(f"Fish with no clear preference: {sum(stats_df[(stats_df['Category']=='Fish') & (stats_df['Mean'] >= 0.45) & (stats_df['Mean'] <= 0.55)]['Subcategory'].notna())}")
        
        if args.save_stats:
            stats_df.to_csv(args.save_stats, index=False)
            print(f"✓ Statistics saved to {args.save_stats}")
    
    # Create plots
    if args.plot_stimulus:
        plot_proportion_by_stimulus(df, save_path=args.save_stimulus)
    
    if args.plot_orientation:
        plot_orientation_analysis(df, save_path=args.save_orientation)
    
    if args.plot_individual:
        plot_individual_fish_analysis(df, save_dir=args.save_individual)
    
    # Default: show stimulus plot if no specific plot requested
    if not any([args.plot_stimulus, args.plot_orientation, args.plot_individual, 
                args.print_stats, args.save_stats]):
        print("\nCreating default stimulus plot...")
        plot_proportion_by_stimulus(df)


if __name__ == '__main__':
    main()