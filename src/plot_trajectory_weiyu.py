#!/usr/bin/env python3
"""
Plot fish trajectories from exported CSV files with support for multiple coordinate systems.
Supports both individual and combined trajectory plots with sub-dish overlays.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import argparse
from pathlib import Path
import yaml
from typing import Optional, List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

def load_metadata(directory: Path) -> Optional[Dict]:
    """Load export metadata if available."""
    metadata_path = directory / 'export_metadata.yaml'
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            return yaml.safe_load(f)
    return None

def plot_single_trajectory(csv_path: str, 
                         coordinate_system: str = 'auto',
                         show_interpolated: bool = True,
                         no_interpolated: bool = False,
                         show_subdish: bool = True,
                         output_path: Optional[str] = None):
    """
    Plot a single fish trajectory from CSV file.
    
    Args:
        csv_path: Path to CSV file
        coordinate_system: '640', 'full', 'norm', or 'auto'
        show_interpolated: Show interpolated points differently
        no_interpolated: Filter out interpolated detections completely
        show_subdish: Show sub-dish boundaries if available
        output_path: Custom output path (default: same as CSV with .png)
    """
    csv_path = Path(csv_path)
    if output_path is None:
        output_path = csv_path.with_suffix('.png')
    else:
        output_path = Path(output_path)
    
    background_path = csv_path.parent / "background_model.png"
    
    print(f"📄 Loading data from: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"❌ Error: File not found at {csv_path}")
        return
    
    if df.empty:
        print("⚠️ Warning: CSV file is empty")
        return
    
    # Extract fish ID from filename
    fish_id = int(csv_path.stem.split('_')[1]) if 'fish_' in csv_path.stem else 0
    
    # Determine coordinate system
    if coordinate_system == 'auto':
        if background_path.exists():
            coordinate_system = '640'
        else:
            coordinate_system = 'norm'
    
    # Set up the plot
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Load and display background if available
    if background_path.exists() and coordinate_system == '640':
        bg_img = plt.imread(background_path)
        img_height, img_width = bg_img.shape[:2]
        ax.imshow(bg_img, cmap='gray', extent=[0, img_width, img_height, 0])
        ax.set_xlim(0, img_width)
        ax.set_ylim(img_height, 0)
    
    # Select coordinate columns
    if coordinate_system == '640':
        x_col, y_col = 'center_x_640', 'center_y_640'
        x_label, y_label = 'X (pixels)', 'Y (pixels)'
        if coordinate_system == '640' and not background_path.exists():
            ax.set_xlim(0, 640)
            ax.set_ylim(640, 0)
    elif coordinate_system == 'full':
        x_col, y_col = 'center_x_full', 'center_y_full'
        x_label, y_label = 'X (pixels)', 'Y (pixels)'
        ax.set_xlim(0, 4512)
        ax.set_ylim(4512, 0)
    else:  # norm
        x_col, y_col = 'center_x_norm', 'center_y_norm'
        x_label, y_label = 'X (normalized)', 'Y (normalized)'
        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0)
    
    # Filter out interpolated if requested
    if no_interpolated and 'detection_type' in df.columns:
        df = df[df['detection_type'] == 'original'].copy()
        if df.empty:
            print("⚠️ Warning: No original detections found after filtering")
            return
    
    # Separate original and interpolated detections if available
    if 'detection_type' in df.columns and show_interpolated and not no_interpolated:
        orig_df = df[df['detection_type'] == 'original']
        interp_df = df[df['detection_type'] == 'interpolated']
        
        # Plot interpolated as dotted line
        if not interp_df.empty:
            ax.plot(interp_df[x_col], interp_df[y_col],
                   'o', color='lightblue', markersize=2, alpha=0.3,
                   label='Interpolated')
        
        # Plot original as solid line
        if not orig_df.empty:
            ax.plot(orig_df[x_col], orig_df[y_col],
                   marker='o', linestyle='-', markersize=3, 
                   color='darkblue', alpha=0.7,
                   label=f'Fish {fish_id} Original')
    else:
        # Plot all as single trajectory
        ax.plot(df[x_col], df[y_col],
               marker='o', linestyle='-', markersize=3, alpha=0.6,
               label=f'Fish {fish_id} Trajectory')
    
    # Mark start and end points
    ax.plot(df[x_col].iloc[0], df[y_col].iloc[0],
           'o', color='lime', markersize=12, 
           markeredgecolor='black', markeredgewidth=2, label='Start')
    ax.plot(df[x_col].iloc[-1], df[y_col].iloc[-1],
           's', color='red', markersize=12,
           markeredgecolor='black', markeredgewidth=2, label='End')
    
    # Add sub-dish boundary if available
    if show_subdish and 'subdish_x' in df.columns and coordinate_system == '640':
        x, y = df['subdish_x'].iloc[0], df['subdish_y'].iloc[0]
        w, h = df['subdish_w'].iloc[0], df['subdish_h'].iloc[0]
        rect = patches.Rectangle((x, y), w, h, linewidth=2,
                                edgecolor='yellow', facecolor='none',
                                linestyle='--', label='Sub-dish ROI')
        ax.add_patch(rect)
    
    # Add statistics text box
    total_frames = df['frame'].max() - df['frame'].min() + 1
    detection_rate = len(df) / total_frames * 100
    
    stats_text = f"Frames: {df['frame'].min()}-{df['frame'].max()}\n"
    stats_text += f"Detections: {len(df)}/{total_frames} ({detection_rate:.1f}%)"
    
    if 'detection_type' in df.columns:
        orig_count = len(df[df['detection_type'] == 'original'])
        interp_count = len(df[df['detection_type'] == 'interpolated'])
        stats_text += f"\nOriginal: {orig_count}, Interpolated: {interp_count}"
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.set_title(f"Trajectory for Fish {fish_id}", fontsize=16, fontweight='bold')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_aspect('equal', adjustable='box')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Plot saved to: {output_path}")
    plt.show()

def plot_all_trajectories(directory: str,
                        coordinate_system: str = 'auto',
                        show_interpolated: bool = False,
                        no_interpolated: bool = False,
                        show_subdish: bool = True,
                        output_path: Optional[str] = None):
    """
    Plot all trajectories in a directory on a single figure.
    
    Args:
        directory: Directory containing CSV files
        coordinate_system: '640', 'full', 'norm', or 'auto'
        show_interpolated: Show interpolated points differently
        no_interpolated: Filter out interpolated detections completely
        show_subdish: Show sub-dish boundaries
        output_path: Custom output path
    """
    directory = Path(directory)
    if output_path is None:
        output_path = directory / "all_trajectories_plot.png"
    else:
        output_path = Path(output_path)
    
    background_path = directory / "background_model.png"
    
    # Find all fish CSV files (new naming pattern)
    csv_files = sorted(list(directory.glob("fish_*_tracking.csv")))
    if not csv_files:
        # Try old naming pattern
        csv_files = sorted(list(directory.glob("fish_id_*.csv")))
    
    if not csv_files:
        print(f"❌ No fish tracking CSV files found in {directory}")
        return
    
    print(f"📊 Found {len(csv_files)} trajectory files")
    
    # Load metadata if available
    metadata = load_metadata(directory)
    
    # Set up the plot
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Determine coordinate system
    if coordinate_system == 'auto':
        if background_path.exists():
            coordinate_system = '640'
        else:
            coordinate_system = 'norm'
    
    # Load and display background
    if background_path.exists() and coordinate_system == '640':
        bg_img = plt.imread(background_path)
        img_height, img_width = bg_img.shape[:2]
        ax.imshow(bg_img, cmap='gray', extent=[0, img_width, img_height, 0], alpha=0.7)
        ax.set_xlim(0, img_width)
        ax.set_ylim(img_height, 0)
    elif coordinate_system == '640':
        ax.set_xlim(0, 640)
        ax.set_ylim(640, 0)
    elif coordinate_system == 'full':
        ax.set_xlim(0, 4512)
        ax.set_ylim(4512, 0)
    else:
        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0)
    
    # Select coordinate columns
    if coordinate_system == '640':
        x_col, y_col = 'center_x_640', 'center_y_640'
        x_label, y_label = 'X (pixels)', 'Y (pixels)'
    elif coordinate_system == 'full':
        x_col, y_col = 'center_x_full', 'center_y_full'
        x_label, y_label = 'X (pixels)', 'Y (pixels)'
    else:
        x_col, y_col = 'center_x_norm', 'center_y_norm'
        x_label, y_label = 'X (normalized)', 'Y (normalized)'
    
    # Use a colormap for different fish
    colors = plt.cm.get_cmap('tab20', len(csv_files))
    
    # Track sub-dish ROIs
    subdish_rois = {}
    
    for i, csv_file in enumerate(csv_files):
        df = pd.read_csv(csv_file)
        if df.empty:
            continue
        
        # Extract fish ID
        if 'fish_' in csv_file.stem:
            parts = csv_file.stem.split('_')
            fish_id = int(parts[1])
        else:
            fish_id = i
        
        color = colors(i)
        
        # Filter out interpolated if requested
        if no_interpolated and 'detection_type' in df.columns:
            df = df[df['detection_type'] == 'original'].copy()
        
        # Plot trajectory
        if 'detection_type' in df.columns and show_interpolated and not no_interpolated:
            orig_df = df[df['detection_type'] == 'original']
            interp_df = df[df['detection_type'] == 'interpolated']
            
            if not interp_df.empty:
                ax.plot(interp_df[x_col], interp_df[y_col],
                       'o', color=color, markersize=1, alpha=0.2)
            
            if not orig_df.empty:
                ax.plot(orig_df[x_col], orig_df[y_col],
                       color=color, linewidth=1.5, alpha=0.7,
                       label=f'Fish {fish_id}')
        else:
            # Use only original detections if available
            if 'detection_type' in df.columns:
                df = df[df['detection_type'] == 'original']
            
            ax.plot(df[x_col], df[y_col],
                   color=color, linewidth=1.5, alpha=0.7,
                   label=f'Fish {fish_id}')
        
        # Mark start and end
        ax.plot(df[x_col].iloc[0], df[y_col].iloc[0],
               'o', color=color, markersize=8,
               markeredgecolor='white', markeredgewidth=1.5)
        ax.plot(df[x_col].iloc[-1], df[y_col].iloc[-1],
               's', color=color, markersize=8,
               markeredgecolor='white', markeredgewidth=1.5)
        
        # Collect sub-dish info
        if 'subdish_x' in df.columns and fish_id not in subdish_rois:
            subdish_rois[fish_id] = (
                df['subdish_x'].iloc[0], df['subdish_y'].iloc[0],
                df['subdish_w'].iloc[0], df['subdish_h'].iloc[0]
            )
    
    # Draw sub-dish boundaries
    if show_subdish and subdish_rois and coordinate_system == '640':
        for fish_id, (x, y, w, h) in subdish_rois.items():
            rect = patches.Rectangle((x, y), w, h, linewidth=1.5,
                                    edgecolor='cyan', facecolor='none',
                                    linestyle='--', alpha=0.8)
            ax.add_patch(rect)
            # Add label
            ax.text(x + w/2, y + h + 5, f'Dish {fish_id}',
                   ha='center', fontsize=9, color='cyan',
                   bbox=dict(boxstyle='round,pad=0.3', 
                            facecolor='black', alpha=0.7))
    
    # Add legend markers for start/end
    ax.plot([], [], 'o', color='gray', markersize=8,
           markeredgecolor='white', markeredgewidth=1.5, label='Start')
    ax.plot([], [], 's', color='gray', markersize=8,
           markeredgecolor='white', markeredgewidth=1.5, label='End')
    
    ax.set_title("All Fish Trajectories", fontsize=18, fontweight='bold')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_aspect('equal', adjustable='box')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Combined plot saved to: {output_path}")
    plt.show()

def plot_trajectory_grid(directory: str,
                        coordinate_system: str = '640',
                        no_interpolated: bool = False,
                        output_path: Optional[str] = None):
    """
    Create a grid of subplots, one for each fish trajectory.
    
    Args:
        directory: Directory containing CSV files
        coordinate_system: Coordinate system to use
        no_interpolated: Filter out interpolated detections
        output_path: Custom output path
    """
    directory = Path(directory)
    if output_path is None:
        output_path = directory / "trajectory_grid.png"
    else:
        output_path = Path(output_path)
    
    background_path = directory / "background_model.png"
    
    # Find all CSV files
    csv_files = sorted(list(directory.glob("fish_*_tracking.csv")))
    if not csv_files:
        csv_files = sorted(list(directory.glob("fish_id_*.csv")))
    
    if not csv_files:
        print(f"❌ No fish tracking CSV files found")
        return
    
    n_fish = len(csv_files)
    n_cols = int(np.ceil(np.sqrt(n_fish)))
    n_rows = int(np.ceil(n_fish / n_cols))
    
    # Load background if available
    bg_img = None
    if background_path.exists() and coordinate_system == '640':
        bg_img = plt.imread(background_path)
    
    # Create grid
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
    if n_fish == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Select coordinate columns
    if coordinate_system == '640':
        x_col, y_col = 'center_x_640', 'center_y_640'
        xlim, ylim = (0, 640), (640, 0)
    elif coordinate_system == 'full':
        x_col, y_col = 'center_x_full', 'center_y_full'
        xlim, ylim = (0, 4512), (4512, 0)
    else:
        x_col, y_col = 'center_x_norm', 'center_y_norm'
        xlim, ylim = (0, 1), (1, 0)
    
    for i, csv_file in enumerate(csv_files):
        ax = axes[i]
        df = pd.read_csv(csv_file)
        
        if df.empty:
            ax.axis('off')
            continue
        
        # Extract fish ID
        if 'fish_' in csv_file.stem:
            fish_id = int(csv_file.stem.split('_')[1])
        else:
            fish_id = i
        
        # Show background
        if bg_img is not None:
            ax.imshow(bg_img, cmap='gray', extent=[0, 640, 640, 0], alpha=0.5)
        
        # Filter out interpolated if requested
        if no_interpolated and 'detection_type' in df.columns:
            df = df[df['detection_type'] == 'original'].copy()
        elif 'detection_type' in df.columns and not no_interpolated:
            # Default to original if not showing interpolated
            df = df[df['detection_type'] == 'original']
        
        ax.plot(df[x_col], df[y_col], linewidth=1, alpha=0.8)
        ax.plot(df[x_col].iloc[0], df[y_col].iloc[0], 'go', markersize=6)
        ax.plot(df[x_col].iloc[-1], df[y_col].iloc[-1], 'ro', markersize=6)
        
        # Add sub-dish boundary
        if 'subdish_x' in df.columns and coordinate_system == '640':
            x, y = df['subdish_x'].iloc[0], df['subdish_y'].iloc[0]
            w, h = df['subdish_w'].iloc[0], df['subdish_h'].iloc[0]
            rect = patches.Rectangle((x, y), w, h, linewidth=1,
                                    edgecolor='yellow', facecolor='none',
                                    linestyle='--')
            ax.add_patch(rect)
        
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect('equal')
        ax.set_title(f'Fish {fish_id}', fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Hide unused subplots
    for i in range(n_fish, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Individual Fish Trajectories', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Grid plot saved to: {output_path}")
    plt.show()

def main():
    parser = argparse.ArgumentParser(
        description="Plot fish trajectories from exported CSV files"
    )
    parser.add_argument("input_path", type=str,
                       help="Path to CSV file or directory with CSV files")
    parser.add_argument("-c", "--coords", type=str, default="auto",
                       choices=['640', 'full', 'norm', 'auto'],
                       help="Coordinate system to use (default: auto)")
    parser.add_argument("-o", "--output", type=str, default=None,
                       help="Custom output path")
    parser.add_argument("--show-interpolated", action="store_true",
                       help="Show interpolated detections differently")
    parser.add_argument("--no-interpolated", action="store_true",
                       help="Filter out interpolated detections completely")
    parser.add_argument("--no-subdish", action="store_true",
                       help="Don't show sub-dish boundaries")
    parser.add_argument("--grid", action="store_true",
                       help="Create grid plot for multiple trajectories")
    
    args = parser.parse_args()
    
    input_path = Path(args.input_path)
    
    if not input_path.exists():
        print(f"❌ Error: Path does not exist: {input_path}")
        return
    
    if input_path.is_dir():
        if args.grid:
            plot_trajectory_grid(
                str(input_path),
                coordinate_system=args.coords,
                no_interpolated=args.no_interpolated,
                output_path=args.output
            )
        else:
            plot_all_trajectories(
                str(input_path),
                coordinate_system=args.coords,
                show_interpolated=args.show_interpolated,
                no_interpolated=args.no_interpolated,
                show_subdish=not args.no_subdish,
                output_path=args.output
            )
    elif input_path.is_file() and input_path.suffix == '.csv':
        plot_single_trajectory(
            str(input_path),
            coordinate_system=args.coords,
            show_interpolated=args.show_interpolated,
            show_subdish=not args.no_subdish,
            output_path=args.output
        )
    else:
        print(f"❌ Error: Invalid input path")

if __name__ == "__main__":
    main()