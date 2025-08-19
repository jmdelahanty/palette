# src/plot_trajectory.py

import pandas as pd
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import os
import cv2 

def plot_single_trajectory(csv_path):
    """
    Reads a single CSV file and plots its 2D trajectory over the background image.
    """
    csv_path = Path(csv_path)
    output_path = csv_path.with_suffix('.png')
    background_image_path = csv_path.parent / "background_model.png"

    print(f"📄 Loading data from: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"❌ Error: File not found at {csv_path}")
        return
        
    if df.empty:
        print("⚠️ Warning: The CSV file is empty. No trajectory to plot.")
        return

    fish_id = df['id'][0]

    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(10, 10))
    
    img_width, img_height = 1, 1
    x_col, y_col = 'center_x_norm', 'center_y_norm'
    
    if background_image_path.exists():
        bg_img = plt.imread(background_image_path)
        img_height, img_width = bg_img.shape[:2]
        ax.imshow(bg_img, cmap='gray', extent=[0, img_width, img_height, 0])
        df['center_x_px'] = df['center_x_norm'] * img_width
        df['center_y_px'] = df['center_y_norm'] * img_height
        x_col, y_col = 'center_x_px', 'center_y_px'
        ax.set_xlim(0, img_width)
        ax.set_ylim(img_height, 0)
    else:
        print("⚠️  background_model.png not found. Plotting on a blank canvas.")
        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0)

    ax.plot(df[x_col], df[y_col], 
            marker='o', linestyle='-', markersize=3, alpha=0.6,
            label=f'Fish ID {fish_id} Trajectory')

    ax.plot(df[x_col].iloc[0], df[y_col].iloc[0], 
            'o', color='lime', markersize=10, markeredgecolor='black', label='Start')
    ax.plot(df[x_col].iloc[-1], df[y_col].iloc[-1], 
            'o', color='red', markersize=10, markeredgecolor='black', label='End')

    ax.set_title(f"Trajectory for Fish ID: {fish_id}", fontsize=16)
    ax.set_xlabel("X Coordinate (pixels)" if background_image_path.exists() else "X Coordinate (Normalized)")
    ax.set_ylabel("Y Coordinate (pixels)" if background_image_path.exists() else "Y Coordinate (Normalized)")
    ax.set_aspect('equal', adjustable='box')
    ax.legend()
    
    plt.savefig(output_path, dpi=300)
    print(f"✅ Plot saved to: {output_path}")
    
    plt.show()

def plot_all_trajectories(directory_path):
    """
    Finds all CSV files in a directory and plots all trajectories on a single graph.
    """
    directory_path = Path(directory_path)
    output_path = directory_path / "all_trajectories_plot.png"
    background_image_path = directory_path / "background_model.png"

    csv_files = sorted(list(directory_path.glob("fish_id_*.csv")))
    
    if not csv_files:
        print(f"❌ Error: No 'fish_id_*.csv' files found in {directory_path}")
        return

    print(f"📊 Found {len(csv_files)} trajectory files. Plotting all...")

    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(12, 12))
    
    img_width, img_height = 1, 1
    x_col, y_col = 'center_x_norm', 'center_y_norm'

    if background_image_path.exists():
        bg_img = plt.imread(background_image_path)
        img_height, img_width = bg_img.shape[:2]
        ax.imshow(bg_img, cmap='gray', extent=[0, img_width, img_height, 0])
        x_col, y_col = 'center_x_px', 'center_y_px'
        ax.set_xlim(0, img_width)
        ax.set_ylim(img_height, 0)
    else:
        print("⚠️  background_model.png not found. Plotting on a blank canvas.")
        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0)

    colors = plt.cm.get_cmap('tab20', len(csv_files))

    for i, csv_file in enumerate(csv_files):
        df = pd.read_csv(csv_file)
        if not df.empty:
            if background_image_path.exists():
                df['center_x_px'] = df['center_x_norm'] * img_width
                df['center_y_px'] = df['center_y_norm'] * img_height

            fish_id = df['id'][0]
            ax.plot(df[x_col], df[y_col], 
                    color=colors(i),
                    alpha=0.8,
                    label=f'ID {fish_id}')
            
            ax.plot(df[x_col].iloc[0], df[y_col].iloc[0], 
                    'o', color=colors(i), markersize=8, markeredgecolor='white', label='_nolegend_')
            ax.plot(df[x_col].iloc[-1], df[y_col].iloc[-1], 
                    'X', color=colors(i), markersize=10, markeredgecolor='white', label='_nolegend_')

    # --- NEW: Create custom legend entries for start/end markers ---
    # These are "dummy" plots with no data, just for creating the legend handles
    ax.plot([], [], 'o', color='gray', markersize=8, markeredgecolor='white', label='Start Point')
    ax.plot([], [], 'X', color='gray', markersize=10, markeredgecolor='white', label='End Point')
    # --- End of new section ---

    ax.set_title("All Fish Trajectories", fontsize=18)
    ax.set_xlabel("X Coordinate (pixels)" if background_image_path.exists() else "X Coordinate (Normalized)")
    ax.set_ylabel("Y Coordinate (pixels)" if background_image_path.exists() else "Y Coordinate (Normalized)")
    ax.set_aspect('equal', adjustable='box')
    ax.legend(title="Legend", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(output_path, dpi=300)
    print(f"✅ Combined plot saved to: {output_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot fish trajectories from CSV files.")
    parser.add_argument("input_path", type=str, 
                        help="Path to a single CSV file or a directory containing multiple CSV files.")
    args = parser.parse_args()

    input_path = Path(args.input_path)

    if not input_path.exists():
        print(f"❌ Error: Input path does not exist: {input_path}")
        return

    if input_path.is_dir():
        plot_all_trajectories(input_path)
    elif input_path.is_file() and input_path.suffix == '.csv':
        plot_single_trajectory(input_path)
    else:
        print(f"❌ Error: Path is not a valid CSV file or directory.")


if __name__ == "__main__":
    main()