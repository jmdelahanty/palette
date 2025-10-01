#!/usr/bin/env python3
# src/fisheye/cli/pick_file.py
"""
Standalone file picker that can be run from command line.

Usage:
    python -m fisheye.cli.pick_file --type zarr
    python -m fisheye.cli.pick_file --type video
    python -m fisheye.cli.pick_file --type config
"""

import sys
import argparse
from pathlib import Path

try:
    from .file_browsers import pick_zarr_path, pick_video_path, pick_config_path
except ImportError:
    # If run as script, adjust path
    import os
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from fisheye.cli.file_browsers import pick_zarr_path, pick_video_path, pick_config_path


def main():
    parser = argparse.ArgumentParser(
        description="Interactive file picker for FishEye pipeline"
    )
    parser.add_argument(
        "--type",
        choices=["zarr", "video", "config"],
        required=True,
        help="Type of file to pick"
    )
    parser.add_argument(
        "--start-dir",
        default="~/Desktop",
        help="Starting directory for browser"
    )
    
    args = parser.parse_args()
    
    if args.type == "zarr":
        result = pick_zarr_path(args.start_dir)
    elif args.type == "video":
        result = pick_video_path(args.start_dir)
    elif args.type == "config":
        result = pick_config_path(args.start_dir)
    
    if result:
        print(result)
        return 0
    else:
        print("Selection cancelled", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())