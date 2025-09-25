#!/usr/bin/env python3
"""Standalone test that adds paths manually."""

import sys
from pathlib import Path

# Add the utils directory to path
sys.path.insert(0, str(Path(__file__).parent / "utils"))

# Now import directly
from system import (
    get_git_info,
    get_platform_info, 
    get_gpu_info,
    get_environment_info
)
import json

def main():
    print("=" * 60)
    print("Testing System Utilities (Standalone)")
    print("=" * 60)
    
    # Test git info
    print("\n1. Git Information:")
    try:
        git_info = get_git_info()
        print(json.dumps(git_info, indent=2))
    except Exception as e:
        print(f"  Error: {e}")
    
    # Test platform info
    print("\n2. Platform Information:")
    try:
        platform_info = get_platform_info(collect_ip=False, disk_path="/tmp")
        # Use default=str to handle non-serializable objects
        print(json.dumps({
            'hostname': platform_info.get('hostname'),
            'system': platform_info.get('system'),
            'cpu_cores': platform_info.get('cpu_cores'),
            'python_version': platform_info.get('python_version')
        }, indent=2))
    except Exception as e:
        print(f"  Error: {e}")
    
    # Test GPU info
    print("\n3. GPU Information:")
    try:
        gpu_info = get_gpu_info()
        print(f"  GPUs available: {gpu_info.get('available', False)}")
        if gpu_info.get('devices'):
            print(f"  Number of devices: {len(gpu_info['devices'])}")
    except Exception as e:
        print(f"  Error: {e}")
    
    print("\nTest complete!")

if __name__ == "__main__":
    main()