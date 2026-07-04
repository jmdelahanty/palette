#!/usr/bin/env python3
"""Test system utilities functionality."""

from fisheye.shared.system_metadata import (
    get_git_info, 
    get_platform_info, 
    get_gpu_info,
    get_environment_info
)
import json
from pathlib import Path

def test_system_utils():
    """Test all system utility functions."""
    
    print("=" * 60)
    print("Testing System Utilities")
    print("=" * 60)
    
    # Test git info
    print("\n1. Git Information:")
    git_info = get_git_info()
    print(json.dumps(git_info, indent=2))
    
    # Test platform info (without IP to avoid hangs)
    print("\n2. Platform Information:")
    platform_info = get_platform_info(collect_ip=False, disk_path="/tmp")
    print(json.dumps(platform_info, indent=2, default=str))
    
    # Test GPU info
    print("\n3. GPU Information:")
    gpu_info = get_gpu_info()
    print(json.dumps(gpu_info, indent=2))
    
    # Test complete environment info
    print("\n4. Complete Environment:")
    env_info = get_environment_info(
        include_all_packages=False,
        disk_path="/tmp",
        collect_ip=False
    )
    
    # Print summary
    print(f"Environment type: {env_info['environment']['environment_type']}")
    print(f"Python version: {env_info['environment']['python_version']}")
    print(f"Total packages: {env_info['environment']['total_packages']}")
    
    if env_info['platform'].get('lsf'):
        print("\nRunning on LSF cluster:")
        print(json.dumps(env_info['platform']['lsf'], indent=2))
    
    return env_info

if __name__ == "__main__":
    test_system_utils()