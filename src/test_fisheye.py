#!/usr/bin/env python3
"""Test FishEye components."""

import sys
from pathlib import Path

# Ensure src is in the path
src_dir = Path(__file__).parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

def test_utils():
    """Test utility imports and basic functionality."""
    from fisheye.utils import (
        get_git_info,
        get_platform_info,
        get_gpu_info,
        get_environment_info
    )
    
    print("Testing FishEye Utils")
    print("=" * 50)
    
    # Platform info
    platform_info = get_platform_info(collect_ip=False)
    print(f"✓ Hostname: {platform_info['hostname']}")
    print(f"✓ System: {platform_info['system']}")
    print(f"✓ CPU cores: {platform_info['cpu_cores']}")
    
    # GPU info
    gpu_info = get_gpu_info()
    print(f"✓ GPU available: {gpu_info['available']}")
    if gpu_info['available']:
        print(f"  - Devices: {len(gpu_info.get('devices', []))}")
        if gpu_info.get('devices'):
            print(f"  - GPU 0: {gpu_info['devices'][0].get('name', 'Unknown')}")
    
    # Git info
    git_info = get_git_info()
    print(f"✓ Git branch: {git_info.get('branch', 'Unknown')}")
    
    print("\nAll utils tests passed!")
    return True

if __name__ == "__main__":
    # Run tests
    utils_ok = test_utils()
    
    if utils_ok:
        print("\n✅ Core utilities working!")
    
    sys.exit(0 if utils_ok else 1)
