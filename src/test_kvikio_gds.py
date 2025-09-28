#!/usr/bin/env python3
"""
Test kvikIO GDS capabilities and configuration
"""

import os
import sys
import time
import numpy as np

# Force GDS mode before importing kvikio
os.environ["KVIKIO_COMPAT_MODE"] = "OFF"

print("=" * 60)
print("kvikIO GDS Capability Test")
print("=" * 60)

# Check kvikio
try:
    import kvikio
    print(f"✓ kvikio imported successfully")
    print(f"  Version: {getattr(kvikio, '__version__', 'unknown')}")
    print(f"  Location: {kvikio.__file__}")
    
    # List available attributes
    print("\nAvailable kvikio attributes:")
    attrs = [a for a in dir(kvikio) if not a.startswith('_')]
    for attr in sorted(attrs):
        print(f"  - {attr}")
    
    # Check for defaults in different ways
    if hasattr(kvikio, 'defaults'):
        print(f"\n✓ kvikio.defaults exists")
        if hasattr(kvikio.defaults, 'compat_mode'):
            compat = kvikio.defaults.compat_mode()
            print(f"  Compat mode: {compat}")
            if compat:
                print("  ⚠️  WARNING: Running in compatibility mode (NOT using GDS)")
        if hasattr(kvikio.defaults, 'get_num_threads'):
            print(f"  Threads: {kvikio.defaults.get_num_threads()}")
    else:
        print("\n⚠️  kvikio.defaults not found - checking alternatives...")
        
        # Try kvikio.config or other methods
        if hasattr(kvikio, 'config'):
            print("  Found kvikio.config")
            
        # Check if we can access compat_mode differently
        try:
            from kvikio import defaults
            print("  ✓ Imported defaults directly")
            compat = defaults.compat_mode()
            print(f"  Compat mode: {compat}")
        except ImportError:
            print("  ✗ Could not import defaults module")
    
except ImportError as e:
    print(f"✗ Failed to import kvikio: {e}")
    sys.exit(1)

# Check CuPy
print("\n" + "-" * 40)
try:
    import cupy as cp
    print(f"✓ CuPy imported successfully")
    print(f"  Version: {cp.__version__}")
    
    # Check CUDA
    cuda_version = cp.cuda.runtime.runtimeGetVersion()
    print(f"  CUDA Runtime: {cuda_version}")
    
    # Check device
    device = cp.cuda.Device()
    try:
        # Try different ways to get device name
        if hasattr(device, 'name'):
            gpu_name = device.name.decode() if isinstance(device.name, bytes) else device.name
        else:
            # Use device ID and properties
            gpu_name = f"Device {device.id}"
            if hasattr(cp.cuda.runtime, 'getDeviceProperties'):
                props = cp.cuda.runtime.getDeviceProperties(device.id)
                if hasattr(props, 'name'):
                    gpu_name = props['name'].decode() if isinstance(props['name'], bytes) else props['name']
        print(f"  GPU: {gpu_name}")
    except:
        print(f"  GPU: Device {device.id}")
    
    # Memory info
    try:
        mem_info = device.mem_info
        print(f"  Memory: {mem_info[1] / 1024**3:.1f} GB total, {mem_info[0] / 1024**3:.1f} GB free")
    except:
        print(f"  Memory: Unable to query")
    
except ImportError as e:
    print(f"✗ Failed to import CuPy: {e}")
    sys.exit(1)

# Test basic kvikio functionality
print("\n" + "-" * 40)
print("Testing kvikIO functionality...")

try:
    # Check if CuFile is available
    if hasattr(kvikio, 'CuFile'):
        print("✓ kvikio.CuFile is available")
    else:
        print("✗ kvikio.CuFile not found")
        sys.exit(1)
    
    # Try a simple write/read test
    test_file = "/tmp/kvikio_test.bin"
    test_size_mb = 100
    test_size = test_size_mb * 1024 * 1024
    
    # Make sure size is 4KiB aligned
    test_size = (test_size // 4096) * 4096
    
    print(f"\nTesting {test_size_mb} MB write/read to {test_file}")
    
    # Create test data
    data = cp.random.randint(0, 255, test_size, dtype=cp.uint8)
    
    # Test write
    print("Testing write...")
    t0 = time.perf_counter()
    try:
        with kvikio.CuFile(test_file, "w") as f:
            nbytes = f.write(data)
        cp.cuda.Stream.null.synchronize()
        dt_write = time.perf_counter() - t0
        
        if nbytes == test_size:
            speed_write = (test_size_mb / dt_write)
            print(f"  ✓ Write successful: {speed_write:.1f} MB/s")
            
            if speed_write > 1000:  # > 1 GB/s suggests GDS
                print(f"  ✓ HIGH SPEED - Likely using GDS!")
            elif speed_write > 100:
                print(f"  ⚠️  Moderate speed - Might be using compatibility mode")
            else:
                print(f"  ⚠️  Low speed - Check configuration")
        else:
            print(f"  ✗ Write size mismatch: {nbytes} != {test_size}")
            
    except Exception as e:
        print(f"  ✗ Write failed: {e}")
        print("\nPossible issues:")
        print("  1. KVIKIO_COMPAT_MODE might be overridden")
        print("  2. libcufile.so might not be installed")
        print("  3. GDS might not be supported on this system")
        print("  4. Target filesystem might not support GDS")
    
    # Test read
    print("Testing read...")
    data2 = cp.empty_like(data)
    t0 = time.perf_counter()
    try:
        with kvikio.CuFile(test_file, "r") as f:
            nbytes = f.read(data2)
        cp.cuda.Stream.null.synchronize()
        dt_read = time.perf_counter() - t0
        
        if nbytes == test_size:
            speed_read = (test_size_mb / dt_read)
            print(f"  ✓ Read successful: {speed_read:.1f} MB/s")
            
            if speed_read > 1000:  # > 1 GB/s suggests GDS
                print(f"  ✓ HIGH SPEED - Likely using GDS!")
            
            # Verify data
            if cp.array_equal(data, data2):
                print(f"  ✓ Data verification passed")
            else:
                print(f"  ✗ Data verification failed")
        else:
            print(f"  ✗ Read size mismatch: {nbytes} != {test_size}")
            
    except Exception as e:
        print(f"  ✗ Read failed: {e}")
    
    # Clean up
    try:
        os.remove(test_file)
        print(f"  ✓ Cleaned up test file")
    except:
        pass
        
except Exception as e:
    print(f"✗ Test failed: {e}")
    import traceback
    traceback.print_exc()

# Check for GDS tools
print("\n" + "-" * 40)
print("Checking for GDS tools...")

gds_paths = [
    "/usr/local/cuda/gds/tools/gdscheck",
    "/usr/local/cuda-12.0/gds/tools/gdscheck",
    "/usr/local/cuda-12.1/gds/tools/gdscheck",
    "/usr/local/cuda-12.2/gds/tools/gdscheck",
    "/usr/local/cuda-12.3/gds/tools/gdscheck",
    "/usr/local/cuda-12.4/gds/tools/gdscheck",
]

gds_found = False
for path in gds_paths:
    if os.path.exists(path):
        print(f"✓ Found GDS tools at: {path}")
        gds_found = True
        
        # Try to run gdscheck
        try:
            import subprocess
            result = subprocess.run([path, "-p"], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                print("  GDS check output:")
                for line in result.stdout.split('\n')[:10]:  # First 10 lines
                    if line.strip():
                        print(f"    {line}")
        except Exception as e:
            print(f"  Could not run gdscheck: {e}")
        break

if not gds_found:
    print("✗ GDS tools not found")
    print("  Install with: apt-get install libcufile-dev")
    print("  Or download from NVIDIA")

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

print("\nEnvironment variables:")
print(f"  KVIKIO_COMPAT_MODE: {os.environ.get('KVIKIO_COMPAT_MODE', 'not set')}")

print("\nRecommendations:")
if 'speed_write' in locals() and speed_write > 1000:
    print("✓ GDS appears to be working!")
else:
    print("⚠️  GDS may not be fully configured. Try:")
    print("  1. Install GDS: apt-get install libcufile-dev")
    print("  2. Set environment: export KVIKIO_COMPAT_MODE=OFF")
    print("  3. Check filesystem support (ext4, XFS)")
    print("  4. Verify CUDA and driver versions are compatible")