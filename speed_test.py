import numpy as np
import time
import os
from pathlib import Path

# --- IMPORTANT: SET THIS TO THE SAME DIRECTORY AS YOUR ZARR OUTPUT ---
# Example: "/home/your_user/zarr_videos/"
OUTPUT_DIRECTORY = "~/Desktop/newone/"

# --- DO NOT CHANGE ANYTHING BELOW ---
MiB = 1024 * 1024
file_size_mib = 1024  # 1 GiB

# FIX 1: Expand the '~' to the full home directory path
output_path = Path(OUTPUT_DIRECTORY).expanduser()

# FIX 2: Create the directory if it doesn't exist
os.makedirs(output_path, exist_ok=True)

file_path = output_path / "speed_test.npy"


print(f"Creating a {file_size_mib} MiB file in memory...")
# Create a 1 GiB array of random bytes
data = np.random.randint(0, 256, size=(file_size_mib, MiB), dtype=np.uint8)

print(f"Writing 1 GiB file to: {file_path}")
t0 = time.perf_counter()
np.save(file_path, data)
t1 = time.perf_counter()

duration = t1 - t0
throughput = file_size_mib / duration

print("\n--- RESULTS ---")
print(f"Time to write: {duration:.2f} seconds")
print(f"Throughput: {throughput:.1f} MB/s")

# Clean up the test file
os.remove(file_path)
print(f"Cleaned up {file_path}")