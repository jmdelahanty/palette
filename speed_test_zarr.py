# import time
# import zarr
# import numpy as np

# print(f"Zarr version: {zarr.__version__}")

# # Create test data matching your video
# test_data = np.random.randint(0, 255, (128, 4512, 4512), dtype=np.uint8)
# mb = test_data.nbytes / (1024**2)
# print(f"Test data size: {mb:.1f} MB")

# # Test 1: Single large chunk (best case)
# z1 = zarr.create(
#     shape=(128, 4512, 4512),
#     chunks=(128, 4512, 4512),  # Single chunk
#     dtype='uint8',
#     store='/home/delahantyj@hhmi.org/Desktop/newone/test1.zarr',
#     overwrite=True
# )

# t0 = time.time()
# z1[:] = test_data
# dt = time.time() - t0
# print(f"\nSingle chunk write: {mb/dt:.1f} MB/s")

# # Test 2: Your current config (4 frames per chunk)
# z2 = zarr.create(
#     shape=(128, 4512, 4512),
#     chunks=(4, 4512, 4512),  # 32 chunks
#     dtype='uint8',
#     store='/home/delahantyj@hhmi.org/Desktop/newone/test2.zarr',
#     overwrite=True
# )

# t0 = time.time()
# z2[:] = test_data
# dt = time.time() - t0
# print(f"Small chunks (4 frames): {mb/dt:.1f} MB/s")

# # Test 3: Medium chunks
# z3 = zarr.create(
#     shape=(128, 4512, 4512),
#     chunks=(64, 4512, 4512),  # 2 chunks
#     dtype='uint8',
#     store='/home/delahantyj@hhmi.org/Desktop/newone/test3.zarr',
#     overwrite=True
# )

# t0 = time.time()
# z3[:] = test_data
# dt = time.time() - t0
# print(f"Medium chunks (64 frames): {mb/dt:.1f} MB/s")

# # Test 4: Raw numpy write for comparison
# import os
# raw_path = '/home/delahantyj@hhmi.org/Desktop/newone/test.raw'
# t0 = time.time()
# with open(raw_path, 'wb') as f:
#     f.write(test_data.tobytes())
# dt = time.time() - t0
# print(f"\nRaw binary write: {mb/dt:.1f} MB/s")
# os.remove(raw_path)

# # Clean up
# import shutil
# for path in ['test1.zarr', 'test2.zarr', 'test3.zarr']:
#     full_path = f'/home/delahantyj@hhmi.org/Desktop/newone/{path}'
#     if os.path.exists(full_path):
#         shutil.rmtree(full_path)


import zarr

# Match your successful test
z = zarr.create(
    shape=(45627, 4512, 4512),
    chunks=(4, 4512, 4512),  # Your fastest config
    dtype='uint8',
    store='/home/delahantyj@hhmi.org/Desktop/newone/test_real.zarr',
    overwrite=True
)

# Test write speed
import numpy as np
import time

test_shard = np.random.randint(0, 255, (128, 4512, 4512), dtype=np.uint8)
t0 = time.time()
z[0:128] = test_shard
dt = time.time() - t0
mb = test_shard.nbytes / (1024**2)
print(f"Real config write: {mb/dt:.1f} MB/s")