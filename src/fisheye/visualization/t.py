# import zarr
# import numpy as np

# z = zarr.open('/nvme1/2026_01_13_22_41_02/Cam2010096.zarr/')

# # Get the latest crop run
# crop_run = z['crop_runs'].attrs.get('latest')
# crop_group = z[f'crop_runs/{crop_run}']

# # Check crop at index 352
# crop_img = crop_group['roi_images'][352]
# frame_idx = crop_group['frame_indices'][352]
# coords = crop_group['roi_coordinates_full'][352]

# print(f"Crop index 352:")
# print(f"  Frame index: {frame_idx}")
# print(f"  Coordinates (x1, y1): {coords}")
# print(f"  Crop stats: shape={crop_img.shape}, min={crop_img.min()}, max={crop_img.max()}, mean={crop_img.mean():.1f}")

# # Check if coords are reasonable for 4512x4512 image
# if coords[0] >= 0 and coords[0] < 4512 and coords[1] >= 0 and coords[1] < 4512:
#     print(f"  ✓ Coordinates are within bounds")
# else:
#     print(f"  ⚠️ Coordinates may be out of bounds for 4512x4512 image!")

# import zarr

# z = zarr.open('//nvme1/2026_01_13_22_41_02/Cam2010096.zarr')

# crop_run = z['crop_runs'].attrs.get('latest')
# crop_group = z[f'crop_runs/{crop_run}']

# # Check crop metadata
# print(f"Crop run: {crop_run}")
# print(f"Source type: {crop_group.attrs.get('detection_source_type')}")
# print(f"Source path: {crop_group.attrs.get('detection_source_path')}")
# print(f"Video source: {crop_group.attrs.get('video_source_type')}")
# print(f"Video path: {crop_group.attrs.get('video_source_path')}")
# print(f"Total crops: {crop_group.attrs.get('summary_statistics', {}).get('total_rois_cropped')}")
# print(f"Status: {crop_group.attrs.get('status')}")

# # Check if crop 352 was supposed to be written
# # Look at a few surrounding crops to see if ANY have data
# for idx in [350, 351, 352, 353, 354, 500, 1000]:
#     crop_img = crop_group['roi_images'][idx]
#     print(f"Crop {idx}: max={crop_img.max()}")

# import zarr
# import numpy as np

# z = zarr.open('/nvme1/2026_01_13_22_41_02/Cam2010096.zarr')

# # Get refined/interpolated detection data
# refined_group = z['refined_detect_runs/refined_detect_2026-01-14_10-44-45/interpolated']
# detection_source = refined_group['detection_source'][:]  # 0=original, 1=interpolated
# frame_indices = refined_group['frame_indices'][:]

# # Check which crops are interpolated
# for idx in [350, 351, 352, 353, 354, 355, 500, 1000]:
#     is_interp = detection_source[idx] == 1
#     frame = frame_indices[idx]
#     print(f"Crop {idx}: frame={frame}, interpolated={is_interp}")

# # Count interpolated detections
# total_interp = np.sum(detection_source == 1)
# print(f"\nTotal interpolated: {total_interp} / {len(detection_source)}")

# """Debug script to trace chunk boundaries and detection indexing for frame 352"""
# import numpy as np
# import zarr

# zarr_path = "/nvme1/2026_01_13_22_41_02/Cam2010096.zarr"
# print(f"Using: {zarr_path}")

# root = zarr.open(zarr_path, mode='r')

# if 'crop_runs' not in root:
#     print("No crop_runs found!")
#     exit(1)

# # Get crop run
# crop_runs = root['crop_runs']
# latest = crop_runs.attrs.get('latest', list(crop_runs.group_keys())[0])
# crop_group = crop_runs[latest]
# print(f"Crop run: {latest}")

# # Get detection source
# source_path = crop_group.attrs.get('detection_source_path', 'unknown')
# print(f"Detection source: {source_path}")

# # Load frame indices from crop
# frame_indices = crop_group['frame_indices'][:]
# print(f"\nTotal crops: {len(frame_indices)}")

# # Check if detections are sorted
# is_sorted = np.all(frame_indices[:-1] <= frame_indices[1:])
# print(f"Frame indices sorted: {is_sorted}")

# if not is_sorted:
#     # Find where sorting breaks
#     breaks = np.where(frame_indices[:-1] > frame_indices[1:])[0]
#     print(f"Sorting breaks at {len(breaks)} locations")
#     if len(breaks) > 0:
#         print(f"First 5 breaks: {breaks[:5].tolist()}")
#         for b in breaks[:3]:
#             print(f"  index {b}: frame {frame_indices[b]} -> index {b+1}: frame {frame_indices[b+1]}")

# # Find crop indices for frames 350-356
# print("\nCrop index to frame mapping for frames 350-356:")
# for target_frame in range(350, 357):
#     crop_indices = np.where(frame_indices == target_frame)[0]
#     if len(crop_indices) > 0:
#         print(f"  Frame {target_frame}: crop indices {crop_indices.tolist()}")
#     else:
#         print(f"  Frame {target_frame}: NO CROP FOUND")

# # Check the roi_images for those crops
# roi_images = crop_group['roi_images']
# print(f"\nROI images shape: {roi_images.shape}")

# # Check actual crop data for specific indices
# print("\nCrop data check (checking if crops have content):")
# for target_frame in range(350, 357):
#     crop_indices = np.where(frame_indices == target_frame)[0]
#     if len(crop_indices) > 0:
#         idx = crop_indices[0]
#         crop_data = roi_images[idx]
#         print(f"  Crop for frame {target_frame} (idx={idx}): min={crop_data.min()}, max={crop_data.max()}, mean={crop_data.mean():.1f}")

# # Chunk analysis
# chunk_size = 32
# print(f"\n=== Chunk Boundary Analysis (chunk_size={chunk_size}) ===")
# for frame in [350, 351, 352, 353, 354, 355]:
#     chunk_num = frame // chunk_size
#     chunk_start = chunk_num * chunk_size
#     chunk_end = (chunk_num + 1) * chunk_size
#     pos_in_chunk = frame - chunk_start
#     print(f"Frame {frame}: chunk {chunk_num} (frames {chunk_start}-{chunk_end-1}), position {pos_in_chunk}")

# # Check detection source for sorting
# if source_path != 'unknown':
#     try:
#         source_group = root[source_path]
#         src_frame_indices = source_group['frame_indices'][:]
#         src_is_sorted = np.all(src_frame_indices[:-1] <= src_frame_indices[1:])
#         print(f"\nSource frame_indices sorted: {src_is_sorted}")
        
#         if not src_is_sorted:
#             breaks = np.where(src_frame_indices[:-1] > src_frame_indices[1:])[0]
#             print(f"Source sorting breaks at {len(breaks)} locations")
#             if len(breaks) > 0:
#                 print(f"First 5 breaks: {breaks[:5].tolist()}")
        
#         # Check around frames 350-356
#         print("\nDetection indices for frames 350-356 in source:")
#         for target_frame in range(350, 357):
#             det_indices = np.where(src_frame_indices == target_frame)[0]
#             if len(det_indices) > 0:
#                 print(f"  Frame {target_frame}: detection indices {det_indices.tolist()}")
#     except Exception as e:
#         print(f"Could not load source: {e}")

# """Check which chunks have black crops"""
# import numpy as np
# import zarr

# zarr_path = "/nvme1/2026_01_13_22_41_02/Cam2010096.zarr"
# root = zarr.open(zarr_path, mode='r')
# crop_runs = root['crop_runs']
# latest = crop_runs.attrs.get('latest', list(crop_runs.group_keys())[0])
# crop_group = crop_runs[latest]

# roi_images = crop_group['roi_images']
# frame_indices = crop_group['frame_indices'][:]
# total_crops = len(frame_indices)

# print(f"Total crops: {total_crops}")
# print(f"Checking which chunks have black crops...\n")

# chunk_size = 32
# num_chunks = (total_crops + chunk_size - 1) // chunk_size

# black_chunks = []
# partial_black_chunks = []

# # Check each chunk
# for chunk_num in range(num_chunks):
#     start_idx = chunk_num * chunk_size
#     end_idx = min((chunk_num + 1) * chunk_size, total_crops)
    
#     # Sample a few crops from this chunk
#     sample_indices = [start_idx, (start_idx + end_idx) // 2, end_idx - 1]
#     sample_indices = [i for i in sample_indices if i < total_crops]
    
#     black_count = 0
#     for idx in sample_indices:
#         crop = roi_images[idx]
#         if crop.max() == 0:
#             black_count += 1
    
#     if black_count == len(sample_indices):
#         black_chunks.append(chunk_num)
#     elif black_count > 0:
#         partial_black_chunks.append((chunk_num, black_count, len(sample_indices)))

# print(f"Chunks with ALL black crops: {black_chunks}")
# print(f"Chunks with SOME black crops: {partial_black_chunks}")

# # More detailed look at black chunks
# print("\n=== Black Chunks Detail ===")
# for chunk_num in black_chunks[:5]:  # First 5
#     start_idx = chunk_num * chunk_size
#     end_idx = min((chunk_num + 1) * chunk_size, total_crops)
#     start_frame = frame_indices[start_idx]
#     end_frame = frame_indices[end_idx - 1]
#     print(f"Chunk {chunk_num}: crops {start_idx}-{end_idx-1}, frames {start_frame}-{end_frame}")

# # Check a few specific frames after the black region
# print("\n=== Checking frames 380-390 (end of chunk 11 / start of chunk 12) ===")
# for frame in range(380, 391):
#     crop_idxs = np.where(frame_indices == frame)[0]
#     if len(crop_idxs) > 0:
#         idx = crop_idxs[0]
#         crop = roi_images[idx]
#         chunk = idx // chunk_size
#         status = "BLACK" if crop.max() == 0 else f"max={crop.max()}"
#         print(f"Frame {frame} (idx={idx}, chunk {chunk}): {status}")

# # Check around chunk boundaries
# print("\n=== Checking around several chunk boundaries ===")
# for boundary_chunk in [10, 11, 12, 15, 20]:
#     boundary_frame = boundary_chunk * chunk_size
#     for offset in [-1, 0, 1]:
#         frame = boundary_frame + offset
#         if frame < 0 or frame >= total_crops:
#             continue
#         crop_idxs = np.where(frame_indices == frame)[0]
#         if len(crop_idxs) > 0:
#             idx = crop_idxs[0]
#             crop = roi_images[idx]
#             chunk = idx // chunk_size
#             status = "BLACK" if crop.max() == 0 else f"max={crop.max()}"
#             print(f"Chunk {boundary_chunk} boundary, frame {frame} (idx={idx}, chunk {chunk}): {status}")
#     print()

# """Check crop run metadata for scheduler info"""
# import zarr

# zarr_path = "/nvme1/2026_01_13_22_41_02/Cam2010096.zarr"
# root = zarr.open(zarr_path, mode='r')
# crop_runs = root['crop_runs']
# latest = crop_runs.attrs.get('latest', list(crop_runs.group_keys())[0])
# crop_group = crop_runs[latest]

# print("Crop run attributes:")
# for key, value in sorted(crop_group.attrs.items()):
#     if 'scheduler' in key.lower() or 'worker' in key.lower() or 'distributed' in key.lower():
#         print(f"  {key}: {value}")

# # Check zarr chunk size vs processing
# roi_images = crop_group['roi_images']
# print(f"\nroi_images shape: {roi_images.shape}")
# print(f"roi_images chunks: {roi_images.chunks}")

# print(f"\nScheduler: {crop_group.attrs.get('scheduler', 'unknown')}")
# print(f"Num workers: {crop_group.attrs.get('num_workers', 'unknown')}")
# print(f"Use distributed: {crop_group.attrs.get('use_distributed', 'unknown')}")

import zarr
root = zarr.open('/nvme1/2026_01_13_22_41_02/Cam2010096.zarr', mode='r')
crop_group = root['crop_runs'][root['crop_runs'].attrs['latest']]
roi_images = crop_group['roi_images']
print(f"roi_images chunks: {roi_images.chunks}")  # Should be (32, 256, 256) now
black_count = sum(1 for i in range(0, roi_images.shape[0], 100) if roi_images[i].max() == 0)
print(f"Sample black crops: {black_count}")
