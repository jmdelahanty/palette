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

# import zarr
# root = zarr.open('/nvme1/2026_01_13_22_41_02/Cam2010096.zarr', mode='r')
# crop_group = root['crop_runs'][root['crop_runs'].attrs['latest']]

# roi_images = crop_group['roi_images']
# print(f"roi_images chunks: {roi_images.chunks}")  # Should be (32, 256, 256) now
# black_count = sum(1 for i in range(0, roi_images.shape[0], 100) if roi_images[i].max() == 0)
# print(f"Sample black crops: {black_count}")

# import zarr, numpy as np
# root = zarr.open("/nvme1/2026_01_13_22_41_02/Cam2010096.zarr", mode="r")
# kp = root["keypoints_runs/keypoints_2026-01-16_16-39-13/keypoints_roi"][:]
# success = root["keypoints_runs/keypoints_2026-01-16_16-39-13/detection_success"][:].astype(bool)
# bad = np.any(~np.isfinite(kp[:, 1:]), axis=(1,2))
# print("bad keypoints:", bad.sum(), "bad but success=True:", np.count_nonzero(bad & success))

# ref = root["refined_eye_masks_runs/refined_eye_masks_2026-01-16_21-34-57"]
# print("ellipse_success pairs:", np.sum(np.all(ref["ellipse_success"][:], axis=1)))

# import zarr
# root = zarr.open('/nvme1/2026_01_13_22_41_02/Cam2010096.zarr/', mode='r')

# refined_kp = root['refined_keypoints_runs/refined_keypoints_2026-01-16_13-18-49']
# print('Refined keypoints datasets:', list(refined_kp.array_keys()))
# print('Refined attrs:', dict(refined_kp.attrs))

# source_name = refined_kp.attrs.get('source_keypoints_run')
# print(f'\nSource run: {source_name}')
# if source_name:
#     source_kp = root.get(f'keypoints_runs/{source_name}')
#     if source_kp:
#         print('Source keypoints datasets:', list(source_kp.array_keys()))

# import zarr

# path = "/nvme1/2026_01_13_22_41_02/Cam2010096.zarr/"  # root of your v3 store

# g = zarr.open_group(path, mode="r")  # or zarr.open(path, mode="r") if you want
# cm = getattr(g.metadata, "consolidated_metadata", None)

# if cm is None:
#     print("❌ No consolidated_metadata attribute found on group metadata (unexpected for recent zarr-python).")
# elif cm is None or getattr(cm, "metadata", None) in (None, {}):
#     print("❌ Consolidated metadata not present (or empty).")
# else:
#     # cm.metadata is a mapping of paths -> metadata objects
#     print("✅ Consolidated metadata present.")
#     print("Entries in consolidated metadata:", len(cm.metadata))
# import zarr
# root = zarr.open_group("/nvme1/2026_01_13_22_41_02/Cam2010096.zarr/", mode="r")
# arr = root["raw_video/images_full"]
# print("chunks:", arr.chunks)
# print("shards:", getattr(arr, "shards", None))
# print("chunk_grid:", getattr(arr.metadata, "chunk_grid", None))
# import zarr

# path = "/nvme1/2026_01_13_22_41_02/Cam2010096.zarr/"

# try:
#     g = zarr.open_consolidated(path, mode="r")
#     print("✅ Consolidated metadata is present (open_consolidated succeeded).")
# except Exception as e:
#     print("❌ Consolidated metadata not present / not readable:", type(e).__name__, e)
# import json
# from pathlib import Path

# root = Path("/nvme1/2026_01_13_22_41_02/Cam2010096.zarr/")
# meta = json.loads((root / "zarr.json").read_text())

# print("has consolidated_metadata key:", "consolidated_metadata" in meta)
# if "consolidated_metadata" in meta:
#     cm = meta["consolidated_metadata"]
#     # depending on representation, this may be a dict with entries
#     print("type:", type(cm))

# import zarr
# path = "/nvme1/2026_01_13_22_41_02/Cam2010096.zarr/"

# g = zarr.open_group(path, mode="r")
# print("g type:", type(g))
# print("metadata type:", type(g.metadata))
# print("has consolidated_metadata attr:", hasattr(g.metadata, "consolidated_metadata"))
# print("metadata dir contains 'consolid':", [x for x in dir(g.metadata) if "consolid" in x])
# import zarr
# path = "/nvme1/2026_01_13_22_41_02/Cam2010096.zarr/"
# zarr.consolidate_metadata(path)
# print("done")
# import zarr
# path = "/nvme1/2026_01_13_22_41_02/Cam2010096.zarr/"

# zarr.open_consolidated(path, mode="r")
# print("✅ consolidated metadata now present and readable")
# import json
# from pathlib import Path

# root = Path("/nvme1/2026_01_13_22_41_02/Cam2010096.zarr/")
# meta = json.loads((root / "zarr.json").read_text())
# print(type(meta.get("consolidated_metadata")), meta.get("consolidated_metadata") is None)
# import zarr
# root = zarr.open_group("/nvme1/2026_01_13_22_41_02/test.zarr", mode="r")
# print(root["raw_video"].attrs.get("chunk_size"))
# print(root["raw_video"].attrs.get("io_batch_size"))
# print(root["raw_video"].attrs.get("batch_size"))
# print(root["raw_video"].attrs.get("chunks_per_shard"))
# print(root["raw_video"].attrs.get("import_timestamp"))

# import zarr, numpy as np
# from pathlib import Path

# zarr_path = Path("/nvme1/recordings/2026-01-28T22-22-57Z_arena_1_Feeding/zarr/2026-01-28T22-22-57Z_arena_1_Feeding.zarr")
# root = zarr.open_group(str(zarr_path), mode="r")

# refined_parent = root.get("refined_detect_runs") or root.get("refined_runs")
# print("refined_parent:", bool(refined_parent))
# latest = refined_parent.attrs.get("latest") if refined_parent else None
# print("latest refined:", latest)
# ref = refined_parent[latest]

# print("operations:", ref.attrs.get("operations"))
# params = ref.attrs.get("parameters") or {}
# print("sampled_import:", params.get("sampled_import"))
# print("sampled_import_meta:", params.get("sampled_import_meta"))
# print("manual_review_latest:", ref.attrs.get("manual_review_latest"))
# print("coverage_frames_total:", ref.attrs.get("coverage_frames_total"))
# print("coverage_frame_source:", ref.attrs.get("coverage_frame_source"))
# print("coverage_frames_full:", ref.attrs.get("coverage_frames_full"))

# raw = root.get("raw_video")
# if raw is None:
#     print("raw_video: MISSING")
#     original_idx = None
# else:
#     original_idx = raw.get("original_frame_indices")
#     print("images_ds:", raw["images_ds"].shape if "images_ds" in raw else None)
#     print("images_full:", raw["images_full"].shape if "images_full" in raw else None)
#     print("original_frame_indices:", original_idx.shape if original_idx is not None else None)
#     print("raw import_mode:", raw.attrs.get("import_mode"))
#     print("raw import_purpose:", raw.attrs.get("import_purpose"))
#     print("raw frame_step:", raw.attrs.get("frame_step"))

# base = ref.get("interpolated") or ref.get("filtered")
# print("base group:", "interpolated" if ref.get("interpolated") is not None else "filtered" if ref.get("filtered") is not None else None)
# fc = base.get("frame_counts") or base.get("n_detections")
# print("frame_counts shape:", fc.shape, "chunks:", fc.chunks)

# n = fc.shape[0]
# chunk = fc.chunks[0] if fc.chunks else 10000
# nz = 0
# for start in range(0, n, chunk):
#     arr = fc[start:start+chunk]
#     nz += int(np.sum(arr > 0))
# print("coverage over frame_counts array:", nz, "/", n, "=", (nz / n * 100.0 if n else None))

# if original_idx is not None:
#     idx = original_idx[:].astype(np.int64, copy=False)
#     idx = idx[(idx >= 0) & (idx < n)]
#     if idx.size:
#         idx = np.unique(idx)
#         sampled_nz = int(np.sum(fc[idx] > 0))
#         print("coverage over original_frame_indices:", sampled_nz, "/", idx.size, "=", sampled_nz / idx.size * 100.0)
# import zarr, numpy as np
# from pathlib import Path

# zarr_path = Path("/nvme1/recordings/2026-01-28T22-22-57Z_arena_1_Feeding/zarr/2026-01-28T22-22-57Z_arena_1_Feeding.zarr")
# root = zarr.open_group(str(zarr_path), mode="r")
# refined_parent = root.get("refined_detect_runs") or root.get("refined_runs")
# run = refined_parent.attrs.get("latest")
# ref = refined_parent[run]

# print("run:", run)
# print("group keys:", list(ref.group_keys()) if hasattr(ref, "group_keys") else list(ref.keys()))
# print("manual_review_latest:", ref.attrs.get("manual_review_latest"))

# def cov(name):
#     grp = ref.get(name)
#     if grp is None:
#         print(f"{name}: MISSING")
#         return
#     fc = grp.get("frame_counts") or grp.get("n_detections")
#     if fc is None:
#         print(f"{name}: no frame_counts")
#         return
#     counts = fc[:]
#     print(f"{name}: {(counts>0).sum()}/{len(counts)} = {(counts>0).sum()/len(counts)*100:.1f}%")

# cov("interpolated")
# cov("manual")

# import zarr
# from pathlib import Path

# zarr_path = Path("/nvme1/recordings/2026-01-28T22-22-57Z_arena_1_Feeding/zarr/2026-01-28T22-22-57Z_arena_1_Feeding.zarr")
# root = zarr.open_group(str(zarr_path), mode="a")
# refined_parent = root.get("refined_detect_runs") or root.get("refined_runs")
# run = refined_parent.attrs.get("latest")
# ref = refined_parent[run]

# if "manual" in ref:
#     ref.attrs["manual_review_latest"] = "manual"
#     print("set manual_review_latest=manual")
# else:
#     print("manual group missing")

# import zarr, numpy as np
# path="/nvme1/recordings/2026-01-28T19-36-18Z_arena_2_Feeding/zarr/2026-01-28T19-36-18Z_arena_2_Feeding.zarr"
# root=zarr.open(path, mode="r")
# run=root["refined_detect_runs"].attrs["latest"]
# manual=root[f"refined_detect_runs/{run}/manual"]
# frame=40
# mf=(manual["frame_indices"][:]==frame).sum()
# print("manual detections @ frame 40:", mf)
# det_run=root["detect_runs"].attrs["latest"]
# det=root[f"detect_runs/{det_run}"]
# df=(det["frame_indices"][:]==frame).sum()
# print("original detections @ frame 40:", df)
# import zarr, numpy as np

# z = zarr.open("/nvme1/recordings/2026-01-28T22-15-03Z_arena_1_DefaultScreen/zarr/2026-01-28T22-15-03Z_arena_1_DefaultScreen.zarr", mode="r")
# crop_run = z["crop_runs"].attrs["latest"]
# cg = z[f"crop_runs/{crop_run}"]

# idx = 93
# roi = cg["roi_images"][idx]
# frame_idx = int(cg["frame_indices"][idx])
# x0, y0 = map(int, cg["roi_coordinates_full"][idx])
# h, w = roi.shape[:2]

# frame = z["raw_video/images_full"][frame_idx]

# # Handle edge padding (crop code pads with zeros if ROI goes out of bounds)
# frame_crop = np.zeros_like(roi)
# y1, y2 = max(0, y0), min(frame.shape[0], y0 + h)
# x1, x2 = max(0, x0), min(frame.shape[1], x0 + w)
# py1, py2 = y1 - y0, y2 - y0
# px1, px2 = x1 - x0, x2 - x0
# frame_crop[py1:py2, px1:px2] = frame[y1:y2, x1:x2]

# diff = np.abs(frame_crop.astype("i2") - roi.astype("i2"))
# print("roi dtype:", roi.dtype, "frame dtype:", frame.dtype)
# print("max abs diff:", diff.max())
# print("mean abs diff:", diff.mean())

# import zarr, numpy as np

# z = zarr.open("/nvme1/recordings/2026-01-28T19-22-28Z_arena_1_DefaultScreen/zarr/2026-01-28T19-22-28Z_arena_1_DefaultScreen.zarr", mode="r")
# crop_run = z["crop_runs"].attrs["latest"]
# cg = z[f"crop_runs/{crop_run}"]

# idx = 170
# x1, y1 = map(int, cg["roi_coordinates_full"][idx])
# h, w = cg["roi_images"].shape[1:]
# H, W = z["raw_video/images_full"].shape[1:]

# print("ROI top-left:", (x1, y1), "size:", (w, h))
# print("Frame size:", (W, H))
# print("Out of bounds?",
#     x1 < 0 or y1 < 0 or (x1 + w) > W or (y1 + h) > H)

# bg_runs = z.get("background_runs")
# if bg_runs is not None:
#     latest = bg_runs.attrs.get("latest")
#     if latest and latest in bg_runs:
#         bg_group = bg_runs[latest]
#         print("background_full present:", "background_full" in bg_group)

# import zarr, numpy as np

# z = zarr.open("/nvme1/recordings/2026-01-28T19-22-28Z_arena_1_DefaultScreen/zarr/2026-01-28T19-22-28Z_arena_1_DefaultScreen.zarr", mode="r")
# crop_run = z["crop_runs"].attrs["latest"]
# cg = z[f"crop_runs/{crop_run}"]

# frame = 170
# det = 0
# frame_counts = cg["frame_counts"][:]
# roi_idx = int(frame_counts[:frame].sum() + det)

# roi = cg["roi_images"][roi_idx]
# x1, y1 = map(int, cg["roi_coordinates_full"][roi_idx])
# h, w = roi.shape
# H, W = z["raw_video/images_full"].shape[1:]

# print("ROI idx:", roi_idx)
# print("ROI top-left:", (x1, y1), "size:", (w, h))
# print("Frame size:", (W, H))
# print("Out of bounds?", x1 < 0 or y1 < 0 or (x1+w) > W or (y1+h) > H)

# bg = z["background_runs"][z["background_runs"].attrs["latest"]]["background_full"][:]
# bg_roi = bg[y1:y1+h, x1:x1+w]
# print("background_full shape:", bg.shape)
# print("background_roi shape:", bg_roi.shape)
# print("roi shape:", roi.shape)

# import json, pathlib
# src = pathlib.Path("keypoint_frame_flags.json")
# data = json.loads(src.read_text())
# out = pathlib.Path("keypoint_manual_list.txt")
# out.write_text("\n".join(sorted(data.keys())) + "\n")
# print(f"Wrote {out}")

# import zarr, numpy as np, collections

# path = "/nvme1/recordings/2026-01-28T19-22-28Z_arena_1_DefaultScreen/zarr/2026-01-28T19-22-28Z_arena_1_DefaultScreen.zarr"
# root = zarr.open_group(path, mode="r")
# parent = root["refined_keypoints_runs"]
# run = parent[parent.attrs["latest"]]

# usable = np.asarray(run["usable_keypoints"][:], dtype=bool)
# print("usable false:", np.sum(~usable))

# for name in ["refined_success", "confidence_valid", "geometry_valid", "heading_valid"]:
#     arr = run.get(name)
#     if arr is not None:
#         vals = np.asarray(arr[:], dtype=bool)
#         print(f"{name} false:", np.sum(~vals))

# reason = run.get("reason")
# if reason is not None:
#     raw = np.asarray(reason[:], dtype=object)
#     counts = collections.Counter()
#     for r in raw[~usable]:
#         if not r:
#             continue
#         for tag in str(r).split("|"):
#             tag = tag.strip()
#             if tag:
#                 counts[tag] += 1
#     print("reason tags (non-usable):", dict(counts))
# import json
# from pathlib import Path
# import zarr

# zarr_path = Path("/nvme1/recordings/2026-01-28T19-22-28Z_arena_1_DefaultScreen/zarr/2026-01-28T19-22-28Z_arena_1_DefaultScreen.zarr")
# root = zarr.open(str(zarr_path), mode="r")

# print("Zarr:", zarr_path)

# # Camera metadata (exposure, framerate, etc.)
# analysis = root.get("analysis_metadata")
# if analysis is None:
#     print("analysis_metadata: missing")
# else:
#     raw = analysis.attrs.get("camera_metadata")
#     if raw is None:
#         print("camera_metadata: missing")
#         cam = None
#     else:
#         if isinstance(raw, (bytes, bytearray)):
#             raw = raw.decode("utf-8", "ignore")
#         if isinstance(raw, str):
#             try:
#                 cam = json.loads(raw)
#             except Exception as e:
#                 print("camera_metadata json load error:", e)
#                 cam = None
#         elif isinstance(raw, dict):
#             cam = raw
#         else:
#             cam = None

#     if isinstance(cam, dict):
#         keys = sorted(cam.keys())
#         print("camera_metadata keys:", keys)
#         for key in keys:
#             k = key.lower()
#             if "exposure" in k or "frame" in k or k in {
#                 "fps","framerate","frame_rate","frame_rate_hz","frame_rate_fps",
#                 "acquisition_fps","acquisition_rate","gain","binning","bit_depth"
#             }:
#                 print(f"  {key}: {cam.get(key)}")
#     else:
#         print("camera_metadata: not dict")

# # Raw video attrs
# raw_video = root.get("raw_video")
# if raw_video is None:
#     print("raw_video: missing")
# else:
#     print("raw_video attrs keys:", sorted(raw_video.attrs.keys()))
#     for key in ("fps","codec","pix_fmt","source_video","compressor","downsampled_resolution","original_resolution"):
#         if key in raw_video.attrs:
#             print(f"  raw_video.{key}:", raw_video.attrs.get(key))

# # Root source video metadata
# source_meta = root.attrs.get("source_video_metadata") or {}
# print("source_video_metadata keys:", sorted(source_meta.keys()))
# for key in ("fps","width","height","total_frames","codec","pix_fmt","source_path","source_video_path"):
#     if key in source_meta:
#         print(f"  source_video_metadata.{key}:", source_meta.get(key))

# # Dish design (arena config or root attrs)
# arena_config = None
# stim_parent = root.get("analysis")
# if stim_parent is not None and "stimulus_runs" in stim_parent:
#     stim_runs = stim_parent["stimulus_runs"]
#     latest = stim_runs.attrs.get("latest")
#     if latest and latest in stim_runs:
#         stim = stim_runs[latest]
#         raw = stim.attrs.get("arena_config_json")
#         if isinstance(raw, (bytes, bytearray)):
#             raw = raw.decode("utf-8", "ignore")
#         if isinstance(raw, str):
#             try:
#                 arena_config = json.loads(raw)
#             except Exception:
#                 arena_config = None
#         elif isinstance(raw, dict):
#             arena_config = raw

# if isinstance(arena_config, dict):
#     print("arena_config keys (sample):", sorted(arena_config.keys())[:15])
#     for key in ("dish_design","arena_design","arena_config_name","experimental_area_shape"):
#         if key in arena_config:
#             print(f"  arena_config.{key}:", arena_config.get(key))
# else:
#     print("arena_config: missing or invalid")

# print("root attrs dish_design:", root.attrs.get("dish_design"))
# from pathlib import Path
# import tempfile, shutil
# import numpy as np, zarr

# root_dir = Path(tempfile.mkdtemp(prefix='zarr_smoke_only_'))
# print("tmp:", root_dir)
# try:
#     zarr_path = root_dir / 'sample.zarr'
#     g = zarr.open_group(str(zarr_path), mode='w')
#     raw = g.create_group('raw_video')
#     raw.create_array('images_ds', data=np.zeros((2,8,8), dtype=np.uint8), chunks=(1,8,8))
#     raw.create_array('images_ds_rgb', data=np.zeros((2,8,8,3), dtype=np.uint8), chunks=(1,8,8,3))
#     raw.attrs['downsample_formats'] = ['gray', 'rgb']
#     print("wrote zarr")
#     r = zarr.open_group(str(zarr_path), mode='r')
#     print("keys:", list(r['raw_video'].array_keys()))
# finally:
#     shutil.rmtree(root_dir, ignore_errors=True)
# from pathlib import Path
# import tempfile, shutil
# import numpy as np, zarr

# root_dir = Path(tempfile.mkdtemp(prefix='zarr_smoke_only_'))
# print("tmp:", root_dir)
# try:
#     zarr_path = root_dir / 'sample.zarr'
#     g = zarr.open_group(str(zarr_path), mode='w')
#     raw = g.create_group('raw_video')
#     raw.create_array('images_ds', data=np.zeros((2,8,8), dtype=np.uint8), chunks=(1,8,8))
#     raw.create_array('images_ds_rgb', data=np.zeros((2,8,8,3), dtype=np.uint8), chunks=(1,8,8,3))
#     raw.attrs['downsample_formats'] = ['gray', 'rgb']
#     print("wrote zarr")
#     r = zarr.open_group(str(zarr_path), mode='r')
#     print("keys:", list(r['raw_video'].array_keys()))
# finally:
#     shutil.rmtree(root_dir, ignore_errors=True)
# from pathlib import Path
# import tempfile, shutil, sqlite3
# import numpy as np, zarr
# from fisheye.registry.db import Registry

# root_dir = Path(tempfile.mkdtemp(prefix='registry_scan_smoke_'))
# print("tmp:", root_dir)
# try:
#     zarr_path = root_dir / 'sample.zarr'
#     g = zarr.open_group(str(zarr_path), mode='w')
#     g.attrs['session_uuid'] = 'session_rgb_smoke'
#     raw = g.create_group('raw_video')
#     raw.create_array('images_ds', data=np.zeros((2,8,8), dtype=np.uint8), chunks=(1,8,8))
#     raw.create_array('images_ds_rgb', data=np.zeros((2,8,8,3), dtype=np.uint8), chunks=(1,8,8,3))
#     raw.attrs['downsample_formats'] = ['gray', 'rgb']

#     db_path = root_dir / 'registry.sqlite'
#     reg = Registry(db_path)
#     print("registry ready")
#     reg.scan_zarr(zarr_path)
#     reg.close()
#     print("scan done")

#     con = sqlite3.connect(db_path)
#     row = con.execute(
#         "SELECT has_images_ds, has_images_ds_rgb, downsample_formats_json FROM provenance WHERE dataset_id=?",
#         ('session_rgb_smoke',),
#     ).fetchone()
#     print("provenance row:", row)
#     con.close()
# finally:
#     shutil.rmtree(root_dir, ignore_errors=True)
# import sqlite3
# con = sqlite3.connect("/nvme1/palette_registry.sqlite")
# print(con.execute("""
# SELECT COUNT(*),
#         SUM(CASE WHEN has_images_ds = 1 THEN 1 ELSE 0 END),
#         SUM(CASE WHEN has_images_ds_rgb = 1 THEN 1 ELSE 0 END)
# FROM provenance
# """).fetchone())
# con.close()
# import sqlite3

# db = "/nvme1/palette_registry.sqlite"
# run_id = "multi_zarr_train_20260206-124840_3037026"
# set_id = "detect_detect_cedar_v001"

# conn = sqlite3.connect(db)
# with conn:
#     cur = conn.execute(
#         "UPDATE training_runs SET set_id = ? WHERE run_id = ?;",
#         (set_id, run_id),
#     )
# print(f"rows_updated={cur.rowcount}")
import json, pathlib
run = pathlib.Path("/nvme1/models/detect/detect_cedar_shadow_v007/omnifin0_cedar_shadow_v007_detect_20260206-235656_25f3fbcb")
onnx_manifest = run / "exports/onnx" / f"{run.name}.onnx.manifest.json"
trt_manifest = run / "exports/tensorrt" / f"{run.name}_fp16.tensorrt.manifest.json"
for p in [onnx_manifest, trt_manifest]:
    print("\n", p)
    d = json.loads(p.read_text())
    print(d.get("onnx", {}).get("outputs"))
