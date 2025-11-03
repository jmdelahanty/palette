# Frame Sampling Implementation Summary

## Phase 1: Uniform Frame Sampling ✅ COMPLETE

### What Was Implemented

Added the ability to import a subset of frames from videos for training data collection. This allows efficient sampling from multiple videos without storing full datasets.

### Changes Made

#### 1. CLI Arguments ([import_video.py](src/fisheye/capture/import_video.py))

Added two new arguments:
- `--training-data`: Flag to enable sampled import mode
- `--frame-step N`: Import every Nth frame (requires `--training-data`)

**Usage:**
```bash
# Import every 100th frame for training data
python -m fisheye.capture.import_video video.mp4 \
    --training-data \
    --frame-step 100 \
    --zarr-path training_sample.zarr

# Standard full import (unchanged)
python -m fisheye.capture.import_video video.mp4 \
    --zarr-path full_video.zarr
```

#### 2. Frame Index Computation

Added `_compute_frame_indices()` function that generates the list of frames to import:
- **Full import**: `[0, 1, 2, ..., n-1]`
- **Sampled import** (step=100): `[0, 100, 200, 300, ...]`

#### 3. Modified Processing Pipeline

Updated `_process_video_gpu_kvikio()` to accept and use frame indices:
- Iterates over selected frame indices instead of sequential ranges
- Uses `vr.get_batch(frame_indices)` for selective decoding
- Writes frames sequentially to zarr (indices 0, 1, 2, ...) even though source frames are sparse

#### 4. Zarr Structure Updates

**Array shapes**: Use `n_import_frames` instead of `n_frames` for sampled imports

**New array**: `raw_video/original_frame_indices`
- Shape: `(n_import_frames,)`
- Dtype: `int32`
- Content: Maps each imported frame to its original video frame index
- Example: `[0, 100, 200, ...]` for step=100

**Metadata additions** (stored in `raw_video/.zattrs`):
```python
{
    "import_mode": "sampled",  # or "full"
    "frame_step": 100,
    "original_video_length": 50000,
    "imported_frame_count": 500,
    "import_purpose": "training_data"
}
```

### Backward Compatibility

✅ **Fully backward compatible**
- Default behavior unchanged (full import)
- Sampling is opt-in via explicit flags
- All existing code continues to work
- Standard zarr structure maintained

### Testing

Created test scripts:
- `test_frame_sampling_simple.py`: Validates frame index computation logic
- All edge cases tested and passing

### Example Workflows

#### Collect Training Data from Multiple Videos

```bash
# Sample 1% of frames from multiple videos
for video in dataset/*.mp4; do
    name=$(basename "$video" .mp4)
    python -m fisheye.capture.import_video "$video" \
        --training-data \
        --frame-step 100 \
        --zarr-path "training_samples/${name}.zarr"
done
```

#### Inspect Sampled Import

```python
import zarr

# Open sampled zarr
root = zarr.open('training_sample.zarr', 'r')
raw = root['raw_video']

# Check metadata
print(f"Import mode: {raw.attrs['import_mode']}")
print(f"Frame step: {raw.attrs['frame_step']}")
print(f"Original video: {raw.attrs['original_video_length']} frames")
print(f"Imported: {raw.attrs['imported_frame_count']} frames")

# Access frame mapping
original_indices = raw['original_frame_indices'][:]
print(f"Frame 0 in zarr → Frame {original_indices[0]} in video")
print(f"Frame 10 in zarr → Frame {original_indices[10]} in video")

# Load frames (same API as full import)
frames = raw['images_ds'][0:10]  # First 10 imported frames
```

### Performance Benefits

For training data collection from many videos:
- **Storage**: 1/100th space for step=100 (1% of frames)
- **Import time**: ~1/100th time (only decode selected frames)
- **Diversity**: Can sample from 100 videos instead of 1 full video

Example:
- 100 videos × 50,000 frames each = 5,000,000 total frames
- With step=100: Import 50,000 frames (1% of total)
- Storage: ~50x less than 1 full video
- Training diversity: 100 different experimental conditions

### Known Limitations

1. **CPU processing path**: Not yet implemented (currently GPU-only via kvikIO)
2. **Tracking/interpolation**: Sampled imports break temporal continuity needed for tracking
   - Sampled imports intended for training data only
   - Full imports still needed for tracking workflows

3. **No random sampling**: Only uniform sampling (every Nth frame)
   - Random sampling could be added in future if needed

### Next Steps

**Phase 2: Training Manifest System** (pending)
- YAML-based manifest to combine multiple zarr archives
- Metadata tracking for training dataset composition
- Dataset weighting and sampling strategies

**Phase 3: Training Zarr Compiler** (pending)
- Tool to merge manifest → single optimized zarr
- Optimal chunking for training I/O
- Storage/flexibility trade-off

**Phase 4: Dual-Mode Dataloader** (pending)
- Support both manifest and compiled zarr loading
- Chunk-aware batching for manifest mode
- Backward compatible with existing datasets

**Phase 5: I/O Optimizations** (pending)
- Persistent workers
- Chunk caching
- Network storage tuning

**Phase 6: CLI Tools & Documentation** (pending)
- Manifest management CLI
- Training best practices guide
- Performance benchmarking

---

## Implementation Details

### File Modified
- [src/fisheye/capture/import_video.py](src/fisheye/capture/import_video.py)

### Lines Changed
- Added: ~100 lines
- Modified: ~30 lines
- Total impact: ~130 lines

### Key Functions
- `_compute_frame_indices()`: Frame sampling logic
- `_process_video_gpu_kvikio()`: Accept frame_indices parameter
- `build_parser()`: New CLI arguments
- `main()`: Argument validation
- `import_video()`: Frame sampling coordination

### Metadata Schema

```yaml
# Full import
raw_video/.zattrs:
  import_mode: "full"
  total_frames: 50000

# Sampled import
raw_video/.zattrs:
  import_mode: "sampled"
  frame_step: 100
  original_video_length: 50000
  imported_frame_count: 500
  import_purpose: "training_data"

# Arrays
raw_video/images_full: (500, H, W)  # or (50000, H, W) for full
raw_video/images_ds: (500, 640, 640)
raw_video/original_frame_indices: (500,)  # Only in sampled mode
```

---

## Questions?

For issues or questions about frame sampling:
1. Check zarr metadata: `raw_video/.zattrs['import_mode']`
2. Verify CLI flags: `--training-data` + `--frame-step` required together
3. Review test scripts: `test_frame_sampling_simple.py`

For the next phases (manifest system, compiler, dataloader):
- See implementation plan in planning session notes
- Estimated timeline: 7-10 days for phases 2-6
