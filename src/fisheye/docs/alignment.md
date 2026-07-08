# Stimulus Alignment & Metadata

This document describes the current state of stimulus data import, interpolation, and alignment handling in the Palette Zarr workflow.

## Data Architecture

### H5 = Immutable Source of Truth
The source H5 file contains raw experimental data that is **never modified**:
- Raw stimulus frame metadata (may contain gaps or duplicate frames)
- Raw chaser tracking states (may have missing stimulus frames)
- Events, protocol snapshots, calibration data

### Zarr = Clean Working Copy
Each import creates a new timestamped run under `analysis/stimulus_runs/`:

```
analysis/stimulus_runs/
    └── stimulus_20251107_160756/
        ├── video_metadata/
        │   ├── frame_metadata/              # Interpolated, contiguous
        │   └── interpolation_mask           # Boolean: True=original, False=interpolated
        ├── tracking_data/
        │   ├── chaser_states/               # Interpolated, contiguous
        │   ├── chaser_interpolation_mask/   # Boolean: True=original, False=interpolated
        │   └── bounding_boxes/
        ├── frame_alignment/
        │   ├── camera_to_metadata_index     # Maps camera frames to metadata rows
        │   └── camera_interpolation_mask    # Tracks which camera frames use interpolated metadata
        ├── events/
        ├── calibration/
        └── protocol_snapshots/
```

## Interpolation Strategy

### What Gets Interpolated
During import (via `import_stimulus_to_zarr.py`), gaps are automatically detected and filled:

1. **Stimulus Metadata** (`video_metadata/frame_metadata/`)
   - Missing stimulus frames are synthesized via linear/nearest-neighbor interpolation
   - Creates a contiguous sequence of metadata records

2. **Chaser States** (`tracking_data/chaser_states/`)
   - Missing chaser positions are interpolated based on surrounding frames
   - Each chaser index is processed independently
   - Positions, velocities, timestamps are all interpolated to align with metadata

### What Gets Preserved
- **Boolean masks** identify which rows are original vs synthetic:
  - `interpolation_mask` (metadata level)
  - `chaser_interpolation_mask` (chaser_states level)
  - `camera_interpolation_mask` (camera frame level)

- **Attributes** record the interpolation:
  - `interpolated` (bool) - Whether interpolation occurred
  - `original_records` (int) - Count before interpolation
  - `total_records` (int) - Count after interpolation

### Filtering Back to Original Data
Downstream tools can filter to ground-truth measurements:

```python
chaser_states = run["tracking_data"]["chaser_states"]
mask = run["tracking_data"]["chaser_interpolation_mask"][:]
original_only = chaser_states[mask]  # True = original H5 data
```

## Diagnostic Scripts

### `check_stimulus_alignment.py`
**Location**: `src/fisheye/diagnostics/check_stimulus_alignment.py`

**Purpose**: Analyzes camera→stimulus frame alignment and reports:
- Camera to stimulus frame ratios (typically ~2:1 for 120fps stimulus, 60fps camera)
- Preview of frame mappings
- Gap statistics

**Current Status**: ✅ Fully functional

---

### `check_chaser_alignment.py`
**Location**: `src/fisheye/diagnostics/check_chaser_alignment.py`

**Purpose**: Compares chaser coverage across different timeline representations:
- Raw vs interpolated chaser states
- Time statistics (min/max timestamps, coverage)
- Drift analysis using anchor points

**Current Status**: ✅ Fully functional
- Uses interpolation masks to filter when available

---

### `check_chaser_periodicity.py`
**Location**: `src/fisheye/diagnostics/check_chaser_periodicity.py`

**Purpose**: Analyzes temporal patterns in chaser data gaps:
- Identifies dominant gap periods (e.g., ~60-frame cadence)
- Uses FFT-based periodicity detection
- Reports top gap frequencies

**Current Status**: ✅ Fully functional

---

### `plot_chaser_alignment.py`
**Location**: `src/fisheye/analysis/plot_chaser_alignment.py`

**Purpose**: Visual diagnostic showing:
1. Raw stimulus timeline
2. Camera frame alignment (red dots = missing samples)
3. Interpolated coverage (green dots = data from interpolated dataset)

**Current Status**: ✅ Fully functional
- Shows gap statistics for camera frame coverage
- Visualizes data before and after interpolation

---

### `inspect_stimulus_mapping.py`
**Location**: `src/fisheye/diagnostics/inspect_stimulus_mapping.py` (untracked)

**Purpose**: Detailed inspection of stimulus frame mappings
- Reports drift statistics
- Analyzes camera to stimulus ratios

**Current Status**: ⚠️ May need updates for current alignment format

## The "Happy Path" for Downstream Tools

Most analysis tools should:
1. Use `chaser_states` directly (interpolated, contiguous data)
2. Iterate frame-by-frame without gap handling
3. Trust that camera frames align 1:1 with metadata indices

**Filtering to original-only data is the exception, not the rule.** The masks exist for:
- Debugging alignment issues
- Quality assessment (how much data was interpolated?)
- Specialized analyses requiring ground-truth measurements only

## Re-import Safety

Each `import_stimulus_to_zarr` run creates an **independent, timestamped group**:
- No risk of re-interpolating already-interpolated data
- Each run starts fresh from the immutable H5 source
- Previous runs remain untouched
- The `latest` attribute points to the most recent import

## Typical Workflow

1. **Create Zarr** with detections/tracking:
   ```bash
   palette detect ...
   palette keypoints ...
   ```

2. **Import stimulus data**:
   ```bash
   scripts/py -m fisheye.analysis.import_stimulus_to_zarr /path/to/archive.zarr
   ```
   - Auto-detects the matching `.h5` file
   - Creates timestamped run under `analysis/stimulus_runs/`
   - Automatically interpolates if gaps are detected (default: `repair_chaser_gaps=True`)
   - Use `--skip-chaser-repair` to disable interpolation

3. **Analysis tools** read from:
   ```python
   run = zarr.open("archive.zarr/analysis/stimulus_runs/latest")
   chaser_states = run["tracking_data"]["chaser_states"]
   metadata = run["video_metadata"]["frame_metadata"]
   ```

## Future Enhancements

### Potential Improvements
- Generate actual "corrected" alignment arrays if sequential frame numbers are needed
- Add visualizer support for toggling between raw/interpolated overlays
- Emit timestamp ranges for gap analysis (not just frequencies)
- Build CLI to quantify interpolation error magnitude

### Upstream Fix
Long-term fix should happen in the **stimulus logging pipeline** so that `stimulus_frame_num` comes out sequential already. The current interpolation serves as a workaround, but fixing the source eliminates the need for post-processing.

### Performance Considerations
- Interpolation runs automatically during import (typically fast)
- For very large archives, consider profiling if import becomes slow
- The `--skip-chaser-repair` flag can bypass interpolation if needed

## Fixing Frame Numbering Issues

### Problem: Double-Incrementing Frame Counter

**Symptom**: The `stimulus_frame_num` field increments by 2 instead of 1, resulting in frame numbers like `[1, 3, 5, 7...]` instead of `[0, 1, 2, 3...]`.

**Root Cause**: A double-incrementing frame counter in the stimulus logging code caused frame numbers to skip every other value, even though all frames were being processed.

**Impact**:
- 50% of camera frames fail to map to chaser positions
- Visualization shows extensive "missing" data that isn't actually missing
- Interpolation cannot fill gaps caused by frame numbering mismatch

### Diagnostic Tools

Use these scripts to identify if your dataset has the issue:

1. **`count_h5_chaser_states.py`** - Count and validate chaser states
   ```bash
   scripts/py -m fisheye.diagnostics.count_h5_chaser_states /path/to/stimulus.h5
   ```
   Look for: "Regular 2:1 spacing detected" or gaps in frame sequence

2. **`analyze_h5_frame_spacing.py`** - Analyze frame spacing patterns
   ```bash
   scripts/py -m fisheye.diagnostics.analyze_h5_frame_spacing /path/to/stimulus.h5
   ```
   Look for: Dominant spacing of 2 instead of 1

3. **`diagnose_camera_chaser_mapping.py`** - Diagnose mapping failures
   ```bash
   scripts/py -m fisheye.analysis.diagnose_camera_chaser_mapping /path/to/archive.zarr
   ```
   Look for: ~50% failed mappings with reason "stimulus frame not in chaser_states"

### Fix Procedure

**⚠️ Important**: This modifies the H5 file. Always create a backup first!

#### Step 1: Create Backup
```bash
cp /path/to/stimulus.h5 /path/to/stimulus.h5.original
```

#### Step 2: Fix Frame Numbering

**Dry run** (preview changes without modifying):
```bash
scripts/py -m fisheye.diagnostics.fix_h5_chaser_frame_numbers /path/to/stimulus.h5 --dry-run
```

**Apply fix** (modifies both chaser_states and frame_metadata):
```bash
scripts/py -m fisheye.diagnostics.fix_h5_chaser_frame_numbers /path/to/stimulus.h5 --fix-all
```

The script will:
- Automatically create a backup at `/path/to/stimulus.h5.bak`
- Renumber frames sequentially: `0, 1, 2, 3...`
- Fix both `/tracking_data/chaser_states` and `/video_metadata/frame_metadata`
- Validate the fix after completion

Options:
- `--fix-all` - Fix both chaser_states and metadata (recommended)
- `--fix-metadata` - Fix only metadata
- `--no-backup` - Skip backup creation (not recommended)
- `--backup /custom/path.h5` - Use custom backup location

#### Step 3: Re-import to Zarr
```bash
scripts/py -m fisheye.analysis.import_stimulus_to_zarr /path/to/archive.zarr
```

This creates a new timestamped run with corrected frame mappings.

#### Step 4: Verify Fix

**Check mapping success rate**:
```bash
scripts/py -m fisheye.analysis.diagnose_camera_chaser_mapping /path/to/archive.zarr
```
Should show: ~100% successful mappings (down from ~50%)

**Visualize alignment**:
```bash
scripts/py -m fisheye.analysis.plot_chaser_alignment /path/to/archive.zarr
```
Should show: Significant reduction in red dots (missing samples)

**Check chaser positions**:
```bash
scripts/py -m fisheye.diagnostics.check_h5_chaser_positions /path/to/stimulus.h5
```
Should show: Positions varying over time

### Expected Results

After fixing and re-importing:
- **Camera→chaser mapping**: Improves from ~50% to ~100% success rate
- **Data coverage**: Improves from ~50% to ~98% valid chaser positions
- **Remaining gaps**: 2-3% gaps are legitimate data drops, not frame numbering issues

### Recovery from Backup

If something goes wrong:
```bash
cp /path/to/stimulus.h5.bak /path/to/stimulus.h5
```

Or use the original backup:
```bash
cp /path/to/stimulus.h5.original /path/to/stimulus.h5
```

### Prevention

**Upstream fix**: The stimulus logging code should be updated to prevent double-incrementing the frame counter. This will eliminate the need for post-processing fixes on new recordings.

For 120fps stimulus with 60fps camera:
- Frame counter should increment by 1 each frame: `0, 1, 2, 3...`
- Chaser states logged at 60fps naturally skip frames but use correct frame numbers
- No need to "double count" to match camera rate

## Summary

**What works today:**
- ✅ H5 data is preserved (never modified)
- ✅ Zarr imports create clean, contiguous datasets
- ✅ Interpolation fills gaps automatically
- ✅ Masks allow filtering back to original data
- ✅ All diagnostic scripts are functional
- ✅ Each import is independent (no re-interpolation risk)
- ✅ Frame numbering fix available to correct double-incrementing bug

**Design philosophy:**
The current implementation prioritizes **simplicity for downstream consumers** - use the interpolated data directly, filter to original when needed via masks. This approach has proven sufficient for production workflows.

Once the frame numbering bug is fixed at the H5 source level (using `fix_h5_chaser_frame_numbers.py`), the standard alignment arrays provide accurate, sequential frame mappings with ~100% camera→chaser mapping success.
