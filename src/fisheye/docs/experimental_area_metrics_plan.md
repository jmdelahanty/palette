# Plan: Experimental Area Metrics

## Motivation
- Provide downstream analyses with fish distance to the arena center and boundary so behaviors near edges are quantifiable.
- Reuse calibration artifacts already captured in Citrus HDF5 snapshots; avoid manual annotation.
- Keep computation deterministic and reproducible across imports and analysis reruns.

## Inputs We Already Have
- `calibration_snapshot/arena_config_json` in the source HDF5 file carries shape, center, radii/width/height, and optional corner radius in projector pixels.
- Per-camera calibration groups (e.g. `calibration_snapshot/<camera_id>/`) store homography matrices (`homography_matrix_yml`) and pixel-per-millimeter ratios (`pixels_per_mm_*`).
- Zarr analysis runs already reference the source `stimulus` group and expose `source_h5` attributes that point back to the HDF5 calibration bundle.
- Fish positions in camera coordinates exist throughout the pipeline (`bbox_centers`, keypoint centroids, track positions) and can be paired with per-frame pixels-per-mm metadata.

## Proposed Processing Flow
1. **Load Arena Definition**
   - Parse `arena_config_json` for shape parameters.
   - Decide projector canvas dimensions; prefer explicit values if available in calibration metadata, otherwise fall back to known defaults (e.g. `(1920, 1080)`).
2. **Synthesize Projector-Space Mask**
   - For circles: draw filled circle with `center` and `radius_px`.
   - For rectangles: draw filled rectangle with optional rounded corners; ensure anti-aliasing is disabled for deterministic masks.
   - Store mask as `uint8` with values `{0, 255}` for compatibility with OpenCV distance transforms.
3. **Transform to Camera Space**
   - Invert the homography matrix (`H⁻¹`) to map projector coordinates into camera pixels.
   - Warp projector mask into each camera’s resolution; derive camera dimensions from Zarr video metadata or fallback defaults.
   - (Optional) generate a distance transform image once per camera to avoid recomputation later.
4. **Distance Metrics**
   - **Boundary distance (px):** sample distance transform of camera mask at each fish position; return negative distances for points outside mask using inverted mask transform.
   - **Center distance (px/mm):** compute Euclidean distance between fish position and arena center projected into camera coordinates; convert to millimeters via per-frame pixels-per-mm metadata when available.
   - Package pixels-per-mm conversions carefully: missing or zero ratios should yield `NaN`, not zero.
5. **Aggregation and Storage**
   - Write per-frame arrays (`distance_to_boundary_px`, `distance_to_boundary_mm`, `distance_to_center_px`, `distance_to_center_mm`) into analysis Zarr runs.
   - Expose mask datasets (`experimental_area_mask_camera`, `experimental_area_mask_projector`) if we decide to precompute during import so that downstream tooling can visualize boundaries directly.

## Integration Points
- **Import Pipeline (`src/fisheye/analysis/import_stimulus_to_zarr.py`):**
  - Option A: generate and store mask datasets during import to Zarr.
  - Option B: store arena config parameters and compute masks lazily inside analysis modules.
- **Metrics Computation (`compute_chaser_fish_metrics.py`, `movement_analysis.py`, `swim_bout_statistics.py`):**
  - Load masks on-demand; cache per-run to avoid repeated HDF5 reads.
  - Add distance arrays to existing per-frame/per-fish data structures before writing to Zarr.
- **Visualization (`visualize_experiment_timeline.py`, `visualize_detect_quality.py`, etc.):**
  - Overlay mask boundaries on plots for QA.
- **Tracking (`assign_ids.py`, `refine_detect.py`):**
  - Optional use: mask can filter out detections outside arena prior to linking.

## Validation Strategy
- Unit-test mask synthesis for representative configurations:
  - Circle vs rectangle with/without corner radius.
  - Multiple projector resolutions.
  - Homography inversion sanity checks (round-trip known points).
- Integration test on a small HDF5/Zarr fixture:
  - Confirm stored mask datasets match on-demand regeneration.
  - Verify distance metrics stay finite for in-bounds tracks and negative outside.
- Visual smoke tests:
  - Plot mask overlay on sample frames to confirm alignment.

## Open Questions & Decisions
- Should we persist the camera mask inside Zarr to avoid recomputation, or derive on-the-fly each time?
- Do we need distances per detection, per track, or both? (Impacts storage volume.)
- Is pixels-per-mm constant over time, or do we need per-frame calibration to ensure accurate millimeter conversions?
- How do we handle multi-camera sessions? (Separate masks per camera, or require camera selection in metrics.)
- Where should we surface failures (e.g., missing homography or arena config)?

## Next Steps
1. Prototype mask generation helper module (no integration).
2. Decide on storage strategy (precompute vs lazy).
3. Add metrics to one analysis pipeline and validate on sample data.
4. Roll out to remaining analyses and visualizations once behavior is vetted.
