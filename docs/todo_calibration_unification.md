# TODO: Calibration metadata unification

We currently expose chaser calibration in two places:

1. `analysis/stimulus_runs/<run>.attrs['coordinate_transform']`
   * Texture/camera dimensions and `texture_to_camera_scale`.
   * Legacy summary used by existing tooling.

2. `analysis/stimulus_runs/<run>/calibration/<camera_id>`
   * Full snapshot copied from the stimulus H5 calibration block.
   * Contains `pixels_per_mm_projector`, `pixels_per_mm_camera`, `homography_matrix_yml`, etc.

Follow-up tasks:

- Decide on a single canonical location for calibration data.
- Update downstream consumers (offline metrics, visualization, diagnostics) to read from that location.
- Remove redundant data or add explicit pointers so analysts are not confused by duplicate metadata.
- Consider normalising homography storage (YAML vs structured array) before freezing the schema.
