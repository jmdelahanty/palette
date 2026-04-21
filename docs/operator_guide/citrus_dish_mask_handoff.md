# Citrus Agent Handoff: Runtime Dish Mask Creation

This document is a prompt/reference for the Citrus data acquisition agent. It
describes how Palette currently creates and consumes dish masks, and what
Citrus would need to write into its H5 output so Palette can skip the manual
tuning cycle entirely.

---

## Context: why this matters

Today, after every recording session, an operator must:

1. Open an interactive tuner on one recording per camera
2. Adjust a circle to fit the dish boundary
3. Batch-propagate the Hough parameters to other recordings
4. Manually review each recording because dish center/scale drifts

This is tedious and error-prone. Citrus already knows the dish boundary at
runtime — it has the camera feed and arena configuration. If Citrus writes the
dish mask into the H5 file, Palette can import it automatically and the entire
tuning cycle disappears.

---

## How Palette currently consumes dish masks

The dish mask lives on each analysis Zarr at:

```
analysis_metadata.attrs["dish_mask"]
```

It is read by:
- **Traditional detection** (`detect_traditional.py`) — zeros out pixels
  outside the dish before blob detection
- **Subject segmentation** (`subject_segmentation.py`) — gates body mask
  proposals to the dish region
- **Subject mask tuner** — uses dish mask in preview overlays
- **Arena assignment** — uses dish boundary for spatial gating

### The payload schema

The `dish_mask` attribute is a JSON-like dict with this structure:

#### Circle mask (most common)

```json
{
  "shape": "circle",
  "version": "2.0",
  "method": "hough_circle",
  "tuned_timestamp": "2026-01-28T19:45:00",
  "tuned_on_array": "images_ds",
  "tuned_on_frame": 500,
  "source": {
    "array": "images_ds",
    "frame": 500
  },
  "detected_circle": {
    "center": [328, 310],
    "radius": 295
  },
  "hough_params": {
    "param1": 50,
    "param2": 30,
    "radius_adjustment": 4
  },
  "metrics": {
    "image_shape": [640, 640],
    "center_px": [328, 310],
    "center_norm": [0.5125, 0.484375],
    "radius_px": 295,
    "radius_norm": 0.4609375,
    "area_px": 273397.2,
    "area_fraction": 0.6676
  }
}
```

#### Rectangle mask (rare, for non-circular dishes)

```json
{
  "shape": "rectangle",
  "version": "2.0",
  "method": "manual_rectangle",
  "tuned_timestamp": "2026-01-28T19:45:00",
  "tuned_on_array": "images_ds",
  "tuned_on_frame": 500,
  "source": {
    "array": "images_ds",
    "frame": 500
  },
  "rectangle": {
    "roi": [50, 40, 540, 560]
  },
  "metrics": {
    "image_shape": [640, 640],
    "center_px": [320, 320],
    "center_norm": [0.5, 0.5],
    "area_px": 302400.0,
    "area_fraction": 0.7383
  }
}
```

### How the circle is consumed at runtime

Downstream code extracts the circle and creates a binary mask image:

```python
center = dish_mask["detected_circle"]["center"]  # [x, y] in pixels
radius = dish_mask["detected_circle"]["radius"]   # in pixels
# Creates a uint8 image: 255 inside the circle, 0 outside
cv2.circle(mask, (center[0], center[1]), radius, 255, -1)
```

The coordinate space depends on `tuned_on_array`:
- `"images_ds"` — coordinates are in the downsampled frame space (e.g.
  640x640)
- `"images_full"` — coordinates are in the full camera resolution

Most dish masks are tuned on `images_ds`. When used on full-resolution data,
Palette scales the coordinates proportionally.

---

## What Citrus should write into the H5

Citrus should write a **dish mask group or set of attributes** into the H5
root that Palette's import step can read and transplant into
`analysis_metadata.attrs["dish_mask"]` on the Zarr.

### Recommended H5 structure

Write these as H5 root attributes (alongside existing attributes like
`session_uuid`, `camera_id`, etc.):

```
dish_mask_shape           : str    — "circle" or "rectangle"
dish_mask_center_x        : int    — circle center X in camera pixels
dish_mask_center_y        : int    — circle center Y in camera pixels
dish_mask_radius          : int    — circle radius in camera pixels
dish_mask_image_width     : int    — camera frame width used for detection
dish_mask_image_height    : int    — camera frame height used for detection
dish_mask_method          : str    — how it was detected (e.g. "citrus_runtime_hough", "citrus_arena_config")
```

For rectangle masks (if needed):

```
dish_mask_shape           : str    — "rectangle"
dish_mask_roi_x           : int    — top-left X
dish_mask_roi_y           : int    — top-left Y
dish_mask_roi_w           : int    — width
dish_mask_roi_h           : int    — height
dish_mask_image_width     : int
dish_mask_image_height    : int
dish_mask_method          : str
```

Alternatively, write a single JSON-encoded attribute:

```
dish_mask_json : str — JSON string with the full payload
```

### Coordinate space

The coordinates must be in the **camera's native resolution** (the resolution
of the frames being written to the camera MP4). Palette will handle any
downscaling needed for its internal arrays.

This is important: Palette's existing masks are often tuned on 640x640
downsampled images. If Citrus writes coordinates in the camera's native
resolution (e.g. 1920x1200), the import step needs to know the source
resolution so it can convert. That's what `dish_mask_image_width` and
`dish_mask_image_height` are for.

### What Citrus already knows

Citrus has access to:
- The camera feed dimensions
- The arena configuration (`arena_config_json`) which includes dish type,
  calibration data, and often the dish boundary
- The Hough circle detection it already runs for arena setup

The simplest path is to snapshot whatever circle/rectangle Citrus uses
internally for arena boundary detection and write it to the H5.

---

## What Palette needs to change (on the import side)

Once Citrus writes the dish mask, Palette's import step
(`import_recording_analysis.py` or the organize stage) needs to:

1. Check if the H5 has `dish_mask_*` attributes
2. If present, construct the `dish_mask` payload in Palette's schema
3. Write it to `analysis_metadata.attrs["dish_mask"]` on the Zarr
4. Set `method` to something like `"citrus_runtime"` so operators can
   distinguish runtime masks from manually tuned ones

This is a small change on the Palette side — the import already reads many H5
attributes and writes them to the Zarr.

---

## Validation

A Citrus-provided dish mask should be treated as a strong initial guess but
not necessarily final. Palette should:

- Accept it without requiring manual tuning
- Allow operators to override it via the existing tuner if needed (the tuner
  already handles pre-existing masks)
- Log the mask source (`citrus_runtime` vs `hough_circle` vs
  `manual_rectangle`) so provenance is clear

---

## Summary of what to implement

**Citrus side:**
1. At session end (or during arena setup), detect/extract the dish boundary
2. Write `dish_mask_*` attributes to the H5 root alongside existing metadata
3. Include the coordinate space dimensions so Palette can rescale

**Palette side:**
1. In the import step, read `dish_mask_*` from the H5
2. Convert to the `analysis_metadata.attrs["dish_mask"]` schema
3. Rescale coordinates from camera resolution to the target array resolution
4. Skip the tuning prompt when a Citrus-provided mask is present
