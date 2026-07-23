# Keypoint Review Status: Storage + Readback Notes

This note summarizes why `keypoint_review_status` did not show up immediately
and how we made it reliable.

## What happened
- Manual approval **did** run and printed success.
- The refined run’s `keypoint_review_status` attribute was **not visible** via
  the Zarr API (`run.attrs.get(...)` returned `None`).
- The value **was present on disk** in `zarr.json` for the refined run after
  setting it with the CLI.

## Root cause
We’re on Zarr v3 stores where group metadata is kept in `zarr.json`. The Zarr
API sometimes returns stale or empty attributes for newly written fields unless
the attributes are written via a full `.attrs.put(...)` or read directly from
the file. Consolidated metadata isn’t guaranteed to reflect new fields, either.

## Fixes applied
1. **Write path**  
   We now write review status using `.attrs.put(...)`:
   - `src/fisheye/tune/keypoint_failure_review.py`
   - `src/fisheye/utils/set_keypoint_review_status.py`

2. **Read path (fallback)**  
   When `run.attrs.get("keypoint_review_status")` is `None`, we read directly
   from `zarr.json` (or `.zattrs` for v2):
   - `src/fisheye/utils/show_keypoint_review_status.py`
   - `src/fisheye/utils/check_recording_steps.py`

This makes the status visible in both the CLI and the review table even when
the live attrs cache is stale.

## Practical takeaway
- Approvals are stored on the refined run.
- If you don’t see them, it’s likely a **readback** issue, not a failed write.
- The current tooling now handles this automatically.

## Optional troubleshooting
- Use `show_keypoint_review_status --show-raw` to print the raw attr and
  on-disk value.
