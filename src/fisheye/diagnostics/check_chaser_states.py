# #!/usr/bin/env python3
# import argparse
# from pathlib import Path

# import numpy as np
# import zarr

# TRIAL_STATE_LABELS = {0: "PRE_PERIOD", 1: "TRAINING", 2: "POST_PERIOD"}

# def main() -> None:
#     parser = argparse.ArgumentParser(
#         description="Report trial_state counts for chaser_states in a Palette Zarr run."
#     )
#     parser.add_argument("zarr_path", type=Path, help="Path to the Palette Zarr archive.")
#     parser.add_argument(
#         "--run",
#         dest="run_name",
#         help="Name under analysis/stimulus_runs/. Defaults to archive’s latest attribute.",
#     )
#     args = parser.parse_args()

#     root = zarr.open(str(args.zarr_path), mode="r")
#     runs_parent = root["analysis/stimulus_runs"]
#     run_name = args.run_name or runs_parent.attrs.get("latest")
#     if run_name is None or run_name not in runs_parent:
#         raise SystemExit(f"Unable to resolve run name (got {run_name!r}).")

#     chaser = runs_parent[f"{run_name}/tracking_data/chaser_states"][:]
#     if "trial_state" not in chaser.dtype.names:
#         raise SystemExit("trial_state field missing from chaser_states dataset.")

#     states, counts = np.unique(chaser["trial_state"], return_counts=True)
#     print(f"Run: {run_name}")
#     for state, count in zip(states, counts):
#         label = TRIAL_STATE_LABELS.get(int(state), f"STATE_{int(state)}")
#         print(f"  {int(state)} ({label}): {int(count)} rows")
#     missing = [s for s in TRIAL_STATE_LABELS if s not in states]
#     if missing:
#         print("Missing states:", ", ".join(f"{s} ({TRIAL_STATE_LABELS[s]})" for s in missing))

# if __name__ == "__main__":
#     main()
# import numpy as np, zarr

# root = zarr.open("/nvme1/sesh3/2025-09-23T22-11-11Z_arena_4_chaser_arena4.zarr/", mode="r")
# chaser = root["analysis/stimulus_runs/stimulus_20251030_191209/tracking_data/chaser_states"][:]
# kp = root["keypoints_runs/keypoints_2025-10-25_20-20-57"]

# stim_frames = np.unique(chaser["stimulus_frame_num"])
# roi_frames = np.unique(kp["frame_indices"])

# only_in_keypoints = np.setdiff1d(roi_frames, stim_frames)
# print("ROI frames without chaser state:", only_in_keypoints.size)

import numpy as np, zarr
root = zarr.open("/nvme1/sesh3/2025-09-23T22-11-11Z_arena_4_chaser_arena4.zarr/", mode="r")
chaser = root["analysis/stimulus_runs/stimulus_20251030_191209/tracking_data/chaser_states"]
kp = root["keypoints_runs/keypoints_2025-10-25_20-20-57"]
stim_ids = np.unique(chaser["stimulus_frame_num"])
camera_ids = np.unique(kp["frame_indices"])
print("Stim-only frames:", np.setdiff1d(stim_ids, camera_ids).size)
print("Camera-only frames:", np.setdiff1d(camera_ids, stim_ids).size)
