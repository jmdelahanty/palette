#!/usr/bin/env bash
set -euo pipefail

REG="${1:-${PALETTE_REGISTRY_PATH:-/groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite}}"
MANIFEST="${2:-/nvme1/training/datasets/eye_mask_cedar_shadow_omnifin0_auto_gray_lr_b9164009_v002/eye_mask_cedar_shadow_omnifin0_auto_gray_lr_b9164009_v002.manifest.json}"
MAP="${3:-/tmp/eye_mask_review_fix_rows.tsv}"

if [[ ! -f "$MANIFEST" ]]; then
  echo "Manifest not found: $MANIFEST" >&2
  exit 1
fi

echo "Registry: $REG"
echo "Manifest: $MANIFEST"
echo "Map file: $MAP"

MANIFEST="$MANIFEST" MAP="$MAP" scripts/py - <<'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import zarr

manifest_path = Path(os.environ["MANIFEST"])
map_path = Path(os.environ["MAP"])
payload = json.loads(manifest_path.read_text(encoding="utf-8"))
exclusions = payload.get("quality_exclusions") or []

now = datetime.now(timezone.utc).isoformat()
reviewer = os.environ.get("USER", "unknown")
rows: list[tuple[str, str, str]] = []

def pick_source_run(root: zarr.Group):
    for stage in ("refined_eye_masks_runs", "eye_masks_runs"):
        parent = root.get(stage)
        if parent is None:
            continue
        latest = parent.attrs.get("latest")
        if latest and latest in parent:
            run = str(latest)
            return stage, run, parent[run], parent
        keys = sorted(parent.group_keys()) if hasattr(parent, "group_keys") else sorted(parent.keys())
        if keys:
            run = str(keys[-1])
            return stage, run, parent[run], parent
    return None

for row in exclusions:
    ds = str(row.get("dataset_id") or "").strip()
    zp = str(row.get("zarr_path") or "").strip()
    if not ds or not zp:
        continue
    zarr_path = Path(zp)
    if not zarr_path.exists():
        print(f"skip missing zarr: {ds} {zarr_path}")
        continue
    try:
        try:
            root = zarr.open_group(str(zarr_path), mode="a", use_consolidated=False)
        except TypeError:
            root = zarr.open_group(str(zarr_path), mode="a")
    except Exception as exc:
        print(f"skip open error: {ds} {zarr_path} ({exc})")
        continue

    picked = pick_source_run(root)
    if picked is None:
        print(f"skip no eye-mask run: {ds} {zarr_path}")
        continue

    stage, run, grp, parent = picked
    grp.attrs["eye_mask_review_status"] = {
        "state": "approved",
        "method": "manual",
        "intended_use": "training",
        "reviewer": reviewer,
        "timestamp": now,
        "notes": "approved for training after manual review",
    }
    parent.attrs["eye_mask_review_status_latest"] = run
    rows.append((ds, str(zarr_path), f"{stage}/{run}"))

map_path.write_text(
    "".join("\t".join(item) + "\n" for item in rows),
    encoding="utf-8",
)
print(f"updated_rows={len(rows)}")
print(f"wrote_map={map_path}")
PY

if [[ ! -s "$MAP" ]]; then
  echo "No excluded rows were updated. Nothing else to do."
  exit 0
fi

echo
echo "Updated rows:"
cat "$MAP"
echo

echo "Backfilling eye-mask profiles for updated rows..."
while IFS=$'\t' read -r ds zarr_path source_path; do
  [[ -z "${ds:-}" ]] && continue
  scripts/py -m fisheye.utils.backfill_eye_mask_profiles "$zarr_path" \
    --zarr-use training \
    --source-eye-mask-path "$source_path" \
    --registry "$REG" \
    --apply
done < "$MAP"

echo
echo "Syncing eye_mask_data_profile_latest rows..."
SYNC_ARGS=()
while IFS=$'\t' read -r ds _rest; do
  [[ -z "${ds:-}" ]] && continue
  SYNC_ARGS+=(--dataset-id "$ds")
done < "$MAP"

scripts/py -m fisheye.utils.sync_eye_mask_profile_registry \
  --registry "$REG" \
  --zarr-use training \
  --apply \
  "${SYNC_ARGS[@]}"

echo
echo "Done. Re-run your pipeline command."
