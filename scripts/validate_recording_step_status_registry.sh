#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/validate_recording_step_status_registry.sh [options]

Validates recording step status registry parity for one recording end-to-end:
1) optional scoped backfill refresh for the target recording
2) compare-mode parity check against filesystem state
3) deterministic SQL checks for required views/rows

Options:
  --recording-dir DIR        Recording directory (required)
  --registry PATH            Registry SQLite path
                             (default: $PALETTE_REGISTRY_PATH or /groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite)
  --zarr-use USE             Zarr use to validate: training|analysis
                             (default: training)
  --recording-id ID          Recording ID override (default: auto-derived)
  --skip-backfill            Skip maintenance backfill refresh step
  --tmp-root DIR             Temp parent directory (default: /tmp)
  -h, --help                 Show this help
EOF
}

RECORDING_DIR=""
REGISTRY="${PALETTE_REGISTRY_PATH:-/groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite}"
ZARR_USE="training"
RECORDING_ID=""
SKIP_BACKFILL="0"
TMP_ROOT="/tmp"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --recording-dir)
      RECORDING_DIR="$2"
      shift 2
      ;;
    --registry)
      REGISTRY="$2"
      shift 2
      ;;
    --zarr-use)
      ZARR_USE="$2"
      shift 2
      ;;
    --recording-id)
      RECORDING_ID="$2"
      shift 2
      ;;
    --skip-backfill)
      SKIP_BACKFILL="1"
      shift
      ;;
    --tmp-root)
      TMP_ROOT="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "$RECORDING_DIR" ]]; then
  echo "--recording-dir is required." >&2
  usage
  exit 2
fi

if [[ "$ZARR_USE" != "training" && "$ZARR_USE" != "analysis" ]]; then
  echo "--zarr-use must be 'training' or 'analysis'." >&2
  exit 2
fi

if [[ ! -x "scripts/py" ]]; then
  echo "Expected executable wrapper not found: scripts/py" >&2
  exit 2
fi

if [[ ! -d "$RECORDING_DIR" ]]; then
  echo "Recording directory not found: $RECORDING_DIR" >&2
  exit 2
fi

if [[ ! -d "$TMP_ROOT" ]]; then
  mkdir -p "$TMP_ROOT"
fi

if [[ ! -f "$REGISTRY" ]]; then
  echo "Registry not found: $REGISTRY" >&2
  exit 2
fi

if ! command -v rg >/dev/null 2>&1; then
  echo "ripgrep (rg) is required." >&2
  exit 2
fi

RECORDING_DIR="$(cd "$RECORDING_DIR" && pwd)"
RECORDING_NAME="$(basename "$RECORDING_DIR")"
TARGET_ZARR="$RECORDING_DIR/zarr/${RECORDING_NAME}_${ZARR_USE}.zarr"
if [[ ! -d "$TARGET_ZARR" ]]; then
  echo "Target zarr not found for --zarr-use ${ZARR_USE}: $TARGET_ZARR" >&2
  exit 2
fi

if [[ -z "$RECORDING_ID" ]]; then
  RECORDING_ID="$(scripts/py - "$RECORDING_DIR" <<'PY'
import sys
from pathlib import Path

recording_dir = Path(sys.argv[1])
recording_id = ""

try:
    import h5py  # type: ignore
except Exception:
    h5py = None

if h5py is not None:
    for h5_path in sorted((recording_dir / "h5").glob("*.h5")):
        try:
            with h5py.File(h5_path, "r") as handle:
                for key in ("recording_id", "session_uuid"):
                    value = handle.attrs.get(key)
                    if value is None:
                        continue
                    if isinstance(value, bytes):
                        value = value.decode("utf-8", "ignore")
                    text = str(value).strip()
                    if text:
                        recording_id = text
                        break
                if recording_id:
                    break
        except Exception:
            continue

print(recording_id)
PY
)"
fi

if [[ -z "$RECORDING_ID" ]]; then
  if [[ "$RECORDING_NAME" =~ ^(.+_arena_[0-9]+) ]]; then
    RECORDING_ID="${BASH_REMATCH[1]}"
  fi
fi

if [[ -z "$RECORDING_ID" ]]; then
  echo "Unable to derive recording id automatically. Pass --recording-id explicitly." >&2
  exit 2
fi

SAFE_RECORDING_NAME="$(printf '%s' "$RECORDING_NAME" | tr -cs 'A-Za-z0-9._-' '_')"
TMP_DIR="$(mktemp -d "${TMP_ROOT%/}/recording_step_status_validate_${SAFE_RECORDING_NAME}_XXXXXX")"

section() {
  printf '\n=== %s ===\n' "$1"
}

echo "TMP_DIR=$TMP_DIR"
echo "REGISTRY=$REGISTRY"
echo "RECORDING_DIR=$RECORDING_DIR"
echo "RECORDING_ID=$RECORDING_ID"
echo "ZARR_USE=$ZARR_USE"
echo "TARGET_ZARR=$TARGET_ZARR"

if [[ "$SKIP_BACKFILL" != "1" ]]; then
  section "Scoped Backfill Refresh"
  scripts/py -m fisheye.registry.maintenance \
    --registry "$REGISTRY" \
    --backfill-recording-step-status \
    --recording-step-zarr-use "$ZARR_USE" \
    --recording-step-recording-id "$RECORDING_ID" \
    "$RECORDING_DIR" \
    | tee "$TMP_DIR/backfill.log"

  if ! rg -q "Recording step status summary JSON:" "$TMP_DIR/backfill.log"; then
    echo "Backfill did not emit summary JSON line; cannot validate refresh deterministically." >&2
    exit 1
  fi
fi

section "Parity Compare (Filesystem vs Registry)"
if ! scripts/py -m fisheye.utils.check_recording_steps \
  "$RECORDING_DIR" \
  --recursive \
  --zarr-use "$ZARR_USE" \
  --registry "$REGISTRY" \
  --status-source compare \
  --no-rich \
  >"$TMP_DIR/compare.log" 2>&1; then
  cat "$TMP_DIR/compare.log"
  echo "compare command failed." >&2
  exit 1
fi
cat "$TMP_DIR/compare.log"

if ! rg -q "No status mismatches found\\." "$TMP_DIR/compare.log"; then
  echo "Parity mismatch detected. Expected 'No status mismatches found.' output." >&2
  exit 1
fi

section "SQL View/Row Validation"
set +e
SQL_SUMMARY="$(
scripts/py - "$REGISTRY" "$RECORDING_ID" "$ZARR_USE" <<'PY'
import json
import sqlite3
import sys

registry_path, recording_id, zarr_use = sys.argv[1], sys.argv[2], sys.argv[3]

required_views = [
    "recording_step_status_latest",
    "recording_step_overview",
    "recording_step_status_wide",
]
required_steps = [
    "detect",
    "refined_detect",
    "crop",
    "keypoints",
    "refined_keypoints",
    "eye_masks",
    "refined_eye_masks",
    "id_assignment",
    "tracks",
]

summary = {
    "recording_id": recording_id,
    "zarr_use": zarr_use,
}
ok = True
conn = None

try:
    conn = sqlite3.connect(registry_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    view_presence = {}
    for view_name in required_views:
        row = cur.execute(
            "SELECT 1 FROM sqlite_master WHERE type='view' AND name=? LIMIT 1",
            (view_name,),
        ).fetchone()
        view_presence[view_name] = bool(row)
    summary["views"] = view_presence
    if not all(view_presence.values()):
        ok = False

    latest_rows = int(
        cur.execute(
            """
            SELECT COUNT(*)
            FROM recording_step_status_latest
            WHERE recording_id=? AND zarr_use=?
            """,
            (recording_id, zarr_use),
        ).fetchone()[0]
    )
    overview_rows = int(
        cur.execute(
            """
            SELECT COUNT(*)
            FROM recording_step_overview
            WHERE recording_id=?
            """,
            (recording_id,),
        ).fetchone()[0]
    )
    wide_rows = int(
        cur.execute(
            """
            SELECT COUNT(*)
            FROM recording_step_status_wide
            WHERE [Recording]=? AND [Use]=?
            """,
            (recording_id, zarr_use),
        ).fetchone()[0]
    )

    steps = [
        str(row[0])
        for row in cur.execute(
            """
            SELECT DISTINCT step_name
            FROM recording_step_status_latest
            WHERE recording_id=? AND zarr_use=?
            ORDER BY step_name
            """,
            (recording_id, zarr_use),
        ).fetchall()
    ]
    missing_required_steps = [name for name in required_steps if name not in steps]

    summary["latest_rows"] = latest_rows
    summary["overview_rows"] = overview_rows
    summary["wide_rows"] = wide_rows
    summary["distinct_steps"] = steps
    summary["missing_required_steps"] = missing_required_steps

    if latest_rows <= 0 or overview_rows <= 0 or wide_rows <= 0:
        ok = False
    if missing_required_steps:
        ok = False
except Exception as exc:
    summary["error"] = str(exc)
    ok = False
finally:
    if conn is not None:
        conn.close()

summary["ok"] = ok
print(json.dumps(summary, sort_keys=True))
sys.exit(0 if ok else 1)
PY
)"
SQL_RC=$?
set -e

printf '%s\n' "$SQL_SUMMARY" > "$TMP_DIR/sql_validation.json"
cat "$TMP_DIR/sql_validation.json"

if [[ "$SQL_RC" -ne 0 ]]; then
  echo "SQL validation failed. See $TMP_DIR/sql_validation.json" >&2
  exit 1
fi

section "Validation Summary"
echo "status=PASS"
echo "artifacts=$TMP_DIR"
echo "Validated recording step status registry parity successfully."
