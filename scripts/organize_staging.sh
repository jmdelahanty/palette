#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"

# Environment variables:
# - PALETTE_STAGING_ROOT: default source root when no positional source is given.
# - PALETTE_RECORDINGS_ROOT: default dest root for organized recordings.
# - PALETTE_LOG_ROOT: default log directory for JSONL logs.

if [[ $# -eq 0 ]]; then
  set -- --process-all --require-done --recursive --cleanup-staging \
    --dest-root "${PALETTE_RECORDINGS_ROOT:-/nvme1/recordings}" \
    --rename-cams \
    --apply \
    --write-manifest \
    --cleanup-empty
else
  has_source=false
  for arg in "$@"; do
    if [[ "$arg" != -* ]]; then
      has_source=true
      break
    fi
  done
  if [[ "$has_source" == false ]]; then
    set -- --process-all --require-done --recursive --cleanup-staging "$@"
  fi
fi

"${PYTHON_BIN}" "${REPO_ROOT}/src/fisheye/utils/organize_recordings.py" "$@"
