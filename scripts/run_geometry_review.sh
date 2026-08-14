#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

HOST="${PALETTE_GEOMETRY_REVIEW_HOST:-127.0.0.1}"
PORT="${PALETTE_GEOMETRY_REVIEW_PORT:-8772}"
TOKEN="${PALETTE_GEOMETRY_REVIEW_TOKEN:-}"

usage() {
  cat <<'EOF'
Usage:
  scripts/run_geometry_review.sh --registry /path/palette_registry.sqlite [--run-id RUN]
  scripts/run_geometry_review.sh --zarr-path /path/recording_analysis.zarr [--run-id RUN]

Registry mode reads queue paths from SQLite and opens only the selected Zarr.
Direct mode opens exactly one explicit canonical analysis Zarr.

Optional environment:
  PALETTE_GEOMETRY_REVIEW_HOST   bind host (default 127.0.0.1)
  PALETTE_GEOMETRY_REVIEW_PORT   bind port (default 8772)
  PALETTE_GEOMETRY_REVIEW_TOKEN  Marimo access token
EOF
}

if [[ $# -lt 2 ]]; then
  usage >&2
  exit 2
fi

APP_ARGS=("$@")
MODE_COUNT=0
for arg in "$@"; do
  if [[ "$arg" == "--registry" || "$arg" == "--zarr-path" ]]; then
    MODE_COUNT=$((MODE_COUNT + 1))
  fi
done
if [[ "$MODE_COUNT" -ne 1 ]]; then
  echo "Choose exactly one of --registry or --zarr-path." >&2
  usage >&2
  exit 2
fi

MARIMO_ARGS=(
  -m marimo run apps/marimo/geometry_review.py
  --host "$HOST"
  --port "$PORT"
  --headless
)
if [[ -n "$TOKEN" ]]; then
  MARIMO_ARGS+=(--token-password "$TOKEN")
fi

exec scripts/py "${MARIMO_ARGS[@]}" -- "${APP_ARGS[@]}"
