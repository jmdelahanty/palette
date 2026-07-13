#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/run_palette_explorer.sh [options]
  scripts/run_palette_explorer.sh /path/to/analysis.zarr

Starts the general Palette marimo explorer on the workstation for use through
an SSH tunnel from a laptop.

Options:
  --zarr-path PATH   Analysis Zarr to open.
                    (default: first backfilled GoodCopBadCop /groups archive)
  --recordings-root PATH
                    Root containing sibling recording directories. The explorer
                    uses this for the top-level recording selector.
  --registry PATH   Optional read-only Palette registry used for lazy recording
                    discovery without opening every Zarr in the collection.
  --recording-name-contains TEXT
                    Recording path/name filter for the selector.
                    (default: GoodCopBadCop)
  --port PORT        Local workstation port for marimo. (default: 2720)
  --host HOST        Host interface for marimo. (default: 127.0.0.1)
  --renderer ID      Initial renderer filter passed to the explorer.
  --run-path PATH    Initial run path filter passed to the explorer.
  --artifact NAME    Initial artifact filter passed to the explorer.
  --token            Enable marimo token authentication.
  --no-token         Disable marimo token authentication. (default)
  --watch            Ask marimo to reload when the notebook changes.
  --include-code     Include notebook code in the app UI.
  -h, --help         Show this help.

Laptop tunnel:
  ssh -N -L <port>:127.0.0.1:<port> <your-workstation-ssh-host>

Then open:
  http://localhost:<port>
EOF
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

DEFAULT_ZARR_PATH="/groups/johnson/johnsonlab/jeremy/recordings/2026-06-14T21-12-08Z_arena_1_GoodCopBadCop/zarr/2026-06-14T21-12-08Z_arena_1_GoodCopBadCop_analysis.zarr"

ZARR_PATH="${PALETTE_EXPLORER_ZARR_PATH:-$DEFAULT_ZARR_PATH}"
PORT="${PALETTE_EXPLORER_PORT:-2720}"
HOST="${PALETTE_EXPLORER_HOST:-127.0.0.1}"
TOKEN_ARG="--no-token"
MARIMO_ARGS=()
APP_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --zarr-path)
      ZARR_PATH="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --host)
      HOST="$2"
      shift 2
      ;;
    --renderer|--run-path|--artifact|--recordings-root|--recording-name-contains|--registry)
      APP_ARGS+=("$1" "$2")
      shift 2
      ;;
    --token)
      TOKEN_ARG="--token"
      shift
      ;;
    --no-token)
      TOKEN_ARG="--no-token"
      shift
      ;;
    --watch|--include-code)
      MARIMO_ARGS+=("$1")
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      APP_ARGS+=("$@")
      break
      ;;
    -*)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      ZARR_PATH="$1"
      shift
      ;;
  esac
done

if [[ ! "$PORT" =~ ^[0-9]+$ ]]; then
  echo "Port must be an integer: $PORT" >&2
  exit 2
fi

if [[ ! -x "scripts/py" ]]; then
  echo "Expected executable wrapper not found: scripts/py" >&2
  exit 2
fi

if [[ ! -f "apps/marimo/palette_explorer.py" ]]; then
  echo "Palette explorer app not found: apps/marimo/palette_explorer.py" >&2
  exit 2
fi

if [[ ! -d "$ZARR_PATH" ]]; then
  echo "Zarr path not found: $ZARR_PATH" >&2
  exit 2
fi

mkdir -p /tmp/palette-matplotlib /tmp/palette-pycache

cat <<EOF
Starting Palette Explorer
  zarr_path: $ZARR_PATH
  workstation URL: http://$HOST:$PORT

From your laptop, run:
  ssh -N -L $PORT:127.0.0.1:$PORT <your-workstation-ssh-host>

Then open:
  http://localhost:$PORT

Press Ctrl-C here to stop the marimo server.
EOF

exec env \
  MPLCONFIGDIR=/tmp/palette-matplotlib \
  PYTHONPYCACHEPREFIX=/tmp/palette-pycache \
  scripts/py -m marimo run \
    --headless \
    "$TOKEN_ARG" \
    --host "$HOST" \
    -p "$PORT" \
    "${MARIMO_ARGS[@]}" \
    apps/marimo/palette_explorer.py -- \
    --zarr-path "$ZARR_PATH" \
    "${APP_ARGS[@]}"
