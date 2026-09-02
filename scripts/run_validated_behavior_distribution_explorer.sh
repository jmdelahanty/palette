#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/run_validated_behavior_distribution_explorer.sh [options]
  scripts/run_validated_behavior_distribution_explorer.sh /exact/distribution/dir

Starts the read-only validated-behavior distribution explorer for use through
an SSH tunnel.

Options:
  --distribution-dir PATH  Exact immutable distribution generation.
  --metric ID              Initial exact metric ID.
  --port PORT              Workstation port. (default: 2722)
  --host HOST              Host interface. (default: 127.0.0.1)
  --token                   Enable marimo token authentication.
  --no-token                Disable token authentication. (default)
  --watch                   Reload when the app source changes.
  --include-code            Include notebook code in the UI.
  -h, --help                Show this help.

The path may also be supplied with PALETTE_VALIDATED_BEHAVIOR_DISTRIBUTION_DIR.
EOF
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

DISTRIBUTION_DIR="${PALETTE_VALIDATED_BEHAVIOR_DISTRIBUTION_DIR:-}"
PORT="${PALETTE_DISTRIBUTION_EXPLORER_PORT:-2722}"
HOST="${PALETTE_DISTRIBUTION_EXPLORER_HOST:-127.0.0.1}"
TOKEN_ARG="--no-token"
MARIMO_ARGS=()
APP_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --distribution-dir)
      DISTRIBUTION_DIR="$2"
      shift 2
      ;;
    --metric)
      APP_ARGS+=("--metric" "$2")
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
      DISTRIBUTION_DIR="$1"
      shift
      ;;
  esac
done

if [[ -z "$DISTRIBUTION_DIR" ]]; then
  echo "An exact distribution directory is required." >&2
  usage >&2
  exit 2
fi
if [[ ! -d "$DISTRIBUTION_DIR" ]]; then
  echo "Distribution directory not found: $DISTRIBUTION_DIR" >&2
  exit 2
fi
if [[ ! "$PORT" =~ ^[0-9]+$ ]]; then
  echo "Port must be an integer: $PORT" >&2
  exit 2
fi

mkdir -p /tmp/palette-matplotlib /tmp/palette-pycache

echo "Starting Validated Behavior Distribution Explorer"
echo "  distribution_dir: $DISTRIBUTION_DIR"
echo "  workstation URL: http://$HOST:$PORT"
echo
echo "From your laptop, run:"
echo "  ssh -N -L $PORT:127.0.0.1:$PORT <your-workstation-ssh-host>"
echo
echo "Then open: http://localhost:$PORT"
echo "Press Ctrl-C here to stop the marimo server."

exec env \
  MPLCONFIGDIR=/tmp/palette-matplotlib \
  PYTHONPYCACHEPREFIX=/tmp/palette-pycache \
  scripts/py -m marimo run \
    --headless \
    "$TOKEN_ARG" \
    --host "$HOST" \
    -p "$PORT" \
    "${MARIMO_ARGS[@]}" \
    apps/marimo/validated_behavior_distribution_explorer.py -- \
    --distribution-dir "$DISTRIBUTION_DIR" \
    "${APP_ARGS[@]}"
