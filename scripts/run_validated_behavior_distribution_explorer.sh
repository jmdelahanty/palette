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
  --export-root PATH       Exact validated-behavior publication root. Resolves
                           a co-located distribution through its product catalog.
  --source-export-run-id ID
                           Exact export run ID required with --export-root.
  --distribution-run-id ID
                           Exact cataloged distribution run. May be omitted only
                           when the catalog contains one compatible distribution.
  --metric ID              Initial exact metric ID.
  --port PORT              Workstation port. (default: 2722)
  --host HOST              Host interface. (default: 127.0.0.1)
  --token                   Enable marimo token authentication.
  --no-token                Disable token authentication. (default)
  --watch                   Reload when the app source changes.
  --include-code            Include notebook code in the UI.
  -h, --help                Show this help.

The direct path may also be supplied with
PALETTE_VALIDATED_BEHAVIOR_DISTRIBUTION_DIR. Catalog discovery may use
PALETTE_VALIDATED_BEHAVIOR_EXPORT_ROOT,
PALETTE_VALIDATED_BEHAVIOR_EXPORT_RUN_ID, and
PALETTE_VALIDATED_BEHAVIOR_DISTRIBUTION_RUN_ID.
EOF
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

DISTRIBUTION_DIR="${PALETTE_VALIDATED_BEHAVIOR_DISTRIBUTION_DIR:-}"
EXPORT_ROOT="${PALETTE_VALIDATED_BEHAVIOR_EXPORT_ROOT:-}"
SOURCE_EXPORT_RUN_ID="${PALETTE_VALIDATED_BEHAVIOR_EXPORT_RUN_ID:-}"
DISTRIBUTION_RUN_ID="${PALETTE_VALIDATED_BEHAVIOR_DISTRIBUTION_RUN_ID:-}"
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
    --export-root)
      EXPORT_ROOT="$2"
      shift 2
      ;;
    --source-export-run-id)
      SOURCE_EXPORT_RUN_ID="$2"
      shift 2
      ;;
    --distribution-run-id)
      DISTRIBUTION_RUN_ID="$2"
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

if [[ -n "$DISTRIBUTION_DIR" && -n "$EXPORT_ROOT" ]]; then
  echo "Choose either --distribution-dir or --export-root, not both." >&2
  usage >&2
  exit 2
fi
if [[ -z "$DISTRIBUTION_DIR" && -z "$EXPORT_ROOT" ]]; then
  echo "An exact distribution directory or export root is required." >&2
  usage >&2
  exit 2
fi
if [[ -n "$DISTRIBUTION_DIR" && ! -d "$DISTRIBUTION_DIR" ]]; then
  echo "Distribution directory not found: $DISTRIBUTION_DIR" >&2
  exit 2
fi
if [[ -n "$EXPORT_ROOT" && ! -d "$EXPORT_ROOT" ]]; then
  echo "Validated-behavior export root not found: $EXPORT_ROOT" >&2
  exit 2
fi
if [[ -n "$EXPORT_ROOT" && -z "$SOURCE_EXPORT_RUN_ID" ]]; then
  echo "--export-root requires --source-export-run-id." >&2
  exit 2
fi
if [[ -z "$EXPORT_ROOT" && ( -n "$SOURCE_EXPORT_RUN_ID" || -n "$DISTRIBUTION_RUN_ID" ) ]]; then
  echo "Export and distribution run IDs require --export-root." >&2
  exit 2
fi
if [[ ! "$PORT" =~ ^[0-9]+$ ]]; then
  echo "Port must be an integer: $PORT" >&2
  exit 2
fi

mkdir -p /tmp/palette-matplotlib /tmp/palette-pycache

echo "Starting Validated Behavior Distribution Explorer"
SOURCE_ARGS=()
if [[ -n "$DISTRIBUTION_DIR" ]]; then
  echo "  distribution_dir: $DISTRIBUTION_DIR"
  SOURCE_ARGS+=("--distribution-dir" "$DISTRIBUTION_DIR")
else
  echo "  export_root: $EXPORT_ROOT"
  echo "  source_export_run_id: $SOURCE_EXPORT_RUN_ID"
  echo "  distribution_run_id: ${DISTRIBUTION_RUN_ID:-unique compatible product}"
  SOURCE_ARGS+=(
    "--export-root" "$EXPORT_ROOT"
    "--source-export-run-id" "$SOURCE_EXPORT_RUN_ID"
  )
  if [[ -n "$DISTRIBUTION_RUN_ID" ]]; then
    SOURCE_ARGS+=("--distribution-run-id" "$DISTRIBUTION_RUN_ID")
  fi
fi
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
    "${SOURCE_ARGS[@]}" \
    "${APP_ARGS[@]}"
