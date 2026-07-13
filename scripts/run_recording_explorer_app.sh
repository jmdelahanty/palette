#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  pixi run -e recording recording-app -- --zarr-path /path/to/recording_analysis.zarr

Starts the read-only Palette recording Marimo application. A direct launch
shows only the selected Zarr. Add --recordings-root or --registry explicitly
to enable collection browsing.

Application arguments after `--` are passed to the notebook. The most useful
ones are:

  --zarr-path PATH             Required analysis Zarr directory.
  --recordings-root PATH       Optional sibling-recording collection root.
  --registry PATH              Optional read-only Palette registry.
  --recording-name-contains S  Optional collection filter.
  --renderer ID                Optional initial renderer filter.
  --run-path PATH              Optional initial run-path filter.
  --artifact NAME              Optional initial artifact filter.

Server configuration is provided through environment variables:

  PALETTE_RECORDING_APP_HOST    Bind host (local default: 127.0.0.1).
  PALETTE_RECORDING_APP_PORT    Bind port (local default: 2720).
  PALETTE_RECORDING_APP_TOKEN   Optional Marimo access token.

When FileGlancer provides FG_SERVICE_PORT and FG_SERVICE_TOKEN, this launcher
binds to 0.0.0.0 and enforces the FileGlancer token automatically.
EOF
}

if [[ "${1:-}" == "--" ]]; then
  shift
fi

if [[ "${1:-}" == "--launcher-help" ]]; then
  usage
  exit 0
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
APP_PATH="${REPO_ROOT}/apps/marimo/palette_explorer.py"

if [[ ! -f "$APP_PATH" ]]; then
  echo "Palette recording app not found: $APP_PATH" >&2
  exit 2
fi
if ! command -v marimo >/dev/null 2>&1; then
  echo "marimo is not available; run through 'pixi run -e recording recording-app'." >&2
  exit 2
fi

if [[ -n "${FG_SERVICE_PORT:-}" ]]; then
  DEFAULT_HOST="0.0.0.0"
else
  DEFAULT_HOST="127.0.0.1"
fi
HOST="${PALETTE_RECORDING_APP_HOST:-$DEFAULT_HOST}"
PORT="${FG_SERVICE_PORT:-${PALETTE_RECORDING_APP_PORT:-2720}}"
TOKEN="${FG_SERVICE_TOKEN:-${PALETTE_RECORDING_APP_TOKEN:-}}"

if [[ ! "$PORT" =~ ^[0-9]+$ ]] || (( PORT < 1 || PORT > 65535 )); then
  echo "Palette recording app port must be an integer from 1 through 65535: $PORT" >&2
  exit 2
fi

CACHE_ROOT="${TMPDIR:-/tmp}/palette-recording-app-${UID:-user}"
mkdir -p "$CACHE_ROOT/matplotlib" "$CACHE_ROOT/pycache" "$CACHE_ROOT/xdg-cache"

export MPLCONFIGDIR="${MPLCONFIGDIR:-$CACHE_ROOT/matplotlib}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-$CACHE_ROOT/pycache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$CACHE_ROOT/xdg-cache}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

TOKEN_ARGS=(--no-token)
if [[ -n "$TOKEN" ]]; then
  TOKEN_ARGS=(--token-password "$TOKEN")
fi

cd "$REPO_ROOT"
exec marimo run \
  --headless \
  --host "$HOST" \
  --port "$PORT" \
  "${TOKEN_ARGS[@]}" \
  "$APP_PATH" -- \
  "$@"
