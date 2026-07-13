#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  pixi run app
  pixi run app -- --export-root /path/to/palette_analytics

Starts the read-only Palette group analytics Marimo application.

Application arguments after `--` are passed to the notebook. The most useful
ones are:

  --export-root PATH       Authorized analytics export root.
  --export-run-id ID       Initially selected export, or "latest" (default).
  --stats-run-id ID        Statistics run, or "auto" (default).
  --panel ID               Initially selected visualization panel.

Server configuration is provided through environment variables:

  PALETTE_ANALYTICS_APP_HOST    Bind host (local default: 127.0.0.1).
  PALETTE_ANALYTICS_APP_PORT    Bind port (local default: 2718).
  PALETTE_ANALYTICS_APP_TOKEN   Optional Marimo access token.

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
APP_PATH="${REPO_ROOT}/apps/marimo/group_analytics_explorer.py"

if [[ ! -f "$APP_PATH" ]]; then
  echo "Palette group analytics app not found: $APP_PATH" >&2
  exit 2
fi
if ! command -v marimo >/dev/null 2>&1; then
  echo "marimo is not available; run this launcher through 'pixi run app'." >&2
  exit 2
fi

if [[ -n "${FG_SERVICE_PORT:-}" ]]; then
  DEFAULT_HOST="0.0.0.0"
else
  DEFAULT_HOST="127.0.0.1"
fi
HOST="${PALETTE_ANALYTICS_APP_HOST:-$DEFAULT_HOST}"
PORT="${FG_SERVICE_PORT:-${PALETTE_ANALYTICS_APP_PORT:-2718}}"
TOKEN="${FG_SERVICE_TOKEN:-${PALETTE_ANALYTICS_APP_TOKEN:-}}"

if [[ ! "$PORT" =~ ^[0-9]+$ ]] || (( PORT < 1 || PORT > 65535 )); then
  echo "Palette analytics app port must be an integer from 1 through 65535: $PORT" >&2
  exit 2
fi

CACHE_ROOT="${TMPDIR:-/tmp}/palette-analytics-app-${UID:-user}"
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
