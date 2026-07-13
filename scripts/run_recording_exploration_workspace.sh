#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  pixi run -e recording recording-workspace -- \
    --zarr-path /path/to/recording_analysis.zarr

Starts an editable copy of the Palette recording explorer. The rendered app
appears above a visible exploration cell, and Marimo Pair can attach to the
live editor session.

The editor runs in a Bubblewrap mount namespace. Only these host paths are
visible:

  - the Palette Pixi checkout, mounted read-only;
  - the selected Zarr, mounted read-only at /data/recording.zarr;
  - a per-session notebook directory, mounted read/write at /workspace; and
  - minimal operating-system runtime paths.

Supported notebook arguments:

  --zarr-path PATH    Required analysis Zarr directory.
  --renderer ID       Optional initial renderer filter.
  --run-path PATH     Optional initial run-path filter.
  --artifact NAME     Optional initial artifact filter.

Collection browsing is deliberately unavailable in the first editable canary:
each session receives exactly one read-only recording mount.

Server configuration is provided through environment variables:

  PALETTE_RECORDING_WORKSPACE_HOST    Bind host (local default: 127.0.0.1).
  PALETTE_RECORDING_WORKSPACE_PORT    Bind port (local default: 2721).
  PALETTE_RECORDING_WORKSPACE_TOKEN   Optional Marimo access token.
  PALETTE_RECORDING_WORKSPACE_ROOT    Host directory for writable notebook copies.
  PALETTE_RECORDING_WORKSPACE_PYTHON  Pixi Python executable override for tests.

When FileGlancer provides FG_SERVICE_PORT and FG_SERVICE_TOKEN, this launcher
binds to 0.0.0.0 and enforces the FileGlancer token automatically. Pair clients
use that same token through their MARIMO_TOKEN environment variable.
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
if ! command -v bwrap >/dev/null 2>&1; then
  echo "Bubblewrap (bwrap) is required for the read-only exploration workspace." >&2
  exit 2
fi

ZARR_PATH=""
NOTEBOOK_ARGS=()
while (( $# > 0 )); do
  case "$1" in
    --zarr-path)
      if (( $# < 2 )); then
        echo "--zarr-path requires a directory." >&2
        exit 2
      fi
      ZARR_PATH="$2"
      shift 2
      ;;
    --renderer|--run-path|--artifact)
      if (( $# < 2 )); then
        echo "$1 requires a value." >&2
        exit 2
      fi
      NOTEBOOK_ARGS+=("$1" "$2")
      shift 2
      ;;
    --recordings-root|--registry|--recording-name-contains)
      echo "$1 is unavailable in the single-recording read-only workspace." >&2
      exit 2
      ;;
    *)
      echo "Unsupported recording workspace argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "$ZARR_PATH" ]]; then
  echo "Required --zarr-path is missing." >&2
  exit 2
fi
if [[ ! -d "$ZARR_PATH" ]]; then
  echo "Recording Zarr directory was not found: $ZARR_PATH" >&2
  exit 2
fi
ZARR_PATH="$(realpath -e -- "$ZARR_PATH")"

if [[ -n "${FG_SERVICE_PORT:-}" ]]; then
  DEFAULT_HOST="0.0.0.0"
else
  DEFAULT_HOST="127.0.0.1"
fi
HOST="${PALETTE_RECORDING_WORKSPACE_HOST:-$DEFAULT_HOST}"
PORT="${FG_SERVICE_PORT:-${PALETTE_RECORDING_WORKSPACE_PORT:-2721}}"
TOKEN="${FG_SERVICE_TOKEN:-${PALETTE_RECORDING_WORKSPACE_TOKEN:-}}"

if [[ ! "$PORT" =~ ^[0-9]+$ ]] || (( PORT < 1 || PORT > 65535 )); then
  echo "Palette recording workspace port must be an integer from 1 through 65535: $PORT" >&2
  exit 2
fi

PYTHON_EXECUTABLE="${PALETTE_RECORDING_WORKSPACE_PYTHON:-$(command -v python || true)}"
if [[ -z "$PYTHON_EXECUTABLE" || ! -x "$PYTHON_EXECUTABLE" ]]; then
  echo "A Python executable from the Pixi recording environment is required." >&2
  exit 2
fi
PYTHON_EXECUTABLE="$(realpath -e -- "$PYTHON_EXECUTABLE")"
PYTHON_PREFIX="$(cd -- "$(dirname -- "$PYTHON_EXECUTABLE")/.." && pwd)"

WORKSPACE_ROOT="${PALETTE_RECORDING_WORKSPACE_ROOT:-${HOME}/.palette/marimo-workspaces}"
mkdir -p -- "$WORKSPACE_ROOT"
WORKSPACE_ROOT="$(realpath -e -- "$WORKSPACE_ROOT")"
SESSION_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RECORDING_NAME="$(basename -- "$ZARR_PATH" | tr -c 'A-Za-z0-9._-' '_')"
SESSION_ROOT="${WORKSPACE_ROOT}/${SESSION_STAMP}-${RECORDING_NAME}-${$}"
mkdir -p -- \
  "$SESSION_ROOT/cache/matplotlib" \
  "$SESSION_ROOT/cache/pycache" \
  "$SESSION_ROOT/cache/xdg" \
  "$SESSION_ROOT/home/.config" \
  "$SESSION_ROOT/tmp"
chmod 700 "$SESSION_ROOT"

case "$SESSION_ROOT" in
  "$ZARR_PATH"|"$ZARR_PATH"/*)
    echo "Writable workspace must not be inside the selected Zarr: $SESSION_ROOT" >&2
    exit 2
    ;;
esac
case "$ZARR_PATH" in
  "$SESSION_ROOT"|"$SESSION_ROOT"/*)
    echo "Selected Zarr must not be inside the writable session: $ZARR_PATH" >&2
    exit 2
    ;;
esac

WORKSPACE_NOTEBOOK="$SESSION_ROOT/palette_recording_workspace.py"
cp -- "$APP_PATH" "$WORKSPACE_NOTEBOOK"
chmod u+rw "$WORKSPACE_NOTEBOOK"

if [[ -n "$TOKEN" ]]; then
  TOKEN_ARGS=(--token-password "$TOKEN")
else
  TOKEN_ARGS=(--no-token)
fi

# The Pixi environment contains absolute-prefix metadata, so bind the checkout
# at its original path while exposing only empty ancestors leading to it.
REPO_PARENT_ARGS=()
REPO_PARENT="$(dirname -- "$REPO_ROOT")"
while [[ "$REPO_PARENT" != "/" ]]; do
  REPO_PARENT_ARGS=(--dir "$REPO_PARENT" "${REPO_PARENT_ARGS[@]}")
  REPO_PARENT="$(dirname -- "$REPO_PARENT")"
done

PYTHON_BIND_ARGS=()
case "$PYTHON_PREFIX" in
  "$REPO_ROOT"|"$REPO_ROOT"/*) ;;
  *)
    PYTHON_PARENT_ARGS=()
    PYTHON_PARENT="$(dirname -- "$PYTHON_PREFIX")"
    while [[ "$PYTHON_PARENT" != "/" ]]; do
      PYTHON_PARENT_ARGS=(--dir "$PYTHON_PARENT" "${PYTHON_PARENT_ARGS[@]}")
      PYTHON_PARENT="$(dirname -- "$PYTHON_PARENT")"
    done
    PYTHON_BIND_ARGS=("${PYTHON_PARENT_ARGS[@]}" --ro-bind "$PYTHON_PREFIX" "$PYTHON_PREFIX")
    ;;
esac

echo "Palette editable notebook (host): $WORKSPACE_NOTEBOOK" >&2
echo "Palette sandbox dataset (read-only): /data/recording.zarr" >&2
echo "Palette sandbox workspace (writable): /workspace" >&2

exec bwrap \
  --die-with-parent \
  --new-session \
  --unshare-pid \
  --unshare-ipc \
  --unshare-uts \
  --cap-drop ALL \
  --clearenv \
  --ro-bind /usr /usr \
  --symlink usr/bin /bin \
  --symlink usr/sbin /sbin \
  --symlink usr/lib /lib \
  --symlink usr/lib64 /lib64 \
  --ro-bind /etc /etc \
  --ro-bind /sys /sys \
  --proc /proc \
  --dev /dev \
  --tmpfs /run \
  --bind "$SESSION_ROOT/tmp" /tmp \
  "${REPO_PARENT_ARGS[@]}" \
  --ro-bind "$REPO_ROOT" "$REPO_ROOT" \
  "${PYTHON_BIND_ARGS[@]}" \
  --dir /data \
  --ro-bind "$ZARR_PATH" /data/recording.zarr \
  --bind "$SESSION_ROOT" /workspace \
  --setenv HOME /workspace/home \
  --setenv USER "${USER:-palette}" \
  --setenv LOGNAME "${LOGNAME:-${USER:-palette}}" \
  --setenv PATH "$(dirname -- "$PYTHON_EXECUTABLE"):/usr/bin:/bin" \
  --setenv PYTHONPATH "${REPO_ROOT}/src:${REPO_ROOT}" \
  --setenv PYTHONPYCACHEPREFIX /workspace/cache/pycache \
  --setenv XDG_CACHE_HOME /workspace/cache/xdg \
  --setenv XDG_CONFIG_HOME /workspace/home/.config \
  --setenv MPLCONFIGDIR /workspace/cache/matplotlib \
  --setenv MPLBACKEND Agg \
  --setenv TMPDIR /tmp \
  --setenv LANG C.UTF-8 \
  --setenv LC_ALL C.UTF-8 \
  --chdir /workspace \
  "$PYTHON_EXECUTABLE" -m marimo edit \
  --headless \
  --skip-update-check \
  --host "$HOST" \
  --port "$PORT" \
  "${TOKEN_ARGS[@]}" \
  /workspace/palette_recording_workspace.py -- \
  --zarr-path /data/recording.zarr \
  --workspace true \
  "${NOTEBOOK_ARGS[@]}"
