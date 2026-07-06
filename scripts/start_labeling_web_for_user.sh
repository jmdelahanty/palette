#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat >&2 <<'EOF'
Usage:
  scripts/start_labeling_web_for_user.sh USER PORT [LOCAL_TUNNEL_PORT]

Starts one fixed-user Palette labeling web server on a dedicated port.
This is intended as a short-term multi-labeler helper when there is no trusted auth proxy yet.

Examples:
  scripts/start_labeling_web_for_user.sh alice 8791
  scripts/start_labeling_web_for_user.sh bob 8792 8872

Useful environment overrides:
  PALETTE_LABELING_STORE=/path/to/labeling.sqlite
  PALETTE_REGISTRY_PATH=/path/to/palette_registry.sqlite
  PALETTE_LABELING_ADMIN_USER=delahantyj
  PALETTE_LABELING_HOST=127.0.0.1
  PALETTE_LABELING_REMOTE_HOST=delahantyj-ws1
EOF
}

USER_ID="${1:-}"
PORT="${2:-}"
LOCAL_TUNNEL_PORT="${3:-${PALETTE_LABELING_LOCAL_PORT:-${PORT}}}"

if [[ -z "${USER_ID}" || -z "${PORT}" ]]; then
  usage
  exit 2
fi

if [[ ! "${PORT}" =~ ^[0-9]+$ ]]; then
  echo "PORT must be numeric: ${PORT}" >&2
  exit 2
fi

if [[ -n "${LOCAL_TUNNEL_PORT}" && ! "${LOCAL_TUNNEL_PORT}" =~ ^[0-9]+$ ]]; then
  echo "LOCAL_TUNNEL_PORT must be numeric: ${LOCAL_TUNNEL_PORT}" >&2
  exit 2
fi

HOST="${PALETTE_LABELING_HOST:-127.0.0.1}"
ADMIN_USER="${PALETTE_LABELING_ADMIN_USER:-delahantyj}"
LOG_PATH="${PALETTE_LABELING_LOG:-/tmp/palette-labeling-web-${USER_ID}-${PORT}.log}"
PID_PATH="${PALETTE_LABELING_PID:-/tmp/palette-labeling-web-${USER_ID}-${PORT}.pid}"
REMOTE_HOST="${PALETTE_LABELING_REMOTE_HOST:-<workstation-hostname>}"
STORE="${PALETTE_LABELING_STORE:-${HOME}/.palette/labeling_work.sqlite}"

PALETTE_LABELING_AUTH_MODE=fixed \
PALETTE_LABELING_USER="${USER_ID}" \
PALETTE_LABELING_ADMIN_USER="${ADMIN_USER}" \
PALETTE_LABELING_HOST="${HOST}" \
PALETTE_LABELING_PORT="${PORT}" \
PALETTE_LABELING_STORE="${STORE}" \
PALETTE_LABELING_LOG="${LOG_PATH}" \
PALETTE_LABELING_PID="${PID_PATH}" \
  "${SCRIPT_DIR}/start_labeling_web.sh"

cat <<EOF

Per-user fixed-mode server summary:
user=${USER_ID}
port=${PORT}
host=${HOST}
store=${STORE}
admin_user=${ADMIN_USER}
labeler_url=http://${HOST}:${PORT}/my-datasets
admin_url=http://${HOST}:${PORT}/admin/datasets
stop_command=PALETTE_LABELING_PORT=${PORT} PALETTE_LABELING_PID=${PID_PATH} scripts/stop_labeling_web.sh

Tunnel command for this user/port from a remote machine:
PALETTE_LABELING_REMOTE_HOST=${REMOTE_HOST} \\
PALETTE_LABELING_REMOTE_PORT=${PORT} \\
PALETTE_LABELING_LOCAL_PORT=${LOCAL_TUNNEL_PORT} \\
scripts/tunnel_labeling_web.sh

Remote browser URL after tunnel:
http://127.0.0.1:${LOCAL_TUNNEL_PORT}/my-datasets

Operational constraint: only assign distinct recordings to each fixed-user server.
EOF
