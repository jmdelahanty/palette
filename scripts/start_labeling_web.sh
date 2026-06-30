#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

HOST="${PALETTE_LABELING_HOST:-127.0.0.1}"
PORT="${PALETTE_LABELING_PORT:-8795}"
STORE="${PALETTE_LABELING_STORE:-${HOME}/.palette/labeling_work.sqlite}"
AUTH_MODE="${PALETTE_LABELING_AUTH_MODE:-fixed}"
AUTH_HEADER="${PALETTE_LABELING_AUTH_HEADER:-X-Forwarded-User}"
LABELING_USER="${PALETTE_LABELING_USER:-delahantyj}"
ADMIN_USER="${PALETTE_LABELING_ADMIN_USER:-${LABELING_USER}}"
REGISTRY_PATH="${PALETTE_REGISTRY_PATH:-/groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite}"
LOG_PATH="${PALETTE_LABELING_LOG:-/tmp/palette-labeling-web-${PORT}.log}"
PID_PATH="${PALETTE_LABELING_PID:-/tmp/palette-labeling-web-${PORT}.pid}"
START_WAIT_SECONDS="${PALETTE_LABELING_START_WAIT_SECONDS:-2}"

case "${AUTH_MODE}" in
  fixed)
    AUTH_ARGS=(--user "${LABELING_USER}")
    ;;
  header)
    AUTH_ARGS=(--trust-auth-header --auth-header "${AUTH_HEADER}")
    ;;
  *)
    echo "Unsupported PALETTE_LABELING_AUTH_MODE=${AUTH_MODE}" >&2
    echo "Use PALETTE_LABELING_AUTH_MODE=fixed or PALETTE_LABELING_AUTH_MODE=header." >&2
    exit 2
    ;;
esac

if [[ -f "${PID_PATH}" ]]; then
  existing_pid="$(cat "${PID_PATH}" 2>/dev/null || true)"
  if [[ -n "${existing_pid}" ]] && kill -0 "${existing_pid}" 2>/dev/null; then
    echo "Palette labeling web server is already running."
    echo "pid=${existing_pid}"
    echo "url=http://${HOST}:${PORT}/admin/datasets"
    echo "log=${LOG_PATH}"
    exit 0
  fi
  rm -f "${PID_PATH}"
fi

mkdir -p "$(dirname "${STORE}")"

cd "${REPO_ROOT}"

setsid env PALETTE_REGISTRY_PATH="${REGISTRY_PATH}" \
  "${REPO_ROOT}/scripts/py" -m fisheye.labeling.web \
    --store "${STORE}" \
    serve \
    --host "${HOST}" \
    --port "${PORT}" \
    "${AUTH_ARGS[@]}" \
    --admin-user "${ADMIN_USER}" \
  >"${LOG_PATH}" 2>&1 < /dev/null &

server_pid="$!"
echo "${server_pid}" > "${PID_PATH}"
sleep "${START_WAIT_SECONDS}"

if kill -0 "${server_pid}" 2>/dev/null; then
  echo "Started Palette labeling web server."
  echo "pid=${server_pid}"
  echo "host=${HOST}"
  echo "port=${PORT}"
  echo "store=${STORE}"
  echo "registry=${REGISTRY_PATH}"
  echo "auth_mode=${AUTH_MODE}"
  if [[ "${AUTH_MODE}" == "fixed" ]]; then
    echo "fixed_user=${LABELING_USER}"
  else
    echo "auth_header=${AUTH_HEADER}"
  fi
  echo "admin_user=${ADMIN_USER}"
  echo "log=${LOG_PATH}"
  echo "pid_file=${PID_PATH}"
  echo "admin_url=http://${HOST}:${PORT}/admin/datasets"
  echo "labeler_url=http://${HOST}:${PORT}/my-datasets"
  exit 0
fi

echo "Palette labeling web server failed to stay running." >&2
echo "log=${LOG_PATH}" >&2
tail -n 40 "${LOG_PATH}" >&2 || true
rm -f "${PID_PATH}"
exit 1
