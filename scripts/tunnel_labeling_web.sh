#!/usr/bin/env bash
set -euo pipefail

LOCAL_PORT="${PALETTE_LABELING_LOCAL_PORT:-8795}"
REMOTE_HOST="${PALETTE_LABELING_REMOTE_HOST:-}"
REMOTE_PORT="${PALETTE_LABELING_REMOTE_PORT:-8795}"
REMOTE_BIND_HOST="${PALETTE_LABELING_REMOTE_BIND_HOST:-127.0.0.1}"
SSH_EXTRA_ARGS="${PALETTE_LABELING_SSH_EXTRA_ARGS:-}"

if [[ -z "${REMOTE_HOST}" ]]; then
  echo "Set PALETTE_LABELING_REMOTE_HOST to the workstation SSH host." >&2
  echo "Example:" >&2
  echo "  PALETTE_LABELING_REMOTE_HOST=<workstation-hostname> scripts/tunnel_labeling_web.sh" >&2
  exit 2
fi

echo "Opening Palette labeling SSH tunnel."
echo "local_url=http://127.0.0.1:${LOCAL_PORT}/admin/datasets"
echo "remote=${REMOTE_BIND_HOST}:${REMOTE_PORT}"
echo "ssh_host=${REMOTE_HOST}"
echo "Press Ctrl-C to close the tunnel."

# shellcheck disable=SC2086
exec ssh ${SSH_EXTRA_ARGS} -N -L "${LOCAL_PORT}:${REMOTE_BIND_HOST}:${REMOTE_PORT}" "${REMOTE_HOST}"
