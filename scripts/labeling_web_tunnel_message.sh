#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage:
  scripts/labeling_web_tunnel_message.sh USER REMOTE_PORT [LOCAL_PORT]

Prints copy/paste SSH tunnel instructions for a fixed-user Palette labeling server.
Run this on the workstation/operator machine. The labeler runs the printed ssh command
on their own laptop, then opens the printed local browser URL.

Examples:
  scripts/labeling_web_tunnel_message.sh dougand 8791
  scripts/labeling_web_tunnel_message.sh savinim 8792 8792

Useful environment overrides:
  PALETTE_LABELING_REMOTE_HOST=delahantyj-ws1
  PALETTE_LABELING_SSH_TARGET=campus_user@delahantyj-ws1
  PALETTE_LABELING_REMOTE_BIND_HOST=127.0.0.1
EOF
}

USER_ID="${1:-}"
REMOTE_PORT="${2:-}"
LOCAL_PORT="${3:-${REMOTE_PORT}}"

if [[ -z "${USER_ID}" || -z "${REMOTE_PORT}" ]]; then
  usage
  exit 2
fi

if [[ ! "${REMOTE_PORT}" =~ ^[0-9]+$ ]]; then
  echo "REMOTE_PORT must be numeric: ${REMOTE_PORT}" >&2
  exit 2
fi

if [[ -n "${LOCAL_PORT}" && ! "${LOCAL_PORT}" =~ ^[0-9]+$ ]]; then
  echo "LOCAL_PORT must be numeric: ${LOCAL_PORT}" >&2
  exit 2
fi

if [[ -n "${PALETTE_LABELING_REMOTE_HOST:-}" ]]; then
  REMOTE_HOST="${PALETTE_LABELING_REMOTE_HOST}"
else
  REMOTE_HOST="$(hostname -f 2>/dev/null || hostname)"
fi

REMOTE_BIND_HOST="${PALETTE_LABELING_REMOTE_BIND_HOST:-127.0.0.1}"
SSH_TARGET="${PALETTE_LABELING_SSH_TARGET:-${USER_ID}@${REMOTE_HOST}}"
LABELER_URL="http://127.0.0.1:${LOCAL_PORT}/my-datasets"

cat <<EOF
Palette labeling tunnel instructions for ${USER_ID}

1. Connect to the campus VPN if you are off campus.

2. Open Terminal or PowerShell on your laptop and run:

   ssh -N -L ${LOCAL_PORT}:${REMOTE_BIND_HOST}:${REMOTE_PORT} ${SSH_TARGET}

   Keep that window open while labeling. Press Ctrl-C in that window when done.

3. Open this URL in your browser:

   ${LABELER_URL}

If SSH says "Permission denied" or "Could not resolve hostname", send Jeremy the
full error text. Do not forward this URL to another person; it is a fixed-user
labeling server for ${USER_ID}.
EOF
