#!/usr/bin/env bash
set -euo pipefail

PORT="${PALETTE_LABELING_PORT:-8795}"
PID_PATH="${PALETTE_LABELING_PID:-/tmp/palette-labeling-web-${PORT}.pid}"

stop_pid() {
  local pid="$1"
  if ! kill -0 "${pid}" 2>/dev/null; then
    return 0
  fi

  kill "${pid}" 2>/dev/null || true
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    if ! kill -0 "${pid}" 2>/dev/null; then
      return 0
    fi
    sleep 0.2
  done

  echo "Process ${pid} did not exit after SIGTERM." >&2
  echo "Re-run with manual operator cleanup if needed." >&2
  return 1
}

stopped=0

if [[ -f "${PID_PATH}" ]]; then
  pid="$(cat "${PID_PATH}" 2>/dev/null || true)"
  if [[ -n "${pid}" ]]; then
    if stop_pid "${pid}"; then
      echo "Stopped Palette labeling web server from pid file."
      echo "pid=${pid}"
      stopped=1
    fi
  fi
  rm -f "${PID_PATH}"
fi

matching_pids="$(pgrep -u "$(id -u)" -f "fisheye.labeling.web .*--port ${PORT}" || true)"
for pid in ${matching_pids}; do
  if stop_pid "${pid}"; then
    echo "Stopped Palette labeling web server by process match."
    echo "pid=${pid}"
    stopped=1
  fi
done

if [[ "${stopped}" == "0" ]]; then
  echo "No Palette labeling web server found for port ${PORT}."
fi
