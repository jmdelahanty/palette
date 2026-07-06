#!/usr/bin/env bash
set -euo pipefail

PID_GLOB="${PALETTE_LABELING_PID_GLOB:-/tmp/palette-labeling-web-*.pid}"
CURRENT_UID="$(id -u)"

print_header() {
  printf '%-14s %-7s %-8s %-9s %-42s %-42s %s\n' \
    "USER" "PORT" "PID" "STATUS" "LABELER_URL" "ADMIN_URL" "LOG"
}

value_after() {
  local needle="$1"
  shift
  local prev=""
  local arg
  for arg in "$@"; do
    if [[ "${prev}" == "${needle}" ]]; then
      printf '%s\n' "${arg}"
      return 0
    fi
    prev="${arg}"
  done
  return 1
}

infer_user_from_pid_path() {
  local path="$1"
  local base
  base="$(basename "${path}" .pid)"
  if [[ "${base}" =~ ^palette-labeling-web-(.+)-([0-9]+)$ ]]; then
    printf '%s\n' "${BASH_REMATCH[1]}"
    return 0
  fi
  printf '%s\n' ""
}

infer_port_from_pid_path() {
  local path="$1"
  local base
  base="$(basename "${path}" .pid)"
  if [[ "${base}" =~ ^palette-labeling-web-(.+)-([0-9]+)$ ]]; then
    printf '%s\n' "${BASH_REMATCH[2]}"
    return 0
  fi
  if [[ "${base}" =~ ^palette-labeling-web-([0-9]+)$ ]]; then
    printf '%s\n' "${BASH_REMATCH[1]}"
    return 0
  fi
  printf '%s\n' ""
}

row_for_pid_file() {
  local pid_path="$1"
  local pid status cmd user port host labeler_url admin_url log_path
  pid="$(cat "${pid_path}" 2>/dev/null || true)"
  if [[ -z "${pid}" || ! "${pid}" =~ ^[0-9]+$ ]]; then
    return 0
  fi

  if ! kill -0 "${pid}" 2>/dev/null; then
    status="stale"
    user="$(infer_user_from_pid_path "${pid_path}")"
    port="$(infer_port_from_pid_path "${pid_path}")"
    host="127.0.0.1"
    labeler_url="${port:+http://${host}:${port}/my-datasets}"
    admin_url="${port:+http://${host}:${port}/admin/datasets}"
    log_path=""
    printf '%-14s %-7s %-8s %-9s %-42s %-42s %s\n' \
      "${user:-unknown}" "${port:-unknown}" "${pid}" "${status}" "${labeler_url:-unknown}" "${admin_url:-unknown}" "${log_path:-unknown}"
    return 0
  fi

  # Prefer argv split by NUL so paths containing spaces remain intact enough for flag scanning.
  mapfile -d '' argv < "/proc/${pid}/cmdline" 2>/dev/null || return 0
  cmd="${argv[*]:-}"
  if [[ "${cmd}" != *"fisheye.labeling.web"* ]]; then
    return 0
  fi

  status="running"
  user="$(value_after --user "${argv[@]}" 2>/dev/null || true)"
  port="$(value_after --port "${argv[@]}" 2>/dev/null || true)"
  host="$(value_after --host "${argv[@]}" 2>/dev/null || true)"
  user="${user:-$(infer_user_from_pid_path "${pid_path}")}"
  port="${port:-$(infer_port_from_pid_path "${pid_path}")}"
  host="${host:-127.0.0.1}"

  if [[ "${host}" == "0.0.0.0" || "${host}" == "::" ]]; then
    host="127.0.0.1"
  fi

  labeler_url="${port:+http://${host}:${port}/my-datasets}"
  admin_url="${port:+http://${host}:${port}/admin/datasets}"
  log_path="${pid_path%.pid}.log"
  if [[ ! -f "${log_path}" ]]; then
    log_path="unknown"
  fi

  printf '%-14s %-7s %-8s %-9s %-42s %-42s %s\n' \
    "${user:-unknown}" "${port:-unknown}" "${pid}" "${status}" "${labeler_url:-unknown}" "${admin_url:-unknown}" "${log_path}"
}

shopt -s nullglob
pid_files=( ${PID_GLOB} )

print_header
if [[ "${#pid_files[@]}" -eq 0 ]]; then
  exit 0
fi

for pid_path in "${pid_files[@]}"; do
  row_for_pid_file "${pid_path}"
done

# Also report live processes for this user that do not have matching pid files.
for pid in $(pgrep -u "${CURRENT_UID}" -f "fisheye.labeling.web" || true); do
  already_seen=0
  for pid_path in "${pid_files[@]}"; do
    if [[ "$(cat "${pid_path}" 2>/dev/null || true)" == "${pid}" ]]; then
      already_seen=1
      break
    fi
  done
  if [[ "${already_seen}" == "1" ]]; then
    continue
  fi
  mapfile -d '' argv < "/proc/${pid}/cmdline" 2>/dev/null || continue
  user="$(value_after --user "${argv[@]}" 2>/dev/null || true)"
  port="$(value_after --port "${argv[@]}" 2>/dev/null || true)"
  host="$(value_after --host "${argv[@]}" 2>/dev/null || true)"
  host="${host:-127.0.0.1}"
  if [[ "${host}" == "0.0.0.0" || "${host}" == "::" ]]; then
    host="127.0.0.1"
  fi
  printf '%-14s %-7s %-8s %-9s %-42s %-42s %s\n' \
    "${user:-unknown}" "${port:-unknown}" "${pid}" "running" \
    "${port:+http://${host}:${port}/my-datasets}" \
    "${port:+http://${host}:${port}/admin/datasets}" \
    "unknown"
done
