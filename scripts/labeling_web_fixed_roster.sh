#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROSTER_PATH="${PALETTE_LABELING_FIXED_ROSTER:-${HOME}/.palette/labeling_fixed_servers.tsv}"
REMOTE_HOST_DEFAULT="${PALETTE_LABELING_REMOTE_HOST:-}"

usage() {
  cat >&2 <<'USAGE'
Usage:
  scripts/labeling_web_fixed_roster.sh path
  scripts/labeling_web_fixed_roster.sh list
  scripts/labeling_web_fixed_roster.sh set USER PORT [LOCAL_PORT] [SSH_TARGET]
  scripts/labeling_web_fixed_roster.sh remove USER
  scripts/labeling_web_fixed_roster.sh start [USER|--all]
  scripts/labeling_web_fixed_roster.sh stop [USER|--all]
  scripts/labeling_web_fixed_roster.sh restart [USER|--all]
  scripts/labeling_web_fixed_roster.sh message [USER|--all]

Maintains a local fixed-user server roster, defaulting to:
  ~/.palette/labeling_fixed_servers.tsv

Roster format is tab-separated:
  user_id<TAB>server_port<TAB>local_tunnel_port<TAB>ssh_target

The roster is local operator state and should not be committed. It is intended for
short-term fixed-user multi-labeler operation before a trusted auth proxy exists.
USAGE
}

is_port() {
  [[ "${1:-}" =~ ^[0-9]+$ ]]
}

ensure_roster_dir() {
  mkdir -p "$(dirname "${ROSTER_PATH}")"
}

ensure_roster_exists() {
  ensure_roster_dir
  if [[ ! -f "${ROSTER_PATH}" ]]; then
    cat > "${ROSTER_PATH}" <<'HEADER'
# Palette fixed-user labeling server roster.
# Format: user_id<TAB>server_port<TAB>local_tunnel_port<TAB>ssh_target
HEADER
  fi
}

default_remote_host() {
  if [[ -n "${REMOTE_HOST_DEFAULT}" ]]; then
    printf '%s\n' "${REMOTE_HOST_DEFAULT}"
    return 0
  fi
  hostname -f 2>/dev/null || hostname
}

remote_host_from_ssh_target() {
  local ssh_target="$1"
  if [[ "${ssh_target}" == *@* ]]; then
    printf '%s\n' "${ssh_target#*@}"
    return 0
  fi
  default_remote_host
}

print_roster_header() {
  printf '%-14s %-7s %-7s %s\n' "USER" "PORT" "LOCAL" "SSH_TARGET"
}

read_roster_rows() {
  ensure_roster_exists
  local user port local_port ssh_target
  while IFS=$'\t' read -r user port local_port ssh_target _rest; do
    [[ -z "${user}" || "${user}" == \#* ]] && continue
    local_port="${local_port:-${port}}"
    ssh_target="${ssh_target:-}"
    printf '%s\t%s\t%s\t%s\n' "${user}" "${port}" "${local_port}" "${ssh_target}"
  done < "${ROSTER_PATH}"
}

row_for_user() {
  local wanted="$1"
  local user port local_port ssh_target
  while IFS=$'\t' read -r user port local_port ssh_target; do
    if [[ "${user}" == "${wanted}" ]]; then
      printf '%s\t%s\t%s\t%s\n' "${user}" "${port}" "${local_port}" "${ssh_target}"
      return 0
    fi
  done < <(read_roster_rows)
  return 1
}

selected_rows() {
  local selector="${1:---all}"
  if [[ "${selector}" == "--all" || -z "${selector}" ]]; then
    read_roster_rows
    return 0
  fi
  row_for_user "${selector}"
}

cmd_path() {
  printf '%s\n' "${ROSTER_PATH}"
}

cmd_list() {
  print_roster_header
  local user port local_port ssh_target
  while IFS=$'\t' read -r user port local_port ssh_target; do
    printf '%-14s %-7s %-7s %s\n' "${user}" "${port}" "${local_port}" "${ssh_target:-}"
  done < <(read_roster_rows)
}

cmd_set() {
  local user="${1:-}"
  local port="${2:-}"
  local local_port="${3:-${port}}"
  local ssh_target="${4:-}"
  if [[ -z "${user}" || -z "${port}" ]]; then
    usage
    exit 2
  fi
  if ! is_port "${port}"; then
    echo "PORT must be numeric: ${port}" >&2
    exit 2
  fi
  if ! is_port "${local_port}"; then
    echo "LOCAL_PORT must be numeric: ${local_port}" >&2
    exit 2
  fi

  ensure_roster_exists
  local tmp
  tmp="$(mktemp "${ROSTER_PATH}.tmp.XXXXXX")"
  local found=0 line_user
  while IFS= read -r line || [[ -n "${line}" ]]; do
    if [[ -z "${line}" || "${line}" == \#* ]]; then
      printf '%s\n' "${line}" >> "${tmp}"
      continue
    fi
    line_user="${line%%$'\t'*}"
    if [[ "${line_user}" == "${user}" ]]; then
      printf '%s\t%s\t%s\t%s\n' "${user}" "${port}" "${local_port}" "${ssh_target}" >> "${tmp}"
      found=1
    else
      printf '%s\n' "${line}" >> "${tmp}"
    fi
  done < "${ROSTER_PATH}"
  if [[ "${found}" == "0" ]]; then
    printf '%s\t%s\t%s\t%s\n' "${user}" "${port}" "${local_port}" "${ssh_target}" >> "${tmp}"
  fi
  mv "${tmp}" "${ROSTER_PATH}"
  echo "Set fixed labeling server port."
  echo "user=${user}"
  echo "port=${port}"
  echo "local_port=${local_port}"
  echo "ssh_target=${ssh_target}"
  echo "roster=${ROSTER_PATH}"
}

cmd_remove() {
  local user="${1:-}"
  if [[ -z "${user}" ]]; then
    usage
    exit 2
  fi
  ensure_roster_exists
  local tmp
  tmp="$(mktemp "${ROSTER_PATH}.tmp.XXXXXX")"
  local removed=0 line_user line
  while IFS= read -r line || [[ -n "${line}" ]]; do
    if [[ -z "${line}" || "${line}" == \#* ]]; then
      printf '%s\n' "${line}" >> "${tmp}"
      continue
    fi
    line_user="${line%%$'\t'*}"
    if [[ "${line_user}" == "${user}" ]]; then
      removed=1
      continue
    fi
    printf '%s\n' "${line}" >> "${tmp}"
  done < "${ROSTER_PATH}"
  mv "${tmp}" "${ROSTER_PATH}"
  if [[ "${removed}" == "1" ]]; then
    echo "Removed ${user} from ${ROSTER_PATH}."
  else
    echo "No roster row found for ${user}."
  fi
}

start_row() {
  local user="$1" port="$2" local_port="$3" ssh_target="$4"
  local remote_host
  remote_host="$(remote_host_from_ssh_target "${ssh_target}")"
  echo "Starting fixed-user server: user=${user} port=${port} local_port=${local_port}"
  if [[ -n "${ssh_target}" ]]; then
    PALETTE_LABELING_REMOTE_HOST="${remote_host}" \
    PALETTE_LABELING_SSH_TARGET="${ssh_target}" \
      "${SCRIPT_DIR}/start_labeling_web_for_user.sh" "${user}" "${port}" "${local_port}"
  else
    PALETTE_LABELING_REMOTE_HOST="${remote_host}" \
      "${SCRIPT_DIR}/start_labeling_web_for_user.sh" "${user}" "${port}" "${local_port}"
  fi
}

stop_row() {
  local user="$1" port="$2"
  echo "Stopping fixed-user server: user=${user} port=${port}"
  PALETTE_LABELING_PORT="${port}" \
  PALETTE_LABELING_PID="/tmp/palette-labeling-web-${user}-${port}.pid" \
    "${SCRIPT_DIR}/stop_labeling_web.sh"
}

message_row() {
  local user="$1" port="$2" local_port="$3" ssh_target="$4"
  local remote_host
  remote_host="$(remote_host_from_ssh_target "${ssh_target}")"
  if [[ -n "${ssh_target}" ]]; then
    PALETTE_LABELING_REMOTE_HOST="${remote_host}" \
    PALETTE_LABELING_SSH_TARGET="${ssh_target}" \
      "${SCRIPT_DIR}/labeling_web_tunnel_message.sh" "${user}" "${port}" "${local_port}"
  else
    PALETTE_LABELING_REMOTE_HOST="${remote_host}" \
      "${SCRIPT_DIR}/labeling_web_tunnel_message.sh" "${user}" "${port}" "${local_port}"
  fi
}

for_selected_rows() {
  local selector="$1" action="$2"
  local user port local_port ssh_target count=0
  while IFS=$'\t' read -r user port local_port ssh_target; do
    count=$((count + 1))
    case "${action}" in
      start) start_row "${user}" "${port}" "${local_port}" "${ssh_target}" ;;
      stop) stop_row "${user}" "${port}" ;;
      restart)
        stop_row "${user}" "${port}"
        start_row "${user}" "${port}" "${local_port}" "${ssh_target}"
        ;;
      message) message_row "${user}" "${port}" "${local_port}" "${ssh_target}" ;;
      *) echo "Unknown action: ${action}" >&2; exit 2 ;;
    esac
  done < <(selected_rows "${selector}")
  if [[ "${count}" == "0" ]]; then
    echo "No roster rows matched: ${selector}" >&2
    exit 1
  fi
}

command="${1:-}"
shift || true

case "${command}" in
  path) cmd_path "$@" ;;
  list) cmd_list "$@" ;;
  set) cmd_set "$@" ;;
  remove) cmd_remove "$@" ;;
  start) for_selected_rows "${1:---all}" start ;;
  stop) for_selected_rows "${1:---all}" stop ;;
  restart) for_selected_rows "${1:---all}" restart ;;
  message) for_selected_rows "${1:---all}" message ;;
  -h|--help|help|"") usage ;;
  *) echo "Unknown command: ${command}" >&2; usage; exit 2 ;;
esac
