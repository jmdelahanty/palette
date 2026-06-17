#!/usr/bin/env bash
# Small LSF helper functions for Palette submission scripts.

palette_lsf_extract_jobid() {
  local log_file="$1"
  sed -n 's/.*Job <\([0-9][0-9]*\)>.*/\1/p' "$log_file" | tail -n 1
}

palette_shell_print_command() {
  printf '%q ' "$@"
  printf '\n'
}

palette_lsf_submit_or_print() {
  local submit="$1"
  local log_file="$2"
  shift 2
  if [[ "$submit" == "1" ]]; then
    if ! command -v bsub >/dev/null 2>&1; then
      echo "bsub not found in PATH. Run this on an LSF login node." >&2
      exit 2
    fi
    "$@" 2>&1 | tee "$log_file"
  else
    palette_shell_print_command "$@" | tee "$log_file"
  fi
}
