#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/review_zarrs_with_crimson.sh --list FILE [options]

Required:
  --list FILE              Text file with one .zarr path per line.

Options:
  --crimson-repo DIR       Crimson repo root (default: $HOME/gitrepos/crimson).
  --bin RELPATH            Binary path relative to repo (default: release/redgui).
  --start-at N             1-based index to start from (default: 1).
  --dry-run                Print commands without launching Crimson.
  --no-prompt              Do not pause between recordings.
  -h, --help               Show this help text.

Example:
  scripts/review_zarrs_with_crimson.sh \
    --list /tmp/analysis_without_approval.txt \
    --crimson-repo "$HOME/gitrepos/crimson"
EOF
}

LIST_FILE=""
CRIMSON_REPO="${HOME}/gitrepos/crimson"
CRIMSON_BIN_REL="release/redgui"
START_AT=1
DRY_RUN=0
NO_PROMPT=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --list)
      LIST_FILE="${2:-}"
      shift 2
      ;;
    --crimson-repo)
      CRIMSON_REPO="${2:-}"
      shift 2
      ;;
    --bin)
      CRIMSON_BIN_REL="${2:-}"
      shift 2
      ;;
    --start-at)
      START_AT="${2:-1}"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --no-prompt)
      NO_PROMPT=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "${LIST_FILE}" ]]; then
  echo "Missing required --list FILE" >&2
  usage
  exit 2
fi

if [[ ! -f "${LIST_FILE}" ]]; then
  echo "List file not found: ${LIST_FILE}" >&2
  exit 1
fi

if ! [[ "${START_AT}" =~ ^[0-9]+$ ]] || [[ "${START_AT}" -lt 1 ]]; then
  echo "--start-at must be a positive integer (got: ${START_AT})" >&2
  exit 2
fi

CRIMSON_BIN="${CRIMSON_REPO}/${CRIMSON_BIN_REL}"
if [[ "${DRY_RUN}" -eq 0 ]] && [[ ! -x "${CRIMSON_BIN}" ]]; then
  echo "Crimson binary not executable: ${CRIMSON_BIN}" >&2
  exit 1
fi

mapfile -t RAW_LINES < "${LIST_FILE}"
ZARRS=()
for line in "${RAW_LINES[@]}"; do
  trimmed="${line#"${line%%[![:space:]]*}"}"
  trimmed="${trimmed%"${trimmed##*[![:space:]]}"}"
  if [[ -z "${trimmed}" ]] || [[ "${trimmed}" == \#* ]]; then
    continue
  fi
  ZARRS+=("${trimmed}")
done

TOTAL="${#ZARRS[@]}"
if [[ "${TOTAL}" -eq 0 ]]; then
  echo "No paths found in ${LIST_FILE}" >&2
  exit 1
fi

if [[ "${START_AT}" -gt "${TOTAL}" ]]; then
  echo "--start-at ${START_AT} is greater than number of entries (${TOTAL})" >&2
  exit 1
fi

echo "Loaded ${TOTAL} zarr path(s) from ${LIST_FILE}"
echo "Crimson repo: ${CRIMSON_REPO}"
echo "Binary: ${CRIMSON_BIN_REL}"
if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "Mode: dry-run"
fi
echo

for ((i=START_AT; i<=TOTAL; i++)); do
  zarr_path="${ZARRS[$((i-1))]}"
  echo "[${i}/${TOTAL}] ${zarr_path}"

  if [[ ! -d "${zarr_path}" ]]; then
    echo "  missing directory, skipping"
    echo
    continue
  fi

  cmd=( "${CRIMSON_BIN}" --zarr "${zarr_path}" )
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    printf '  cmd: (cd %q && %q --zarr %q)\n' "${CRIMSON_REPO}" "${CRIMSON_BIN}" "${zarr_path}"
  else
    (
      cd "${CRIMSON_REPO}"
      "${cmd[@]}"
    )
  fi

  if [[ "${NO_PROMPT}" -eq 0 ]]; then
    read -r -p "Press Enter for next, 'q' to quit: " answer
    if [[ "${answer}" == "q" ]] || [[ "${answer}" == "Q" ]]; then
      next_index=$((i + 1))
      if [[ "${next_index}" -le "${TOTAL}" ]]; then
        echo "Stopped at ${i}. Resume with: --start-at ${next_index}"
      fi
      exit 0
    fi
  fi

  echo
done

echo "Done. Reviewed ${TOTAL} entries."
