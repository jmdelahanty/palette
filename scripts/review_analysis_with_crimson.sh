#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/review_analysis_with_crimson.sh [options]

Purpose:
  1) Optionally refresh detect-quality rows in the registry
  2) Build a list of analysis zarr archives (registry-first)
  3) Launch Crimson (redgui) through each archive interactively

Options:
  --recordings-root DIR   Recordings root (default: /nvme1/recordings).
  --registry PATH         Registry sqlite path (default: /nvme1/palette_registry.sqlite).
  --crimson-repo DIR      Crimson repo root (default: $HOME/gitrepos/crimson).
  --mode MODE             all|unapproved (default: all).
  --list FILE             Output list path (default: /tmp/analysis_review_list.txt).
  --details FILE          Optional details TSV (used when --mode unapproved).
  --skip-refresh          Skip registry refresh step.
  --registry-only         Fail if registry list generation cannot be used.
  --start-at N            1-based index to start at when launching redgui.
  --dry-run               Build list and print commands without launching redgui.
  --no-prompt             Do not pause between recordings.
  -h, --help              Show this help.

Examples:
  # Review every analysis zarr, refreshing registry first
  scripts/review_analysis_with_crimson.sh

  # Review only unapproved analysis zarrs
  scripts/review_analysis_with_crimson.sh --mode unapproved

  # Resume from item 12
  scripts/review_analysis_with_crimson.sh --start-at 12
EOF
}

RECORDINGS_ROOT="/nvme1/recordings"
REGISTRY="/nvme1/palette_registry.sqlite"
CRIMSON_REPO="${HOME}/gitrepos/crimson"
MODE="all"
LIST_FILE="/tmp/analysis_review_list.txt"
DETAILS_FILE=""
SKIP_REFRESH=0
REGISTRY_ONLY=0
START_AT=1
DRY_RUN=0
NO_PROMPT=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --recordings-root)
      RECORDINGS_ROOT="${2:-}"
      shift 2
      ;;
    --registry)
      REGISTRY="${2:-}"
      shift 2
      ;;
    --crimson-repo)
      CRIMSON_REPO="${2:-}"
      shift 2
      ;;
    --mode)
      MODE="${2:-}"
      shift 2
      ;;
    --list)
      LIST_FILE="${2:-}"
      shift 2
      ;;
    --details)
      DETAILS_FILE="${2:-}"
      shift 2
      ;;
    --skip-refresh)
      SKIP_REFRESH=1
      shift
      ;;
    --registry-only)
      REGISTRY_ONLY=1
      shift
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

if [[ "${MODE}" != "all" && "${MODE}" != "unapproved" ]]; then
  echo "--mode must be one of: all, unapproved (got: ${MODE})" >&2
  exit 2
fi

if ! [[ "${START_AT}" =~ ^[0-9]+$ ]] || [[ "${START_AT}" -lt 1 ]]; then
  echo "--start-at must be a positive integer (got: ${START_AT})" >&2
  exit 2
fi

if [[ ! -d "${RECORDINGS_ROOT}" ]]; then
  echo "Recordings root not found: ${RECORDINGS_ROOT}" >&2
  exit 1
fi

if [[ "${REGISTRY_ONLY}" -eq 1 ]] && [[ ! -f "${REGISTRY}" ]]; then
  echo "Registry not found: ${REGISTRY}" >&2
  exit 1
fi

LIST_DIR="$(dirname "${LIST_FILE}")"
mkdir -p "${LIST_DIR}"

if [[ "${SKIP_REFRESH}" -eq 0 ]]; then
  if [[ -f "${REGISTRY}" ]]; then
    echo "Refreshing detect_quality registry rows..."
    scripts/py -m fisheye.registry.maintenance \
      --registry "${REGISTRY}" \
      --refresh-detect-quality
  elif [[ "${REGISTRY_ONLY}" -eq 1 ]]; then
    echo "Registry not found for refresh: ${REGISTRY}" >&2
    exit 1
  else
    echo "Registry not found, skipping refresh: ${REGISTRY}"
  fi
fi

if [[ "${MODE}" == "all" ]]; then
  BUILT=0
  if [[ -f "${REGISTRY}" ]]; then
    echo "Building analysis list from registry (all)..."
    if scripts/py -m fisheye.utils.registry_query \
      --registry "${REGISTRY}" \
      --zarr-use analysis \
      --path-contains "${RECORDINGS_ROOT}" \
      --output-file-list "${LIST_FILE}"; then
      BUILT=1
    elif [[ "${REGISTRY_ONLY}" -eq 1 ]]; then
      echo "Registry-only mode: failed to build list from registry." >&2
      exit 1
    else
      echo "Registry query failed; falling back to filesystem crawl."
    fi
  elif [[ "${REGISTRY_ONLY}" -eq 1 ]]; then
    echo "Registry-only mode requires --registry to exist: ${REGISTRY}" >&2
    exit 1
  fi

  if [[ "${BUILT}" -eq 0 ]]; then
    echo "Building analysis list from filesystem (all)..."
    find "${RECORDINGS_ROOT}" -type d -name '*_analysis.zarr' | sort > "${LIST_FILE}"
  fi

  COUNT="$(wc -l < "${LIST_FILE}")"
  echo "wrote: ${LIST_FILE}"
  echo "count: ${COUNT}"
else
  BUILT=0
  if [[ -f "${REGISTRY}" ]]; then
    echo "Building analysis list from registry (unapproved)..."
    if [[ -n "${DETAILS_FILE}" ]]; then
      if scripts/py -m fisheye.utils.list_unapproved_analysis_zarrs \
        --registry "${REGISTRY}" \
        --path-contains "${RECORDINGS_ROOT}" \
        --output "${LIST_FILE}" \
        --details "${DETAILS_FILE}"; then
        BUILT=1
      fi
    else
      if scripts/py -m fisheye.utils.list_unapproved_analysis_zarrs \
        --registry "${REGISTRY}" \
        --path-contains "${RECORDINGS_ROOT}" \
        --output "${LIST_FILE}"; then
        BUILT=1
      fi
    fi
    if [[ "${BUILT}" -eq 0 ]]; then
      if [[ "${REGISTRY_ONLY}" -eq 1 ]]; then
        echo "Registry-only mode: failed to build unapproved list from registry." >&2
        exit 1
      fi
      echo "Registry query failed; falling back to filesystem crawl."
    fi
  elif [[ "${REGISTRY_ONLY}" -eq 1 ]]; then
    echo "Registry-only mode requires --registry to exist: ${REGISTRY}" >&2
    exit 1
  fi

  if [[ "${BUILT}" -eq 0 ]]; then
    echo "Building analysis list from filesystem (unapproved)..."
    if [[ -n "${DETAILS_FILE}" ]]; then
      scripts/py -m fisheye.utils.list_unapproved_analysis_zarrs \
        "${RECORDINGS_ROOT}" \
        --recursive \
        --output "${LIST_FILE}" \
        --details "${DETAILS_FILE}"
    else
      scripts/py -m fisheye.utils.list_unapproved_analysis_zarrs \
        "${RECORDINGS_ROOT}" \
        --recursive \
        --output "${LIST_FILE}"
    fi
  fi
fi

CMD=(
  scripts/review_zarrs_with_crimson.sh
  --list "${LIST_FILE}"
  --crimson-repo "${CRIMSON_REPO}"
  --bin release/redgui
  --start-at "${START_AT}"
)

if [[ "${DRY_RUN}" -eq 1 ]]; then
  CMD+=(--dry-run)
fi
if [[ "${NO_PROMPT}" -eq 1 ]]; then
  CMD+=(--no-prompt)
fi

echo
echo "Launching review runner..."
"${CMD[@]}"
