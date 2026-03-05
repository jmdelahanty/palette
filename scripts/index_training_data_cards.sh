#!/usr/bin/env bash
set -euo pipefail

if [[ ! -x "scripts/py" ]]; then
  echo "Expected executable wrapper not found: scripts/py" >&2
  exit 2
fi

scripts/py -m fisheye.utils.index_training_data_cards "$@"
