#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Restarting Palette labeling web server..."
"${SCRIPT_DIR}/stop_labeling_web.sh"
"${SCRIPT_DIR}/start_labeling_web.sh" "$@"
