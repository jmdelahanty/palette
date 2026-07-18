#!/usr/bin/env bash
set -euo pipefail

# Compatibility entry point. New workflows should use
# scripts/submit_chaser_analytics_bsub.sh and select a versioned protocol
# adapter explicitly.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/submit_chaser_analytics_bsub.sh" "$@"
