#!/usr/bin/env bash
set -euo pipefail

scripts/py -m fisheye.cluster.whole_recording_analysis "$@"
