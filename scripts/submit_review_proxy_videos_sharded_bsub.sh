#!/usr/bin/env bash
set -euo pipefail

scripts/py -m fisheye.utils.submit_review_proxy_videos_sharded_bsub "$@"
