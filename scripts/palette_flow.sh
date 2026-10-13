#!/usr/bin/env bash
# palette-flow: run or inspect a Palette workflow (docs/design/2026-10-07-workflow-runner).
#
#   scripts/palette_flow.sh run intake --config RUNNER.json [snakemake args...]
#   scripts/palette_flow.sh dry-run intake --config RUNNER.json
#   scripts/palette_flow.sh status intake --config RUNNER.json
#   scripts/palette_flow.sh pin-check intake --config RUNNER.json --to-ops OPS --to-lsf LSF
#
# Snakemake comes from the separate palette-flow conda env (ws1 only); all
# Palette code runs through the config's pinned ops deployment. One controller
# per workflow: a non-blocking flock makes overlapping cron ticks exit 0.
set -euo pipefail

usage() { sed -n '2,11p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }
[[ $# -ge 2 ]] || usage
action="$1"; workflow="$2"; shift 2
[[ "$workflow" == "intake" ]] || { echo "unknown workflow: $workflow" >&2; exit 2; }
[[ "${1:-}" == "--config" && -n "${2:-}" ]] || usage
config="$(realpath "$2")"; shift 2

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
snakemake="${PALETTE_FLOW_SNAKEMAKE:-$HOME/miniconda3/envs/palette-flow/bin/snakemake}"
flow_root="$("$repo/scripts/py" -c 'import json,sys; print(json.load(open(sys.argv[1]))["flow_root"])' "$config")"
ops_py="$("$repo/scripts/py" -c 'import json,sys; print(json.load(open(sys.argv[1]))["ops_deployment"])' "$config")/scripts/py"

case "$action" in
  status)
    exec "$ops_py" -m fisheye.flow.intake status --config "$config"
    ;;
  pin-check)
    # Before moving the runner to another deployment: refuses (exit 65) while
    # imported deliveries still need older code to register (§5.4).
    exec "$ops_py" -m fisheye.flow.intake pin-check --config "$config" "$@"
    ;;
  run|dry-run)
    mkdir -p "$flow_root/controller"
    exec 9>"$flow_root/controller/$workflow.lock"
    if ! flock -n 9; then
      echo "$(date -Is) $workflow: another controller is running; exiting" >&2
      exit 0
    fi
    extra=()
    [[ "$action" == "dry-run" ]] && extra+=(--dry-run)
    echo "$(date -Is) $workflow: controller start (config $config)" >&2
    "$snakemake" \
      --snakefile "$repo/workflows/$workflow.smk" \
      --directory "$flow_root/controller" \
      --config "runner_config=$config" \
      --cores 8 --resources lsf_jobs=4 registry_writer=1 \
      --retries 0 --keep-going --rerun-triggers mtime --nolock \
      --printshellcmds \
      "${extra[@]}" "$@"
    ;;
  *) usage ;;
esac
