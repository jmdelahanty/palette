# Transfer-v2 intake: import each delivery on LSF, then register it on ws1.
#
# Design: docs/design/2026-10-07-workflow-runner (§4-§6). Run only through
# scripts/palette_flow.sh, which supplies --config runner_config=... and the
# required flags (--retries 0 --keep-going --rerun-triggers mtime --nolock).
#
# Rules for this file (tests/unit/fisheye/test_flow_intake_snakefile.py):
# - every output is a sentinel under <flow_root>/intake/; never a Zarr, a
#   recording, a registry or staging path, and never directory();
# - every rule is local: Snakemake runs only on ws1. LSF work happens inside
#   `fisheye.flow.intake import`, which submits and waits on NFS evidence;
# - Palette decides done: the step helpers write a sentinel only from a
#   fisheye.intake result whose verdict is true.

import json
import subprocess

RUNNER_CONFIG = config["runner_config"]
with open(RUNNER_CONFIG) as handle:
    FLOW = json.load(handle)
INTAKE = FLOW["flow_root"] + "/intake"
OPS_PY = FLOW["ops_deployment"] + "/scripts/py"
STEP = f"{OPS_PY} -m fisheye.flow.intake"

_plan = subprocess.run(
    [OPS_PY, "-m", "fisheye.flow.intake", "plan", "--config", RUNNER_CONFIG],
    check=True, capture_output=True, text=True,
)
PLAN = json.loads(_plan.stdout)
for item in PLAN["held"] + PLAN["skipped"]:
    print(f"intake: not driving {item['snapshot_sha'][:16]}: {item['why']}")
if PLAN["registry_error"]:
    print(f"intake: registry unreadable this tick: {PLAN['registry_error']}")

localrules: all, import_delivery, register_delivery


rule all:
    input:
        expand(INTAKE + "/{sha}/register.done.json", sha=PLAN["drive"]),


rule import_delivery:
    output:
        INTAKE + "/{sha}/import.done.json",
    resources:
        lsf_jobs=1,
    shell:
        "{STEP} import {wildcards.sha} --config {RUNNER_CONFIG}"


rule register_delivery:
    input:
        INTAKE + "/{sha}/import.done.json",
    output:
        INTAKE + "/{sha}/register.done.json",
    resources:
        registry_writer=1,
    shell:
        "{STEP} register {wildcards.sha} --config {RUNNER_CONFIG}"
