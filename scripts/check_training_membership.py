#!/usr/bin/env python
"""Report heuristic recording/model training-membership overlap candidates.

This read-only historical probe does not establish exact-example overlap,
held-out evaluation leakage, or scientific authority. Run/model selection is
heuristic; unresolved membership is not evidence that a recording is clean.

Expands ``training_sets.dataset_ids_json`` through ``dataset_lineage`` (``training_merge_source``)
to leaf recording ids per ``training_models`` row, then joins to the model identity found on each
recording's selected run in a ``scripts/sweep_run_selectors.py`` sweep. Model identity resolves by
registry run id in a schema binding, model path substring, model file sha256, or ONNX/TensorRT sha256.

Usage:
    scripts/py scripts/check_training_membership.py --sweep sweep.jsonl [--registry PATH] [--json OUT]

Read-only. See docs/diagnostics/store_measurements_selectors_and_training_membership_2026-09-03.md
and item T-1 of docs/diagnostics/training_data_and_model_provenance_review_2026-09-01.md.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sqlite3
from pathlib import Path

DEFAULT_REGISTRY = "/groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite"
FAMILY_TASK = {"detect_runs": "detect", "keypoints_runs": "pose", "subject_mask_runs": "subject_masks"}
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def load_membership(registry: str) -> dict:
    con = sqlite3.connect(f"file:{registry}?mode=ro", uri=True)
    dataset_recording = dict(con.execute("select dataset_id, recording_id from datasets"))
    parents: dict[str, set[str]] = {}
    for child, parent in con.execute(
        "select child_dataset_id, parent_dataset_id from dataset_lineage where relationship_type='training_merge_source'"
    ):
        parents.setdefault(child, set()).add(parent)

    def leaves(dataset_id: str, seen: set[str]) -> set[str]:
        if dataset_id in seen:
            return set()
        seen.add(dataset_id)
        if dataset_id in parents:
            return set().union(*(leaves(p, seen) for p in parents[dataset_id]))
        return {dataset_id}

    sets = {}
    for set_id, ids_json in con.execute("select set_id, dataset_ids_json from training_sets"):
        leaf = set()
        for d in json.loads(ids_json or "[]"):
            leaf |= leaves(d, set())
        recordings = {dataset_recording.get(l) or l.split(":")[0] for l in leaf}
        sets[set_id] = {"leaf_datasets": sorted(leaf), "recordings": sorted(r for r in recordings if r)}
    models = {}
    for run_id, set_id, task, status, sha, model_path in con.execute(
        "select run_id, set_id, task_type, status, model_sha256, model_path from training_models"
    ):
        models[run_id] = {"set_id": set_id, "task": task, "status": status, "sha": sha, "model_path": model_path,
                          "recordings": sets.get(set_id, {}).get("recordings", [])}
    deploy = {}
    for table in ("onnx_models", "tensorrt_models"):
        for run_id, sha in con.execute(f"select run_id, sha256 from {table}"):
            if sha:
                deploy[sha] = run_id
    return {"sets": sets, "models": models, "deploy": deploy}


def resolve_model(identity_pairs, models: dict, deploy: dict):
    for _, value in identity_pairs:
        if value in models:
            return value, "registry_run_id"
    for _, value in identity_pairs:
        if HEX64.match(value):
            for run_id, info in models.items():
                if info["sha"] == value:
                    return run_id, "model_sha"
            if value in deploy:
                return deploy[value], "deploy_sha"
    for _, value in identity_pairs:
        for run_id in models:
            if run_id in value:
                return run_id, "path"
    return None, None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--registry", default=DEFAULT_REGISTRY)
    parser.add_argument("--sweep", type=Path, required=True)
    parser.add_argument("--json", type=Path, help="write recording/model overlap candidates here")
    args = parser.parse_args()

    membership = load_membership(args.registry)
    models, deploy = membership["models"], membership["deploy"]
    rows = [json.loads(line) for line in args.sweep.open()]

    print("models:")
    for run_id, info in models.items():
        print(f"  {info['task'] or '?':13} {info['status']:7} train_recordings={len(info['recordings']):3} set={info['set_id']}  {run_id}")
    print("models without a set_id (membership unknowable):", [k for k, v in models.items() if not v["set_id"]])

    resolution = collections.Counter()
    per_model = collections.Counter()
    per_model_hits = collections.Counter()
    hits = []
    for r in rows:
        if "error" in r:
            continue
        for family in FAMILY_TASK:
            f = r["families"].get(family)
            if not f or f.get("absent") or f["n_runs"] == 0:
                continue
            reg = f.get("registry")
            latest = f["selectors"].get("latest")
            run = (reg[0] if reg and reg[1] == "ok" and reg[0] in f["models"] else None) \
                or (latest if latest in f["models"] else None) or f["sorted_last"]
            model, how = resolve_model(f["models"].get(run) or [], models, deploy)
            resolution[(family, "resolved" if model else "unresolved")] += 1
            if not model:
                continue
            per_model[model] += 1
            if r["recording_id"] in models[model]["recordings"]:
                per_model_hits[model] += 1
                hits.append({"recording_id": r["recording_id"], "zarr_use": r["zarr_use"], "family": family,
                             "run": run, "model_run_id": model, "resolved_by": how,
                             "registry_selected": bool(reg and reg[1] == "ok" and reg[0] == run)})
    print("\nresolution:", dict(sorted(resolution.items())))
    print("\nmodel -> recordings inferred / in own training set")
    for model, n in per_model.most_common():
        print(f"  {n:4} / {per_model_hits[model]:3}  {model}")
    print(f"\nRECORDING/MODEL OVERLAP CANDIDATES (not evaluation-leakage proof): {len(hits)}")
    for h in hits:
        print("  ", json.dumps(h))
    if args.json:
        args.json.write_text(json.dumps({"hits": hits, "resolution": {f"{k[0]}:{k[1]}": v for k, v in resolution.items()}}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
