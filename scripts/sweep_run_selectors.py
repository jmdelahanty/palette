#!/usr/bin/env python
"""Read-only sweep of run-family selector state across every registered recording archive.

For each active ``source_recording`` dataset in the registry, opens the archive with
``use_consolidated=False`` (live attrs, not the consolidated cache) and records, per run family:
the selector attrs (``latest``, ``latest_complete``, ``latest_pending``, ``latest_any``,
``latest_materialized``, ``authoritative_run``), the sorted-last child, the registry
``run_name``/``latest_selector`` for the matching step, and model-identity attrs on every
candidate run. Writes one JSON line per dataset, then prints a per-family disagreement table.

Usage:
    scripts/py scripts/sweep_run_selectors.py --out sweep.jsonl [--registry PATH] [--limit N]
    scripts/py scripts/sweep_run_selectors.py --report sweep.jsonl

Never writes to the registry or any archive. See
docs/diagnostics/store_measurements_selectors_and_training_membership_2026-09-03.md.

This is historical discovery evidence, not stage-specific admission. Sorted
children, raw completion attrs, or registry rows do not establish authority;
missing selectors may be intentional. Never use this report to backfill them.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sqlite3
import sys
import time
from pathlib import Path

DEFAULT_REGISTRY = "/groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite"
FAMILY_STEP = {
    "detect_runs": "detect",
    "refined_detect_runs": "refined_detect",
    "crop_runs": "crop",
    "keypoints_runs": "keypoints",
    "refined_keypoints_runs": "refined_keypoints",
    "subject_mask_runs": "subject_masks",
    "refined_subject_masks_runs": "refined_subject_masks",
    "tracking_runs": "tracks",
}
SELECTOR_KEYS = ("latest", "latest_complete", "latest_pending", "latest_any", "latest_materialized", "authoritative_run")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
MODEL_CONTAINER_HINTS = ("model", "artifact", "provenance", "identity", "evidence", "manifest", "payload", "source", "binding", "authority")


def find_model_identity(obj, path="", out=None, depth=0):
    """Collect (attr_path, value) pairs that look like model identity: paths under /models/,
    weight files, 64-hex digests under model-ish keys, and registry run ids in schema bindings."""
    if out is None:
        out = []
    if depth > 8:
        return out
    if isinstance(obj, dict):
        for key, value in obj.items():
            key_path = f"{path}.{key}" if path else key
            key_l = key.lower()
            if isinstance(value, str):
                if key_l in ("registry_run_id", "training_run_id", "model_run_id"):
                    out.append((key_path, value))
                elif "model" in key_l and ("/models/" in value or value.endswith((".pt", ".engine", ".onnx"))):
                    out.append((key_path, value))
                elif ("model" in key_l or "artifact" in key_l or "weights" in key_l) and HEX64.match(value):
                    out.append((key_path, value))
                elif key_l.endswith("manifest_path") and "/models/" in value:
                    out.append((key_path, value))
            elif isinstance(value, (dict, list)) and any(h in key_l for h in MODEL_CONTAINER_HINTS):
                find_model_identity(value, key_path, out, depth + 1)
    elif isinstance(obj, list):
        for i, value in enumerate(obj[:20]):
            find_model_identity(value, f"{path}[{i}]", out, depth + 1)
    return out


def sweep(registry: str, out_path: Path, limit: int | None) -> None:
    import zarr

    con = sqlite3.connect(f"file:{registry}?mode=ro", uri=True)
    datasets = con.execute(
        "select dataset_id, recording_id, zarr_path, zarr_use from datasets "
        "where artifact_kind='source_recording' and status='active' order by zarr_path"
    ).fetchall()
    if limit:
        datasets = datasets[:limit]
    registry_rows = {}
    for dataset_id, step, run, status, selector in con.execute(
        "select dataset_id, step_name, run_name, status, json_extract(details_json,'$.latest_selector') from recording_step_status"
    ):
        registry_rows[(dataset_id, step)] = (run, status, selector)

    started = time.time()
    with out_path.open("w") as fh:
        for index, (dataset_id, recording_id, zarr_path, zarr_use) in enumerate(datasets):
            record = {"dataset_id": dataset_id, "recording_id": recording_id, "zarr_path": zarr_path, "zarr_use": zarr_use, "families": {}}
            try:
                root = zarr.open_group(zarr_path, mode="r", use_consolidated=False)
                for family, step in FAMILY_STEP.items():
                    if family not in root:
                        record["families"][family] = {"absent": True, "registry": registry_rows.get((dataset_id, step))}
                        continue
                    group = root[family]
                    attrs = dict(group.attrs)
                    keys = sorted(group.group_keys())
                    selectors = {k: attrs.get(k) for k in SELECTOR_KEYS if attrs.get(k) is not None}
                    family_record = {
                        "n_runs": len(keys),
                        "selectors": selectors,
                        "sorted_last": keys[-1] if keys else None,
                        "registry": registry_rows.get((dataset_id, step)),
                        "models": {},
                        "complete": {},
                        "dangling": [],
                    }
                    probe = {v for v in selectors.values() if isinstance(v, str)}
                    if keys:
                        probe.add(keys[-1])
                    reg = registry_rows.get((dataset_id, step))
                    if reg and reg[0]:
                        probe.add(reg[0])
                    for run in sorted(probe):
                        if run in group:
                            run_attrs = dict(group[run].attrs)
                            family_record["models"][run] = find_model_identity(run_attrs)[:16]
                            family_record["complete"][run] = bool(run_attrs.get("palette_completion_epoch") or run_attrs.get("status") == "complete")
                        else:
                            family_record["dangling"].append(run)
                    record["families"][family] = family_record
            except Exception as exc:  # noqa: BLE001 - diagnostic sweep records and continues
                record["error"] = f"{type(exc).__name__}: {exc}"
            fh.write(json.dumps(record) + "\n")
            fh.flush()
            if index % 50 == 0:
                print(f"{index}/{len(datasets)} {time.time() - started:.0f}s", file=sys.stderr, flush=True)
    print(f"done {len(datasets)} archives in {time.time() - started:.0f}s -> {out_path}", file=sys.stderr)


def report(sweep_path: Path) -> None:
    rows = [json.loads(line) for line in sweep_path.open()]
    errors = collections.Counter(r["error"].split(":")[0] for r in rows if "error" in r)
    print(f"archives={len(rows)} opened={len(rows) - sum(errors.values())} errors={dict(errors)}")
    columns = ["present", "multi", "latest_set", "latest_unset", "latest!=sorted", "latest!=complete", "pending",
               "reg_ok", "reg!=latest", "reg==sorted_no_latest", "reg_dangling", "reg_sel_sorted"]
    print(" | ".join(["family"] + columns))
    for family in FAMILY_STEP:
        c = collections.Counter()
        for r in rows:
            f = r["families"].get(family)
            if not f or f.get("absent") or f["n_runs"] == 0:
                continue
            c["present"] += 1
            c["multi"] += f["n_runs"] > 1
            sel = f["selectors"]
            latest = sel.get("latest")
            sorted_last = f["sorted_last"]
            c["latest_set" if latest else "latest_unset"] += 1
            c["latest!=sorted"] += bool(latest and sorted_last and latest != sorted_last)
            c["latest!=complete"] += bool(latest and sel.get("latest_complete") and latest != sel["latest_complete"])
            c["pending"] += bool(sel.get("latest_pending"))
            reg = f.get("registry")
            if reg and reg[1] == "ok" and reg[0]:
                c["reg_ok"] += 1
                c["reg!=latest"] += bool(latest and reg[0] != latest)
                c["reg==sorted_no_latest"] += bool(not latest and reg[0] == sorted_last)
                c["reg_dangling"] += reg[0] in (f.get("dangling") or [])
                c["reg_sel_sorted"] += bool(reg[2] and "sorted" in str(reg[2]))
        print(" | ".join([family] + [str(c[k]) for k in columns]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--registry", default=DEFAULT_REGISTRY)
    parser.add_argument("--out", type=Path, help="write sweep JSONL here (runs the sweep)")
    parser.add_argument("--report", type=Path, help="print the disagreement table for an existing sweep JSONL")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    if not args.out and not args.report:
        parser.error("pass --out to sweep and/or --report to summarize")
    if args.out:
        sweep(args.registry, args.out, args.limit)
    report(args.report or args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
