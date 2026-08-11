"""Portable, non-blocking CPU/NUMA execution-placement provenance."""

from __future__ import annotations

import multiprocessing
import os
import platform
from pathlib import Path
import re
import sys
from typing import Any

EXECUTION_PLACEMENT_SCHEMA_ID = "palette.execution_placement"
EXECUTION_PLACEMENT_SCHEMA_VERSION = 1

_THREAD_LIMIT_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def _read_optional_text(path: Path) -> str | None:
    """Read a small kernel metadata file without making provenance blocking."""

    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None


def _parse_index_list(value: str | None) -> tuple[int, ...] | None:
    """Parse Linux CPU/node lists such as ``0-3,8,10-11``."""

    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return ()
    indices: set[int] = set()
    try:
        for token in text.split(","):
            part = token.strip()
            if not part:
                return None
            if "-" in part:
                start_text, stop_text = part.split("-", 1)
                start = int(start_text)
                stop = int(stop_text)
                if start < 0 or stop < start:
                    return None
                indices.update(range(start, stop + 1))
            else:
                index = int(part)
                if index < 0:
                    return None
                indices.add(index)
    except (TypeError, ValueError):
        return None
    return tuple(sorted(indices))


def _format_index_list(indices: tuple[int, ...] | list[int] | set[int]) -> str:
    """Return one canonical compact Linux-style CPU/node list."""

    ordered = sorted({int(value) for value in indices})
    if not ordered:
        return ""
    ranges: list[str] = []
    start = previous = ordered[0]
    for value in ordered[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = value
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(ranges)


def _canonical_index_list(value: str | None) -> str | None:
    parsed = _parse_index_list(value)
    return None if parsed is None else _format_index_list(parsed)


def _node_memory_total_bytes(meminfo: str | None) -> int | None:
    if not meminfo:
        return None
    match = re.search(r"\bMemTotal:\s*(\d+)\s*kB\b", meminfo)
    return int(match.group(1)) * 1024 if match else None


def _numa_topology(sys_node_root: Path) -> dict[str, Any]:
    online = _parse_index_list(_read_optional_text(sys_node_root / "online"))
    if online is None:
        discovered: list[int] = []
        try:
            candidates = list(sys_node_root.glob("node[0-9]*"))
        except OSError:
            candidates = []
        for candidate in candidates:
            suffix = candidate.name.removeprefix("node")
            if suffix.isdigit():
                discovered.append(int(suffix))
        online = tuple(sorted(set(discovered)))

    nodes: list[dict[str, Any]] = []
    for node_index in online:
        node_root = sys_node_root / f"node{node_index}"
        cpu_list = _canonical_index_list(_read_optional_text(node_root / "cpulist"))
        parsed_cpus = _parse_index_list(cpu_list)
        nodes.append(
            {
                "node": int(node_index),
                "cpu_list": cpu_list,
                "cpu_count": len(parsed_cpus) if parsed_cpus is not None else None,
                "memory_total_bytes": _node_memory_total_bytes(
                    _read_optional_text(node_root / "meminfo")
                ),
            }
        )
    return {
        "available": bool(nodes),
        "node_count": len(nodes),
        "online_node_list": _format_index_list(online),
        "nodes": nodes,
    }


def _proc_status_fields(proc_root: Path) -> dict[str, str]:
    text = _read_optional_text(proc_root / "self" / "status")
    if text is None:
        return {}
    fields: dict[str, str] = {}
    for line in text.splitlines():
        key, separator, value = line.partition(":")
        if separator:
            fields[key.strip()] = value.strip()
    return fields


def _numa_memory_snapshot(proc_root: Path) -> dict[str, Any]:
    text = _read_optional_text(proc_root / "self" / "numa_maps")
    if text is None:
        return {
            "available": False,
            "policy_counts": {},
            "resident_pages_by_node": {},
            "total_resident_pages": 0,
            "page_size_bytes": None,
            "estimated_resident_bytes": None,
        }

    policy_counts: dict[str, int] = {}
    pages_by_node: dict[str, int] = {}
    for line in text.splitlines():
        tokens = line.split()
        if len(tokens) < 2:
            continue
        policy = tokens[1]
        if policy == "default" or policy.startswith(
            ("bind:", "interleave:", "prefer:", "preferred:")
        ):
            policy_counts[policy] = policy_counts.get(policy, 0) + 1
        for token in tokens[2:]:
            match = re.fullmatch(r"N(\d+)=(\d+)", token)
            if match:
                node = match.group(1)
                pages_by_node[node] = pages_by_node.get(node, 0) + int(match.group(2))
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, TypeError, ValueError):
        page_size = None
    total_pages = sum(pages_by_node.values())
    return {
        "available": True,
        "policy_counts": dict(sorted(policy_counts.items())),
        "resident_pages_by_node": {
            key: pages_by_node[key]
            for key in sorted(pages_by_node, key=lambda value: int(value))
        },
        "total_resident_pages": int(total_pages),
        "page_size_bytes": page_size,
        "estimated_resident_bytes": (
            int(total_pages * page_size) if page_size is not None else None
        ),
    }


def _threading_runtime() -> dict[str, Any]:
    environment = {
        name: os.environ[name]
        for name in _THREAD_LIMIT_ENV_VARS
        if os.environ.get(name) is not None
    }
    runtime: dict[str, Any] = {
        "multiprocessing_start_method": multiprocessing.get_start_method(
            allow_none=True
        ),
    }
    torch_module = sys.modules.get("torch")
    if torch_module is not None:
        for key, attribute in (
            ("torch_num_threads", "get_num_threads"),
            ("torch_num_interop_threads", "get_num_interop_threads"),
        ):
            function = getattr(torch_module, attribute, None)
            if callable(function):
                try:
                    runtime[key] = int(function())
                except Exception:
                    pass
    return {"environment": environment, "runtime": runtime}


def _numa_locality_summary(
    topology: dict[str, Any],
    process: dict[str, Any],
    memory_snapshot: dict[str, Any],
) -> dict[str, Any]:
    affinity_cpus = _parse_index_list(process.get("sched_getaffinity_cpu_list"))
    affinity_nodes: list[int] = []
    if affinity_cpus is not None:
        affinity_cpu_set = set(affinity_cpus)
        for node in topology.get("nodes", []):
            node_cpus = _parse_index_list(node.get("cpu_list"))
            if node_cpus is not None and affinity_cpu_set.intersection(node_cpus):
                affinity_nodes.append(int(node["node"]))

    pages_by_node = memory_snapshot.get("resident_pages_by_node")
    total_pages = memory_snapshot.get("total_resident_pages")
    on_nodes: int | None = None
    outside_nodes: int | None = None
    outside_fraction: float | None = None
    if (
        affinity_nodes
        and isinstance(pages_by_node, dict)
        and type(total_pages) is int
        and total_pages > 0
    ):
        affinity_node_keys = {str(value) for value in affinity_nodes}
        on_nodes = sum(
            int(value)
            for node, value in pages_by_node.items()
            if str(node) in affinity_node_keys
        )
        outside_nodes = int(total_pages) - int(on_nodes)
        outside_fraction = float(outside_nodes / total_pages)
    return {
        "cpu_affinity_node_list": _format_index_list(tuple(affinity_nodes)),
        "allowed_memory_node_list": process.get("status_allowed_memory_node_list"),
        "resident_pages_on_cpu_affinity_nodes": on_nodes,
        "resident_pages_outside_cpu_affinity_nodes": outside_nodes,
        "resident_page_fraction_outside_cpu_affinity_nodes": outside_fraction,
        "interpretation": (
            "outside_fraction_detects_pages_outside_all_cpu_affinity_nodes_but_"
            "cannot_attribute_cross_node_access_when_affinity_spans_multiple_nodes"
        ),
    }


def get_execution_placement_info(
    *,
    proc_root: Path = Path("/proc"),
    sys_node_root: Path = Path("/sys/devices/system/node"),
) -> dict[str, Any]:
    """Capture compact NUMA, affinity, and thread placement for one run.

    Unsupported kernels and absent proc/sysfs mounts produce explicit
    unavailable fields rather than blocking scientific execution or publication.
    """

    status = _proc_status_fields(proc_root)
    try:
        sched_affinity = tuple(sorted(int(value) for value in os.sched_getaffinity(0)))
    except (AttributeError, OSError, TypeError, ValueError):
        sched_affinity = None
    topology = _numa_topology(sys_node_root)
    process = {
        "sched_getaffinity_cpu_list": (
            _format_index_list(sched_affinity) if sched_affinity is not None else None
        ),
        "sched_getaffinity_cpu_count": (
            len(sched_affinity) if sched_affinity is not None else None
        ),
        "status_allowed_cpu_list": _canonical_index_list(
            status.get("Cpus_allowed_list")
        ),
        "status_allowed_memory_node_list": _canonical_index_list(
            status.get("Mems_allowed_list")
        ),
    }
    memory_snapshot = _numa_memory_snapshot(proc_root)
    return {
        "schema_id": EXECUTION_PLACEMENT_SCHEMA_ID,
        "schema_version": EXECUTION_PLACEMENT_SCHEMA_VERSION,
        "platform": platform.system() or "unknown",
        "topology": topology,
        "process": process,
        "memory_snapshot": memory_snapshot,
        "locality": _numa_locality_summary(topology, process, memory_snapshot),
        "threading": _threading_runtime(),
    }


__all__ = [
    "EXECUTION_PLACEMENT_SCHEMA_ID",
    "EXECUTION_PLACEMENT_SCHEMA_VERSION",
    "get_execution_placement_info",
]
