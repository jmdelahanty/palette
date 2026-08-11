from __future__ import annotations

from pathlib import Path

import pytest

from fisheye.shared import execution_placement as placement_mod
from fisheye.shared.execution_placement import get_execution_placement_info


def test_execution_placement_captures_numa_affinity_and_memory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proc_root = tmp_path / "proc"
    proc_self = proc_root / "self"
    proc_self.mkdir(parents=True)
    (proc_self / "status").write_text(
        "Name:\tpython\n" "Cpus_allowed_list:\t2-3\n" "Mems_allowed_list:\t1\n",
        encoding="utf-8",
    )
    (proc_self / "numa_maps").write_text(
        "00400000 default file=/fixture N0=2 N1=1\n" "00800000 bind:1 anon=3 N1=3\n",
        encoding="utf-8",
    )
    node_root = tmp_path / "sys" / "node"
    (node_root / "node0").mkdir(parents=True)
    (node_root / "node1").mkdir(parents=True)
    (node_root / "online").write_text("0-1\n", encoding="utf-8")
    (node_root / "node0" / "cpulist").write_text("0-1,4-5\n", encoding="utf-8")
    (node_root / "node1" / "cpulist").write_text("2-3,6-7\n", encoding="utf-8")
    (node_root / "node0" / "meminfo").write_text(
        "Node 0 MemTotal: 1024 kB\n", encoding="utf-8"
    )
    (node_root / "node1" / "meminfo").write_text(
        "Node 1 MemTotal: 2048 kB\n", encoding="utf-8"
    )
    monkeypatch.setattr(placement_mod.os, "sched_getaffinity", lambda _pid: {2, 3})
    monkeypatch.setattr(placement_mod.os, "sysconf", lambda _name: 4096)
    monkeypatch.setattr(placement_mod.platform, "system", lambda: "Linux")
    monkeypatch.setenv("OMP_NUM_THREADS", "2")

    placement = get_execution_placement_info(
        proc_root=proc_root,
        sys_node_root=node_root,
    )

    assert placement["schema_id"] == "palette.execution_placement"
    assert placement["schema_version"] == 1
    assert placement["topology"] == {
        "available": True,
        "node_count": 2,
        "online_node_list": "0-1",
        "nodes": [
            {
                "node": 0,
                "cpu_list": "0-1,4-5",
                "cpu_count": 4,
                "memory_total_bytes": 1024 * 1024,
            },
            {
                "node": 1,
                "cpu_list": "2-3,6-7",
                "cpu_count": 4,
                "memory_total_bytes": 2048 * 1024,
            },
        ],
    }
    assert placement["process"] == {
        "sched_getaffinity_cpu_list": "2-3",
        "sched_getaffinity_cpu_count": 2,
        "status_allowed_cpu_list": "2-3",
        "status_allowed_memory_node_list": "1",
    }
    assert placement["memory_snapshot"] == {
        "available": True,
        "policy_counts": {"bind:1": 1, "default": 1},
        "resident_pages_by_node": {"0": 2, "1": 4},
        "total_resident_pages": 6,
        "page_size_bytes": 4096,
        "estimated_resident_bytes": 24_576,
    }
    assert placement["locality"] == {
        "cpu_affinity_node_list": "1",
        "allowed_memory_node_list": "1",
        "resident_pages_on_cpu_affinity_nodes": 4,
        "resident_pages_outside_cpu_affinity_nodes": 2,
        "resident_page_fraction_outside_cpu_affinity_nodes": pytest.approx(1.0 / 3.0),
        "interpretation": (
            "outside_fraction_detects_pages_outside_all_cpu_affinity_nodes_but_"
            "cannot_attribute_cross_node_access_when_affinity_spans_multiple_nodes"
        ),
    }
    assert placement["threading"]["environment"]["OMP_NUM_THREADS"] == "2"


def test_execution_placement_is_nonblocking_without_kernel_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _unavailable_affinity(_pid: int) -> set[int]:
        raise OSError("affinity unavailable")

    monkeypatch.setattr(placement_mod.os, "sched_getaffinity", _unavailable_affinity)

    placement = get_execution_placement_info(
        proc_root=tmp_path / "missing-proc",
        sys_node_root=tmp_path / "missing-sys-node",
    )

    assert placement["topology"] == {
        "available": False,
        "node_count": 0,
        "online_node_list": "",
        "nodes": [],
    }
    assert placement["process"]["sched_getaffinity_cpu_list"] is None
    assert placement["process"]["status_allowed_cpu_list"] is None
    assert placement["memory_snapshot"]["available"] is False
    assert placement["locality"]["cpu_affinity_node_list"] == ""
    assert (
        placement["locality"]["resident_page_fraction_outside_cpu_affinity_nodes"]
        is None
    )
