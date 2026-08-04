from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import platform
import shutil

import pytest

from fisheye.diagnostics.trace_storage_io import (
    parse_gnu_time,
    parse_strace_reads,
    require_storage_io_trace_receipt,
    run_traced_storage_command,
)
from fisheye.analysis_workflows.storage_benchmark_catalog import (
    DERIVED_ANALYSIS_STORAGE_BENCHMARK_BY_STAGE,
    DerivedAnalysisStorageBenchmark,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256


def test_parse_strace_reads_attributes_only_explicit_target_paths(
    tmp_path: Path,
) -> None:
    target = tmp_path / "store.zarr"
    payload = target / "array" / "c" / "0"
    metadata = target / "array" / "zarr.json"
    payload.parent.mkdir(parents=True)
    payload.write_bytes(b"payload")
    metadata.write_text("{}", encoding="utf-8")
    trace = tmp_path / "strace.123"
    trace.write_text(
        "\n".join(
            (
                f'read(3<{payload}>, "", 7) = 7',
                f'pread64(3<{metadata}>, "", 2, 0) = 2',
                f"readv(3<{payload}>, [], 1) = 0",
                f'read(3<{payload}>, "", 7) = -1 EIO (Input/output error)',
                'read(4</usr/lib/libc.so>, "", 32) = 32',
            )
        )
        + "\n",
        encoding="utf-8",
    )

    result = parse_strace_reads([trace], target_roots=[target])

    assert result["read_operations"] == 3
    assert result["read_bytes"] == 9
    assert result["unique_object_count"] == 2
    assert result["unattributed_successful_read_count"] == 1
    assert result["by_storage_kind"]["payload"] == {
        "read_operations": 2,
        "read_bytes": 7,
        "unique_object_count": 1,
        "paths": [str(payload)],
    }
    assert result["by_storage_kind"]["metadata"]["read_bytes"] == 2
    assert result["attributed_syscall_counts"] == {
        "read": 1,
        "pread64": 1,
        "readv": 1,
        "preadv": 0,
        "preadv2": 0,
    }


def test_parse_gnu_time_requires_complete_nonnegative_resource_fields(
    tmp_path: Path,
) -> None:
    path = tmp_path / "resource.txt"
    path.write_text(
        """\
        User time (seconds): 1.25
        System time (seconds): 0.50
        Elapsed (wall clock) time (h:mm:ss or m:ss): 1:02.30
        Maximum resident set size (kbytes): 2048
        File system inputs: 8
        File system outputs: 3
        """,
        encoding="utf-8",
    )

    assert parse_gnu_time(path) == {
        "user_seconds": 1.25,
        "system_seconds": 0.5,
        "elapsed_seconds": 62.3,
        "peak_rss_bytes": 2 * 1024 * 1024,
        "filesystem_input_count": 8,
        "filesystem_output_count": 3,
    }

    path.write_text("User time (seconds): 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="omits required"):
        parse_gnu_time(path)


@pytest.mark.skipif(
    platform.system() != "Linux"
    or shutil.which("strace") is None
    or shutil.which("time") is None,
    reason="real process-tree tracing requires Linux strace and GNU time",
)
def test_real_process_tree_trace_is_replayable_and_nonpromoting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "fixture.zarr"
    payload = target / "array" / "c" / "0"
    payload.parent.mkdir(parents=True)
    payload.write_bytes(b"0123456789")
    (target / "zarr.json").write_text("{}", encoding="utf-8")
    output = tmp_path / ".palette_benchmarks" / "physical-io"
    matrix_path = tmp_path / ".palette_benchmarks" / "matrix.json"
    matrix_path.parent.mkdir()
    matrix = {
        "schema_id": "palette.exact_tabular_candidate_read_matrix",
        "schema_version": 2,
        "payload": {"fixture": True},
        "payload_digest": canonical_json_sha256({"fixture": True}),
    }
    matrix_path.write_text(json.dumps(matrix), encoding="utf-8")
    normalized = {
        "stage_id": "swim_bouts",
        "family_id": "swim_bouts",
        "archive_path": str(target),
        "source_run_path": "analysis/swim_bout_runs/source",
        "candidate_run_path": "analysis/swim_bout_runs/candidate",
    }
    monkeypatch.setattr(
        DerivedAnalysisStorageBenchmark,
        "validated_matrix_identity",
        lambda self, value: (
            normalized
            if self is DERIVED_ANALYSIS_STORAGE_BENCHMARK_BY_STAGE["swim_bouts"]
            and value == matrix
            else pytest.fail("unexpected benchmark matrix binding")
        ),
    )

    receipt = run_traced_storage_command(
        ["/usr/bin/head", "-c", "5", str(payload)],
        target_roots=[target],
        output_dir=output,
        stage_id="swim_bouts",
        matrix_result_path=matrix_path,
    )

    require_storage_io_trace_receipt(
        receipt,
        evidence_root=output,
        replay_benchmark_binding=True,
    )
    persisted = json.loads((output / "receipt.json").read_text(encoding="utf-8"))
    assert persisted == receipt
    assert receipt["payload"]["command_completed"] is True
    assert receipt["payload"]["physical_io_measured"] is True
    assert receipt["payload"]["target_reads_observed"] is True
    assert receipt["payload"]["benchmark_evidence_bound"] is True
    assert receipt["payload"]["benchmark_binding"]["normalized_identity"] == normalized
    assert receipt["payload"]["promotion_authorized"] is False
    assert receipt["payload"]["reads"]["read_operations"] >= 1
    assert receipt["payload"]["reads"]["read_bytes"] >= 5
    assert (
        receipt["payload"]["reads"]["by_storage_kind"]["payload"]["unique_object_count"]
        == 1
    )

    tampered = deepcopy(receipt)
    tampered["payload"]["reads"]["read_bytes"] += 1
    tampered["payload_digest"] = canonical_json_sha256(tampered["payload"])
    with pytest.raises(ValueError, match="aggregate"):
        require_storage_io_trace_receipt(tampered, evidence_root=output)

    raw = output / receipt["payload"]["raw_artifacts"][0]["path"]
    raw.write_text(raw.read_text(encoding="utf-8") + "tamper\n", encoding="utf-8")
    with pytest.raises(ValueError, match="digest or size"):
        require_storage_io_trace_receipt(receipt, evidence_root=output)


def test_trace_refuses_broad_targets_and_nonbenchmark_outputs(tmp_path: Path) -> None:
    target = tmp_path / "fixture.zarr"
    target.mkdir()
    with pytest.raises(ValueError, match="benchmark-only"):
        run_traced_storage_command(
            ["/usr/bin/true"],
            target_roots=[target],
            output_dir=tmp_path / "evidence",
        )
    with pytest.raises(ValueError, match="too broad"):
        run_traced_storage_command(
            ["/usr/bin/true"],
            target_roots=[Path("/")],
            output_dir=tmp_path / ".palette_benchmarks" / "evidence",
        )
