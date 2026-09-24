from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import zarr

from fisheye.registry.maintenance import _required_canonical_detection_errors
from fisheye.shared.instance_keys import mint_detection_instance_keys
from fisheye.shared.zarr.canonical_detection_activation import (
    CanonicalDetectionSelectorActivation,
)
from fisheye.shared.zarr.canonical_detection_manifest import (
    CANONICAL_DETECTION_AUTHORITY_CONTRACT_ATTR,
    CANONICAL_DETECTION_AUTHORITY_CONTRACT_V3,
    CANONICAL_DETECTION_AUTHORITY_DIGEST_ATTR,
    require_active_coordinate_canonical_detection,
)
from fisheye.shared.zarr.detection_snapshot_publication import (
    CanonicalDetectionActivationRefused,
    activate_canonical_detection_successor,
    publish_canonical_detection_successor,
)
from fisheye.utils import publish_canonical_detection_successor as cli


RECORDING_IDENTITY = "successor_activation_recording"
SOURCE = "detect_runs/detect_source"
SUCCESSOR = "detect_source_canonical_v3_active"


def _build_archive(archive: Path) -> None:
    root = zarr.open_group(str(archive), mode="w", zarr_format=3)
    root.attrs["recording_id"] = RECORDING_IDENTITY
    family = root.create_group("detect_runs")
    family.attrs.update({"latest": "detect_source", "latest_complete": "detect_source"})
    raw = family.create_group("detect_source")
    raw.attrs.update({"source_video_width": 640, "source_video_height": 480})
    frames = np.asarray([1, 3], dtype=np.int32)
    bbox = np.asarray(
        [[0.25, 0.5, 0.1, 0.2], [0.75, 0.5, 0.1, 0.2]],
        dtype=np.float64,
    )
    classes = np.asarray([1, 3], dtype=np.int32)
    raw.create_array("frame_indices", data=frames)
    raw.create_array("bbox_norm_coords", data=bbox)
    raw.create_array("scores", data=np.asarray([0.9, 0.8], dtype=np.float32))
    raw.create_array("class_ids", data=classes)
    raw.create_array(
        "instance_key",
        data=mint_detection_instance_keys(
            recording_identity=RECORDING_IDENTITY,
            frame_indices=frames,
            bbox_norm_coords=bbox.astype(np.float32),
            class_ids=classes,
        ),
    )
    raw.create_array("frame_counts", data=np.asarray([0, 1, 0, 1], dtype=np.int32))
    zarr.consolidate_metadata(str(archive))


@pytest.fixture
def archive(tmp_path: Path) -> Path:
    path = tmp_path / "recording_analysis.zarr"
    _build_archive(path)
    return path


@pytest.fixture
def scratch(tmp_path: Path) -> Path:
    path = tmp_path / "scratch"
    path.mkdir()
    return path


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file():
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _activate(archive: Path, scratch: Path, **kwargs) -> dict:
    return activate_canonical_detection_successor(
        analysis_zarr=archive,
        source_detect_group_path=SOURCE,
        recording_identity=RECORDING_IDENTITY,
        successor_run_id=kwargs.pop("successor_run_id", SUCCESSOR),
        scratch_root=scratch,
        **kwargs,
    )


def _publish(archive: Path, scratch: Path, run_id: str, *, eligible: bool) -> dict:
    return publish_canonical_detection_successor(
        analysis_zarr=archive,
        source_detect_group_path=SOURCE,
        recording_identity=RECORDING_IDENTITY,
        successor_run_id=run_id,
        scratch_root=scratch,
        selector_eligible=eligible,
    )


def _assert_active(archive: Path, run_id: str, digest: str) -> None:
    for use_consolidated in (False, True):
        root = zarr.open_group(
            str(archive), mode="r", use_consolidated=use_consolidated
        )
        family = root["detect_runs"]
        assert family.attrs["latest"] == run_id
        assert family.attrs["latest_complete"] == run_id
        assert family.attrs[CANONICAL_DETECTION_AUTHORITY_CONTRACT_ATTR] == (
            CANONICAL_DETECTION_AUTHORITY_CONTRACT_V3
        )
        assert family.attrs[CANONICAL_DETECTION_AUTHORITY_DIGEST_ATTR] == digest
        assert (
            _required_canonical_detection_errors(
                family, family[run_id], run_name=run_id, required=True
            )
            == ()
        )
        require_active_coordinate_canonical_detection(
            root,
            group_path=f"detect_runs/{run_id}",
            expected_manifest_digest=digest,
        )


def test_eligible_option_seals_manifest_but_leaves_selectors(
    archive: Path, scratch: Path
) -> None:
    result = _publish(archive, scratch, SUCCESSOR, eligible=True)

    assert result["successor"]["manifest_selector_eligible"] is True
    assert result["selector_eligible"] is False
    root = zarr.open_group(str(archive), mode="r", use_consolidated=True)
    family = root["detect_runs"]
    run = family[SUCCESSOR]
    manifest = run.attrs["run_manifest"]
    assert manifest["payload"]["publication"]["stage_selector_eligible"] is True
    assert manifest["payload"]["source_evidence_kind"] == "legacy_conversion"
    assert run.attrs["stage_selector_eligible"] is False
    assert family.attrs["latest"] == "detect_source"
    assert CANONICAL_DETECTION_AUTHORITY_CONTRACT_ATTR not in family.attrs


def test_default_successor_publication_remains_ineligible(
    archive: Path, scratch: Path
) -> None:
    result = _publish(archive, scratch, "detect_source_canonical_v3", eligible=False)

    assert result["successor"]["manifest_selector_eligible"] is False
    run = zarr.open_group(str(archive), mode="r", use_consolidated=False)[
        "detect_runs/detect_source_canonical_v3"
    ]
    assert (
        run.attrs["run_manifest"]["payload"]["publication"]["stage_selector_eligible"]
        is False
    )


def test_dry_run_plans_publication_without_writes(archive: Path, scratch: Path) -> None:
    before = _tree_digest(archive)

    result = _activate(archive, scratch)

    assert result["status"] == "planned"
    assert result["mode"] == "dry_run"
    assert result["zarr_writes"] is False
    assert result["action"] == "publish_and_activate"
    assert result["selector_state"] == "unactivated"
    assert result["selectors_before"]["latest"] == {
        "present": True,
        "value": "detect_source",
    }
    assert result["parity"]["status"] == "no_existing_successor"
    assert _tree_digest(archive) == before
    assert list(scratch.iterdir()) == []


def test_publish_and_activate_satisfies_canonical_contract(
    archive: Path, scratch: Path
) -> None:
    result = _activate(archive, scratch, apply=True)

    assert result["status"] == "activated"
    assert result["action_taken"] == "published_and_activated"
    assert result["registry_updated"] is False
    digest = result["successor"]["manifest_digest"]
    assert (
        result["successor"]["logical_content_digest"]
        == (result["successor"]["expected_logical_content_digest"])
    )
    assert result["publication"]["successor"]["manifest_digest"] == digest
    assert result["selectors_before"]["latest"]["value"] == "detect_source"
    assert result["selectors_after"]["latest"]["value"] == SUCCESSOR
    assert result["selectors_after"][CANONICAL_DETECTION_AUTHORITY_DIGEST_ATTR] == {
        "present": True,
        "value": digest,
    }
    _assert_active(archive, SUCCESSOR, digest)
    run = zarr.open_group(str(archive), mode="r", use_consolidated=False)[
        f"detect_runs/{SUCCESSOR}"
    ]
    assert run.attrs["stage_selector_eligible"] is True
    assert run.attrs["production_selector_activation"] == "complete"
    # The legacy source is retained unchanged.
    assert (
        "detect_source"
        in zarr.open_group(str(archive), mode="r", use_consolidated=False)[
            "detect_runs"
        ]
    )


def test_rerun_on_active_archive_is_noop(archive: Path, scratch: Path) -> None:
    first = _activate(archive, scratch, apply=True)
    before = _tree_digest(archive)

    second = _activate(archive, scratch, apply=True)

    assert second["status"] == "already_active"
    assert second["zarr_writes"] is False
    assert second["selector_state"] == "active"
    assert (
        second["successor"]["manifest_digest"]
        == (first["successor"]["manifest_digest"])
    )
    assert _tree_digest(archive) == before


def test_published_but_unactivated_successor_resumes(
    archive: Path, scratch: Path
) -> None:
    published = _publish(archive, scratch, SUCCESSOR, eligible=True)

    planned = _activate(archive, scratch)
    assert planned["action"] == "activate"
    result = _activate(archive, None, apply=True)

    assert result["action_taken"] == "activated_existing_successor"
    assert result["publication"] is None
    _assert_active(archive, SUCCESSOR, published["successor"]["manifest_digest"])


def test_partial_activation_resumes(archive: Path, scratch: Path) -> None:
    published = _publish(archive, scratch, SUCCESSOR, eligible=True)
    family = zarr.open_group(str(archive), mode="a", use_consolidated=False)[
        "detect_runs"
    ]
    family.attrs["latest"] = SUCCESSOR  # interrupted after the first write

    result = _activate(archive, scratch, apply=True)

    assert result["selector_state"] == "partial_activation"
    assert result["action_taken"] == "resumed_partial_activation"
    _assert_active(archive, SUCCESSOR, published["successor"]["manifest_digest"])


def test_stale_consolidated_activation_is_repaired(
    archive: Path, scratch: Path
) -> None:
    first = _activate(archive, scratch, apply=True)
    digest = first["successor"]["manifest_digest"]
    # Reproduce a crash after direct writes but before reconsolidation.
    family = zarr.open_group(str(archive), mode="a", use_consolidated=False)[
        "detect_runs"
    ]
    family.attrs["latest"] = "detect_source"
    zarr.consolidate_metadata(str(archive))
    family.attrs["latest"] = SUCCESSOR

    result = _activate(archive, scratch, apply=True)

    assert result["status"] == "activated"
    assert result["action_taken"] == "repaired_consolidated_activation_visibility"
    _assert_active(archive, SUCCESSOR, digest)


def test_tampered_source_is_refused(archive: Path, scratch: Path) -> None:
    _publish(archive, scratch, SUCCESSOR, eligible=True)
    source = zarr.open_group(str(archive / SOURCE), mode="a", use_consolidated=False)
    source["scores"][0] = np.float32(0.1)

    with pytest.raises(CanonicalDetectionActivationRefused) as info:
        _activate(archive, scratch, apply=True)

    assert info.value.receipt["status"] == "refused"
    family = zarr.open_group(str(archive), mode="r", use_consolidated=False)[
        "detect_runs"
    ]
    assert family.attrs["latest"] == "detect_source"
    assert family[SUCCESSOR].attrs["stage_selector_eligible"] is False


def test_tampered_successor_array_is_refused(archive: Path, scratch: Path) -> None:
    _publish(archive, scratch, SUCCESSOR, eligible=True)
    successor = zarr.open_group(
        str(archive / "detect_runs" / SUCCESSOR), mode="a", use_consolidated=False
    )
    successor["instances/scores"][0] = np.float32(0.5)

    with pytest.raises(CanonicalDetectionActivationRefused, match="validation"):
        _activate(archive, scratch, apply=True)


@pytest.mark.parametrize("drifted", ["latest", "latest_complete"])
def test_selector_drift_is_refused(archive: Path, scratch: Path, drifted: str) -> None:
    _publish(archive, scratch, SUCCESSOR, eligible=True)
    family = zarr.open_group(str(archive), mode="a", use_consolidated=False)[
        "detect_runs"
    ]
    family.create_group("detect_other")
    family.attrs[drifted] = "detect_other"

    with pytest.raises(CanonicalDetectionActivationRefused, match="drifted"):
        _activate(archive, scratch, apply=True)

    family = zarr.open_group(str(archive), mode="r", use_consolidated=False)[
        "detect_runs"
    ]
    assert family.attrs[drifted] == "detect_other"
    assert CANONICAL_DETECTION_AUTHORITY_CONTRACT_ATTR not in family.attrs


def test_drift_before_publication_writes_nothing(archive: Path, scratch: Path) -> None:
    family = zarr.open_group(str(archive), mode="a", use_consolidated=False)[
        "detect_runs"
    ]
    family.attrs["latest_complete"] = "detect_other"
    before = _tree_digest(archive)

    with pytest.raises(CanonicalDetectionActivationRefused, match="drifted"):
        _activate(archive, scratch, apply=True)

    assert _tree_digest(archive) == before


def test_ineligible_successor_is_refused(archive: Path, scratch: Path) -> None:
    _publish(archive, scratch, SUCCESSOR, eligible=False)

    with pytest.raises(CanonicalDetectionActivationRefused, match="ineligible"):
        _activate(archive, scratch, apply=True)

    family = zarr.open_group(str(archive), mode="r", use_consolidated=False)[
        "detect_runs"
    ]
    assert family.attrs["latest"] == "detect_source"


def test_shared_writer_refuses_ineligible_manifest(
    archive: Path, scratch: Path
) -> None:
    _publish(archive, scratch, SUCCESSOR, eligible=False)
    root = zarr.open_group(str(archive), mode="a", use_consolidated=False)
    run = root[f"detect_runs/{SUCCESSOR}"]
    writer = CanonicalDetectionSelectorActivation(
        archive=archive,
        run_id=SUCCESSOR,
        manifest=dict(run.attrs["run_manifest"]),
        plans=None,
    )

    with pytest.raises(RuntimeError, match="not selector eligible"):
        writer.activate(root, root["detect_runs"], run)

    assert root["detect_runs"].attrs["latest"] == "detect_source"


def test_parity_with_existing_ineligible_successor_matches(
    archive: Path, scratch: Path
) -> None:
    existing = _publish(archive, scratch, "detect_source_canonical_v3", eligible=False)

    result = _activate(archive, scratch, apply=True)

    parity = result["parity"]
    assert parity["status"] == "match"
    [record] = parity["existing_successors"]
    assert record["run_id"] == "detect_source_canonical_v3"
    assert record["manifest_selector_eligible"] is False
    assert (
        record["logical_content_digest"]
        == (existing["successor"]["logical_content_digest"])
    )
    assert (
        result["successor"]["logical_content_digest"]
        == (existing["successor"]["logical_content_digest"])
    )
    _assert_active(archive, SUCCESSOR, result["successor"]["manifest_digest"])


def test_parity_mismatch_with_existing_successor_is_refused(
    archive: Path, scratch: Path
) -> None:
    _publish(archive, scratch, "detect_source_canonical_v3", eligible=False)
    source = zarr.open_group(str(archive / SOURCE), mode="a", use_consolidated=False)
    source["bbox_norm_coords"][0, 0] = 0.3
    before = _tree_digest(archive)

    with pytest.raises(CanonicalDetectionActivationRefused, match="differs") as info:
        _activate(archive, scratch, apply=True)

    parity = info.value.receipt["parity"]
    assert parity["status"] == "mismatch"
    assert parity["existing_successors"][0]["matches"] is False
    assert _tree_digest(archive) == before


def test_cli_dry_run_and_apply_emit_receipts(
    archive: Path, scratch: Path, tmp_path: Path
) -> None:
    receipt = tmp_path / "receipt.json"
    common = [
        "--analysis-zarr",
        str(archive),
        "--source-detect-group",
        SOURCE,
        "--recording-identity",
        RECORDING_IDENTITY,
        "--successor-run",
        SUCCESSOR,
        "--scratch-root",
        str(scratch),
        "--result-json",
        str(receipt),
        "--activate",
    ]
    before = _tree_digest(archive)

    assert cli.main(common) == 0
    planned = json.loads(receipt.read_text())
    assert planned["mode"] == "dry_run"
    assert planned["action"] == "publish_and_activate"
    assert _tree_digest(archive) == before

    assert cli.main([*common, "--apply"]) == 0
    applied = json.loads(receipt.read_text())
    assert applied["status"] == "activated"
    _assert_active(archive, SUCCESSOR, applied["successor"]["manifest_digest"])

    assert cli.main([*common, "--apply"]) == 0
    assert json.loads(receipt.read_text())["status"] == "already_active"


def test_cli_refusal_writes_refused_receipt(
    archive: Path, scratch: Path, tmp_path: Path
) -> None:
    _publish(archive, scratch, SUCCESSOR, eligible=False)
    receipt = tmp_path / "receipt.json"

    code = cli.main(
        [
            "--analysis-zarr",
            str(archive),
            "--source-detect-group",
            SOURCE,
            "--recording-identity",
            RECORDING_IDENTITY,
            "--successor-run",
            SUCCESSOR,
            "--result-json",
            str(receipt),
            "--activate",
            "--apply",
        ]
    )

    assert code == 1
    refused = json.loads(receipt.read_text())
    assert refused["status"] == "refused"
    assert "ineligible" in refused["refusal"]
    assert refused["selectors_before"]["latest"]["value"] == "detect_source"


def test_activation_ignores_stale_nested_family_consolidation(
    archive: Path, scratch: Path
) -> None:
    published = _publish(archive, scratch, SUCCESSOR, eligible=True)
    # A consolidated-view attr write embeds a nested consolidated block in
    # detect_runs/zarr.json (the historical store split-brain shape).
    zarr.open_group(str(archive), mode="a")["detect_runs"].attrs["note"] = "x"
    family_meta = json.loads((archive / "detect_runs" / "zarr.json").read_text())
    assert "consolidated_metadata" in family_meta

    result = _activate(archive, scratch, apply=True)

    assert result["status"] == "activated"
    _assert_active(archive, SUCCESSOR, published["successor"]["manifest_digest"])
