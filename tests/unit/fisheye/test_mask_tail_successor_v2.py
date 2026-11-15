"""Tail-successor format v2: referenced crop run and sharded review runs.

v1 successors copy the crop run and write chunk-only; v2 successors publish
four runs that bind the existing crop run and use the training/review shard
profile. Both must present identical rows, pixels, labels, and masks to every
maintained consumer.
"""

import json
import os
import shutil
from types import SimpleNamespace

import numpy as np
import pytest
import zarr

from fisheye.labeling.web_runtimes import (
    _get_keypoint_runtime,
    _get_subject_mask_runtime,
    _subject_mask_current_payload,
)
from fisheye.shared.recovered_training_review_contract import (
    REFERENCED_CROP_SUCCESSOR_POLICY,
    initial_contract_digest,
)
from fisheye.shared.zarr.training_review_run_storage import (
    REVIEW_STORAGE_PLAN_ATTR,
)
from fisheye.training.mask_tail_apply_refresh import (
    REFRESH_POLICY,
    regenerate_training_tail_version,
    validate_completed_tail_version,
)
from fisheye.training.recover_merged_subject_masks import validate_initial_payload
from fisheye.training.recovered_mask_review_payload import build_review_payload
from fisheye.tune import keypoint_review_backend as editor
from fisheye.tune import refined_subject_mask_review as mask_review

LABELS = ("subject_body", "eyes_union", "swim_bladder")


def _build_archive(path, *, native, rows=2):
    root = zarr.open_group(str(path), mode="w", use_consolidated=False)
    root.attrs.update(
        zarr_purpose="training",
        recording_id="rec",
        stage_selector_eligible=False,
        schema_id="palette.training.merged_pose_detect_recovery_source.v1",
    )
    yy, xx = np.mgrid[:128, :128]
    body = ((xx - 64) / 13) ** 2 + ((yy - 60) / 48) ** 2 <= 1
    swim = ((xx - 64) / 7) ** 2 + ((yy - 43) / 9) ** 2 <= 1
    eyes = (((xx - 58) ** 2 + (yy - 26) ** 2) <= 9) | (
        ((xx - 70) ** 2 + (yy - 26) ** 2) <= 9
    )
    masks = np.repeat(np.stack([body, eyes, swim])[None], rows, axis=0).astype(
        np.uint8
    )
    images = np.repeat((body * 100).astype(np.uint8)[None], rows, axis=0)
    images += np.arange(rows, dtype=np.uint8)[:, None, None]  # Distinct rows.
    arrays = {
        "roi_images": images,
        "masks_roi": masks,
        "head_keypoints_roi": np.array(
            [[[64, 43], [58, 26], [70, 26]]] * rows, np.float32
        ),
        "frame_indices": np.arange(rows) * 16 + 15,
        "source_merged_row": np.arange(rows) + 200,
        "source_pose_local_row": np.arange(rows) + 3,
        "source_bbox_norm_coords": np.linspace(0, 1, rows * 4).reshape(rows, 4),
        "detection_source": np.zeros(rows, np.int8),
        "target_valid_channels": np.ones((rows, 3), bool),
    }
    binding = {
        "mask_run": "original",
        "recording_id": "rec",
        "mask_run_attrs": {"label_schema_id": "subject_v1_union"},
    }
    if native:
        from fisheye.training.recover_merged_training_recording import _sha256_array

        arrays["source_training_crop_row_ids"] = np.arange(rows)[::-1] + 1
        arrays["source_keypoints_roi"] = np.concatenate(
            [
                arrays["head_keypoints_roi"],
                np.array([[[64, 13]]] * rows, np.float32),
            ],
            axis=1,
        )
        binding.update(
            source_kind="native_reviewed_training_masks_v1",
            keypoint_run="old_pose",
            source_array_sha256={
                "keypoints": _sha256_array(arrays["source_keypoints_roi"])
            },
            source_keypoint_labels=["swim_bladder", "eye_left", "eye_right", "snout_tip"],
        )
    result = build_review_payload(
        root, arrays, LABELS, binding, version="v1", native=native
    )
    session = editor.resolve_review_session(
        str(path),
        refined_run=result["paths"]["pose_edit"].split("/")[1],
        include_all=True,
    )
    for row in range(rows):
        points = np.asarray(session.kp_roi_arr[row]).copy()
        points[14:18] = [[74, 43], [80, 46], [54, 43], [48, 46]]
        points[4, 0] += 1  # A manual tail point that must be carried over.
        editor.save_roi_correction(session, position=row, points=points)
    mask = root[result["paths"]["mask_edit"]]
    mask.attrs["edit_revision"] = 1
    mask["masks_roi"][1, 0, 1, 1] = 1  # Fragmented body: one failure row.
    return result


def _refresh(path, initial, successor_format):
    return regenerate_training_tail_version(
        archive=path,
        refined_mask_run=initial["paths"]["mask_edit"].split("/")[1],
        refined_keypoint_run=initial["paths"]["pose_edit"].split("/")[1],
        apply_id="apply-one",
        expected_mask_revision=1,
        successor_format=successor_format,
    )


@pytest.fixture(params=[False, True], ids=["recovered", "native"])
def paired(tmp_path, request):
    """The same source archive, refreshed once as v1 and once as v2."""
    v1_path = tmp_path / "v1" / "training.zarr"
    v1_path.parent.mkdir()
    initial = _build_archive(v1_path, native=request.param)
    v2_path = tmp_path / "v2" / "training.zarr"
    shutil.copytree(v1_path, v2_path)
    v1 = _refresh(v1_path, initial, "v1")
    v2 = _refresh(v2_path, initial, "v2")
    return SimpleNamespace(
        initial=initial, v1_path=v1_path, v2_path=v2_path, v1=v1, v2=v2
    )


def _open(path):
    return zarr.open_group(str(path), mode="r", use_consolidated=False)


def _arrays(group, prefix=""):
    values = {}
    for name, array in group.arrays():
        values[prefix + name] = np.asarray(array[:])
    for name, child in group.groups():
        values.update(_arrays(child, f"{prefix}{name}/"))
    return values


def _files(path):
    return sum(len(files) for _, _, files in os.walk(path))


def test_v2_publishes_four_runs_that_reference_and_bind_the_crop(paired):
    initial_crop = paired.initial["paths"]["crop"]
    crop_name = initial_crop.split("/")[1]
    root = _open(paired.v2_path)
    assert set(paired.v2["paths"]) == {"mask", "mask_edit", "seed", "pose_edit"}
    assert sorted(root["crop_runs"].group_keys()) == [crop_name]
    assert paired.v2["source_crop_run"] == crop_name
    proof = paired.v2["source_bindings"]["mask_apply_refresh"]
    assert proof["policy"] == REFERENCED_CROP_SUCCESSOR_POLICY
    assert proof["source_crop_run"] == initial_crop
    assert proof["source_crop_contract_sha256"] == initial_contract_digest(
        root[initial_crop]
    )
    for path in paired.v2["paths"].values():
        run = root[path]
        assert run.attrs["source_crop_run"] == crop_name
        np.testing.assert_array_equal(
            run["source_crop_row_ids"][:], root[initial_crop]["source_crop_row_ids"][:]
        )
        assert validate_initial_payload(paired.v2_path / path)["valid"]
    v1_proof = paired.v1["source_bindings"]["mask_apply_refresh"]
    assert v1_proof["policy"] == REFRESH_POLICY
    assert "source_crop_contract_sha256" not in v1_proof
    # The proof grammar is versioned with the format.
    assert v1_proof["schema_id"] == "palette.training.mask_apply_tail_successor.v1"
    assert proof["schema_id"] == "palette.training.mask_apply_tail_successor.v2"
    assert paired.v2["schema_id"] == proof["schema_id"]
    mismatched = {**paired.v2["source_bindings"], "mask_apply_refresh": {**proof, "schema_id": v1_proof["schema_id"]}}
    with pytest.raises(ValueError, match="source proof"):
        validate_completed_tail_version(
            archive=paired.v2_path, version=paired.v2["version"], source_bindings=mismatched,
        )
    assert paired.v1["version"] != paired.v2["version"]
    # Completed-effect validation accepts both formats with their own paths.
    for path, result in ((paired.v1_path, paired.v1), (paired.v2_path, paired.v2)):
        assert validate_completed_tail_version(
            archive=path, version=result["version"],
            source_bindings=result["source_bindings"],
        ) == result["paths"]
    # Successor tasks point every consumer at the referenced crop.
    for task in paired.v2["tasks"]:
        assert task["scope"]["crop_run"] == crop_name
    for task in paired.v1["tasks"]:
        assert task["scope"]["crop_run"] == paired.v1["paths"]["crop"].split("/")[1]


def test_v1_and_v2_successors_present_identical_samples(paired):
    v1_root, v2_root = _open(paired.v1_path), _open(paired.v2_path)
    for key, v2_path in paired.v2["paths"].items():
        v1_values = _arrays(v1_root[paired.v1["paths"][key]])
        v2_values = _arrays(v2_root[v2_path])
        assert v1_values.keys() == v2_values.keys(), key
        for name, values in v1_values.items():
            np.testing.assert_array_equal(v2_values[name], values, err_msg=f"{key}/{name}")
    # The pixels a v2 consumer resolves are the pixels v1 copied.
    v1_crop = v1_root[paired.v1["paths"]["crop"]]
    v2_crop = v2_root[f"crop_runs/{paired.v2['source_crop_run']}"]
    for name in ("roi_images", "source_bbox_norm_coords", "frame_indices"):
        np.testing.assert_array_equal(v2_crop[name][:], v1_crop[name][:])


def _task(result, kind):
    return next(task for task in result["tasks"] if task["workflow_kind"] == kind)


def test_v1_and_v2_review_sessions_are_identical_through_real_builders(paired):
    state = SimpleNamespace(keypoint_sessions={}, subject_mask_sessions={})
    sessions = {}
    for label, result in (("v1", paired.v1), ("v2", paired.v2)):
        task = _task(result, "keypoints")
        runtime = _get_keypoint_runtime(
            state,
            {"session_id": f"kp-{label}", "workflow_kind": "keypoints",
             "task_id": task["task_id"], "scope": task["scope"]},
        )
        sessions[label] = runtime.review_session
    one, two = sessions["v1"], sessions["v2"]
    assert one.recovered_roi_only and two.recovered_roi_only
    np.testing.assert_array_equal(two.roi_images[:], one.roi_images[:])
    np.testing.assert_array_equal(two.kp_roi_arr, one.kp_roi_arr)
    np.testing.assert_array_equal(two.frame_indices, one.frame_indices)
    np.testing.assert_array_equal(two.failures, one.failures)
    payloads = {}
    for label, result in (("v1", paired.v1), ("v2", paired.v2)):
        task = _task(result, "subject_mask_component")
        runtime = _get_subject_mask_runtime(
            state,
            {"session_id": f"mask-{label}", "workflow_kind": "subject_mask_component",
             "task_id": task["task_id"], "scope": task["scope"]},
        )
        rows = []
        for position in range(len(runtime.roi_indices)):
            runtime.position = position
            payload = _subject_mask_current_payload(runtime)
            rows.append(
                {key: payload[key] for key in (
                    "roi_idx", "frame_idx", "roi_image", "mask", "mask_area_px",
                    "frame_index_domain", "component_name",
                )}
            )
        payloads[label] = rows
    assert json.dumps(payloads["v2"], sort_keys=True, default=str) == json.dumps(
        payloads["v1"], sort_keys=True, default=str
    )


def _codecs(path):
    return json.loads((path / "zarr.json").read_text())["codecs"]


def test_v2_runs_are_sharded_by_the_review_profile_and_editable(paired):
    root = zarr.open_group(str(paired.v2_path), mode="a", use_consolidated=False)
    seed_path = paired.v2_path / paired.v2["paths"]["seed"]
    (codec,) = _codecs(seed_path / "keypoints_roi")
    rows = root[paired.v2["paths"]["seed"]]["keypoints_roi"].shape[0]
    assert codec["name"] == "sharding_indexed"
    assert codec["configuration"]["chunk_shape"] == [1, 19, 2]  # Unchanged inner.
    json_meta = json.loads((seed_path / "keypoints_roi" / "zarr.json").read_text())
    assert json_meta["chunk_grid"]["configuration"]["chunk_shape"] == [rows, 19, 2]
    v1_seed = paired.v1_path / paired.v1["paths"]["seed"] / "keypoints_roi"
    assert _codecs(v1_seed)[0]["name"] != "sharding_indexed"
    for key, path in paired.v2["paths"].items():
        plan = root[path].attrs[REVIEW_STORAGE_PLAN_ATTR]
        assert plan["storage_profile"]["profile_id"] == "training_review_run_v1"
        expected = "random_update" if key in {"pose_edit", "mask_edit"} else "immutable"
        assert plan["write_mode"] == expected
        assert root[path].attrs["physical_storage_layout"]["exact_decoded_validation"]
    edit_plan = root[paired.v2["paths"]["pose_edit"]].attrs[REVIEW_STORAGE_PLAN_ATTR]
    assert edit_plan["arrays"]["keypoints_roi"]["write_ownership"] == (
        "serialized_whole_shard_rewrite_single_writer"
    )
    mask_codec = _codecs(paired.v2_path / paired.v2["paths"]["mask_edit"] / "masks_roi")
    assert mask_codec[0]["name"] == "sharding_indexed"

    # Browser-style keypoint save into the sharded editable run.
    session = editor.resolve_review_session(
        str(paired.v2_path),
        refined_run=paired.v2["paths"]["pose_edit"].split("/")[1],
        crop_run=paired.v2["source_crop_run"],
        include_all=True,
    )
    points = np.asarray(session.kp_roi_arr[0]).copy()
    points[15, 0] += 2
    editor.save_roi_correction(session, position=0, points=points)
    reread = _open(paired.v2_path)[paired.v2["paths"]["pose_edit"]]
    np.testing.assert_array_equal(reread["keypoints_roi"][0], points)
    # The other row in the same shard is untouched.
    np.testing.assert_array_equal(
        reread["keypoints_roi"][1],
        _open(paired.v1_path)[paired.v1["paths"]["pose_edit"]]["keypoints_roi"][1],
    )

    # Browser-style dense mask save into the sharded editable mask run.
    source, refined = mask_review.prepare_refined_subject_run(
        root,
        refined_run=paired.v2["paths"]["mask_edit"].split("/")[1],
        components=["subject_body"],
    )
    edited = np.asarray(refined.group["masks_roi"][1], dtype=np.uint8)
    other = np.asarray(refined.group["masks_roi"][0], dtype=np.uint8)
    edited[0, 1, 1] = 0
    mask_review.save_refined_subject_roi(
        source=source, refined=refined, roi_idx=1, edited_masks=edited
    )
    reread = _open(paired.v2_path)[paired.v2["paths"]["mask_edit"]]
    np.testing.assert_array_equal(reread["masks_roi"][1], edited)
    np.testing.assert_array_equal(reread["masks_roi"][0], other)


def test_v2_refuses_a_changed_referenced_crop(paired, tmp_path):
    root = zarr.open_group(str(paired.v2_path), mode="a", use_consolidated=False)
    crop = root[f"crop_runs/{paired.v2['source_crop_run']}"]
    pixels = np.asarray(crop["roi_images"][0]).copy()
    pixels[0, 0] += 1
    crop["roi_images"][0] = pixels
    with pytest.raises(ValueError, match="crop"):
        validate_completed_tail_version(
            archive=paired.v2_path, version=paired.v2["version"],
            source_bindings=paired.v2["source_bindings"],
        )


def test_v1_publication_in_flight_resumes_as_v1(tmp_path, monkeypatch):
    """An Apply whose v1 publication was interrupted keeps its v1 version."""
    from fisheye.training import recover_merged_subject_masks as producer

    path = tmp_path / "training.zarr"
    initial = _build_archive(path, native=False)
    original = producer.atomic_publish_run_group
    calls = []

    def interrupted(spec, **kwargs):
        calls.append(spec.target_run_path)
        if len(calls) == 2:
            raise OSError("interrupted publication")
        return original(spec, **kwargs)

    monkeypatch.setattr(producer, "atomic_publish_run_group", interrupted)
    with pytest.raises(OSError):
        _refresh(path, initial, "v1")
    monkeypatch.setattr(producer, "atomic_publish_run_group", original)
    resumed = _refresh(path, initial, None)
    assert "crop" in resumed["paths"]
    assert resumed["source_bindings"]["mask_apply_refresh"]["policy"] == REFRESH_POLICY


def test_representative_file_counts(tmp_path, capsys):
    """Files per successor on a 64-row fixture (reported, loosely bounded)."""
    counts = {}
    for successor_format in ("v1", "v2"):
        path = tmp_path / successor_format / "training.zarr"
        path.parent.mkdir()
        initial = _build_archive(path, native=False, rows=64)
        result = _refresh(path, initial, successor_format)
        counts[successor_format] = {
            key: _files(path / run) for key, run in result["paths"].items()
        }
    with capsys.disabled():
        print(f"\nsuccessor files (64 rows): {json.dumps(counts, sort_keys=True)}")
    v1_total, v2_total = (sum(counts[k].values()) for k in ("v1", "v2"))
    assert "crop" not in counts["v2"]
    assert counts["v2"]["seed"] < counts["v1"]["seed"] - 60
    assert v2_total < v1_total / 2
