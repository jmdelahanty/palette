from __future__ import annotations

import copy

import numpy as np
import pytest

from fisheye.analysis import stimulus_response as response_module
from fisheye.analysis.stimulus_response import (
    ConcentricStepData,
    GratingStepData,
    LoomStepData,
    LoomTrial,
    ProtocolStep,
    _write_stimulus_response_compact_v3,
)
from fisheye.analysis.stimulus_response_concentric_omr import (
    ConcentricRadialOMRStepData,
)
from fisheye.analysis.stimulus_response_omr import OMRStepData
from fisheye.analysis.stimulus_response_io import resolve_stimulus_response_v3_tables
from fisheye.shared.zarr.stimulus_response_schema import (
    KNOWN_BUNDLES,
    STIMULUS_RESPONSE_LAYOUT,
    STIMULUS_RESPONSE_SCHEMA_ID,
    STIMULUS_RESPONSE_SCHEMA_VERSION,
    expected_table_names,
    stimulus_response_array_declarations,
    table_contract,
    validate_mapping,
    validate_stimulus_response_v3_run,
)
from fisheye.shared.zarr.storage_intent import AccessPattern


class _FakeArray:
    def __init__(self, data: np.ndarray) -> None:
        self._data = np.asarray(data)
        self.dtype = self._data.dtype
        self.shape = self._data.shape
        self.attrs: dict[str, object] = {}

    def __getitem__(self, index):
        return self._data[index]


class _FakeGroup:
    def __init__(self) -> None:
        self.attrs: dict[str, object] = {}
        self._children: dict[str, _FakeGroup | _FakeArray] = {}

    def create_group(self, name: str) -> "_FakeGroup":
        group = _FakeGroup()
        self._children[name] = group
        return group

    def __contains__(self, name: str) -> bool:
        try:
            self[name]
        except KeyError:
            return False
        return True

    def __getitem__(self, name: str):
        node: _FakeGroup | _FakeArray = self
        for component in str(name).split("/"):
            if not isinstance(node, _FakeGroup) or component not in node._children:
                raise KeyError(name)
            node = node._children[component]
        return node

    def __delitem__(self, name: str) -> None:
        del self._children[name]

    def get(self, name: str, default=None):
        try:
            return self[name]
        except KeyError:
            return default

    def group_keys(self) -> tuple[str, ...]:
        return tuple(
            name
            for name, child in self._children.items()
            if isinstance(child, _FakeGroup)
        )

    def array_keys(self) -> tuple[str, ...]:
        return tuple(
            name
            for name, child in self._children.items()
            if isinstance(child, _FakeArray)
        )


@pytest.fixture(autouse=True)
def _in_memory_store(monkeypatch):
    def store_array(group: _FakeGroup, name: str, data: np.ndarray) -> _FakeArray:
        array = _FakeArray(np.asarray(data))
        group._children[name] = array
        return array

    monkeypatch.setattr(response_module, "store_array", store_array)


def _global(fish_ids: tuple[int, ...] = (4, 9)) -> dict[str, np.ndarray]:
    n = len(fish_ids)
    return {
        "fish_id": np.asarray(fish_ids, dtype=np.int32),
        "total_distance_mm": np.arange(n, dtype=np.float32),
        "mean_speed_mm_s": np.ones(n, dtype=np.float32),
        "total_active_s": np.ones(n, dtype=np.float32),
        "fraction_moving": np.ones(n, dtype=np.float32),
    }


def _step_metrics(fish_ids: tuple[int, ...] = (4, 9)) -> dict[str, np.ndarray]:
    n = len(fish_ids)
    return {
        "fish_id": np.asarray(fish_ids, dtype=np.int32),
        "total_distance_mm": np.ones(n, dtype=np.float32),
        "mean_speed_mm_s": np.ones(n, dtype=np.float32),
        "median_speed_mm_s": np.ones(n, dtype=np.float32),
        "max_speed_mm_s": np.ones(n, dtype=np.float32),
        "fraction_moving": np.ones(n, dtype=np.float32),
        "coverage": np.ones(n, dtype=np.float32),
    }


def _loom(fish_ids: tuple[int, ...] = (4, 9)) -> LoomStepData:
    shape = (len(fish_ids), 2)
    return LoomStepData(
        trials=[LoomTrial(0, 10, 12, 0.2), LoomTrial(1, 15, 17, 0.2)],
        per_frame={},
        per_trial_per_fish={
            "escaped": np.asarray([[True, False], [False, True]], dtype=bool),
            "escape_latency_s": np.ones(shape, dtype=np.float32),
            "escape_latency_frames": np.ones(shape, dtype=np.int32),
            "peak_escape_speed_mm_s": np.ones(shape, dtype=np.float32),
            "distance_at_escape_mm": np.ones(shape, dtype=np.float32),
            "visual_angle_at_escape_deg": np.ones(shape, dtype=np.float32),
            "escape_heading_deg": np.ones(shape, dtype=np.float32),
        },
        per_fish={
            "n_escape_responses": np.ones(len(fish_ids), dtype=np.int32),
            "escape_probability": np.ones(len(fish_ids), dtype=np.float32),
            "mean_escape_latency_s": np.ones(len(fish_ids), dtype=np.float32),
            "median_escape_latency_s": np.ones(len(fish_ids), dtype=np.float32),
            "mean_peak_escape_speed_mm_s": np.ones(len(fish_ids), dtype=np.float32),
            "mean_distance_at_escape_mm": np.ones(len(fish_ids), dtype=np.float32),
            "mean_visual_angle_at_escape_deg": np.ones(len(fish_ids), dtype=np.float32),
            "habituation_index": np.ones(len(fish_ids), dtype=np.float32),
        },
        time_series={},
    )


def _exact_table_mapping(
    table_name: str,
    *,
    bundles: tuple[str, ...],
    excluded: tuple[str, ...] = ("step_index",),
    fish_ids: tuple[int, ...] = (4, 9),
) -> dict[str, np.ndarray]:
    table = table_contract(table_name, bundles=bundles)
    n_rows = len(fish_ids)
    values: dict[str, np.ndarray] = {}
    for name, field in table.fields:
        if name in excluded:
            continue
        if name == "fish_id":
            values[name] = np.asarray(fish_ids, dtype=field.logical_dtype)
        elif name.endswith("_frame"):
            values[name] = np.arange(n_rows, dtype=field.logical_dtype)
        else:
            values[name] = np.zeros(n_rows, dtype=field.logical_dtype)
    return values


def _write(*, step_name: str = "loom", loom: LoomStepData | None = None) -> _FakeGroup:
    run = _FakeGroup()
    run.attrs.update(
        {
            "schema_id": STIMULUS_RESPONSE_SCHEMA_ID,
            "schema_version": STIMULUS_RESPONSE_SCHEMA_VERSION,
            "layout": STIMULUS_RESPONSE_LAYOUT,
        }
    )
    step = ProtocolStep(0, step_name, "LOOMING_DOT", 7, 0, 20, 2.0)
    _write_stimulus_response_compact_v3(
        run,
        global_metrics=_global(),
        steps=[step],
        step_metrics=[_step_metrics()],
        frame_annotations=None,
        step_bout_metrics=None,
        step_grating_data=None,
        step_concentric_data=None,
        step_loom_data={0: loom} if loom is not None else None,
        global_omr_metrics=None,
    )
    return run


def test_declarations_are_exact_and_use_shared_declaration_type() -> None:
    declarations = stimulus_response_array_declarations(bundles=("looming",))
    paths = {item.path for item in declarations}
    assert "global_per_fish/fish_id" in paths
    assert "looming_per_trial_per_fish/escaped" in paths
    assert "looming_per_trial_per_fish/fish_id" in paths
    assert "looming_per_trial_per_fish/trial_index" in paths
    assert len(paths) == len(declarations)


def test_frame_annotations_are_windowed_while_summaries_are_eager() -> None:
    declarations = {
        declaration.path: declaration
        for declaration in stimulus_response_array_declarations(
            bundles=("frame_annotations",)
        )
    }
    assert (
        declarations["frame_annotations/step_index"].access_pattern
        is AccessPattern.WINDOWED
    )
    assert (
        declarations["global_per_fish/mean_speed_mm_s"].access_pattern
        is AccessPattern.EAGER
    )


def test_complete_standard_profile_has_frozen_310_array_surface() -> None:
    declarations = stimulus_response_array_declarations(bundles=sorted(KNOWN_BUNDLES))
    assert len(declarations) == 310
    assert len({declaration.path for declaration in declarations}) == 310


def test_v3_remains_opt_in_without_changing_catalog_facing_producer_default() -> None:
    assert response_module.STIMULUS_RESPONSE_LAYOUT_DEFAULT == "compact_tabular_v2"
    assert response_module.STIMULUS_RESPONSE_SCHEMA_VERSION == 2


def test_optional_bundle_table_sets_are_closed_and_all_or_none() -> None:
    names = set(expected_table_names(("moving_grating_omr",)))
    assert {
        "global_omr_per_fish",
        "moving_grating_omr_per_fish",
        "moving_grating_omr_per_bout",
        "moving_grating_omr_windows",
        "moving_grating_omr_early_windows",
    } <= names
    with pytest.raises(ValueError, match="Unknown stimulus-response v3 bundles"):
        expected_table_names(("moving_grating_omr_partial",))


def test_writer_flattens_loom_trial_fish_and_preserves_identity() -> None:
    run = _write(loom=_loom())
    assert validate_stimulus_response_v3_run(run) == ()
    assert run["looming_per_trial_per_fish/fish_id"][:].tolist() == [4, 4, 9, 9]
    assert run["looming_per_trial_per_fish/trial_index"][:].tolist() == [0, 1, 0, 1]
    assert run["looming_per_trial_per_fish/escaped"][:].tolist() == [
        True,
        False,
        False,
        True,
    ]
    assert run["looming_per_trial_per_fish/fish_id"].dtype == np.dtype("int32")
    assert run["looming_per_trial_per_fish/escaped"].dtype == np.dtype("bool")


def test_writer_materializes_complete_310_array_surface() -> None:
    bundles = tuple(sorted(KNOWN_BUNDLES))
    fish_ids = (4, 9)
    moving_omr = OMRStepData(
        per_fish=_exact_table_mapping("moving_grating_omr_per_fish", bundles=bundles),
        per_bout=_exact_table_mapping("moving_grating_omr_per_bout", bundles=bundles),
        windows=_exact_table_mapping("moving_grating_omr_windows", bundles=bundles),
        early_windows=_exact_table_mapping(
            "moving_grating_omr_early_windows", bundles=bundles
        ),
        attrs={},
    )
    radial_omr = ConcentricRadialOMRStepData(
        per_frame={},
        per_fish=_exact_table_mapping(
            "concentric_radial_omr_per_fish", bundles=bundles
        ),
        per_bout=_exact_table_mapping(
            "concentric_radial_omr_per_bout", bundles=bundles
        ),
        windows=_exact_table_mapping("concentric_radial_omr_windows", bundles=bundles),
        early_windows=_exact_table_mapping(
            "concentric_radial_omr_early_windows", bundles=bundles
        ),
        attrs={},
    )
    steps = [
        ProtocolStep(0, "moving", "MOVING_GRATING", 1, 0, 10, 1.0),
        ProtocolStep(1, "concentric", "CONCENTRIC_GRATING", 2, 10, 20, 1.0),
        ProtocolStep(2, "loom", "LOOMING_DOT", 3, 20, 40, 2.0),
    ]
    empty_bout = _exact_table_mapping("step_per_bout", bundles=bundles)
    run = _FakeGroup()
    run.attrs.update(
        {
            "schema_id": STIMULUS_RESPONSE_SCHEMA_ID,
            "schema_version": STIMULUS_RESPONSE_SCHEMA_VERSION,
            "layout": STIMULUS_RESPONSE_LAYOUT,
        }
    )
    _write_stimulus_response_compact_v3(
        run,
        global_metrics=_global(fish_ids),
        steps=steps,
        step_metrics=[_step_metrics(fish_ids) for _step in steps],
        frame_annotations={
            "step_index": np.asarray([0, 0, 1, 1, 2], dtype=np.int32),
            "stimulus_mode_id": np.asarray([1, 1, 2, 2, 3], dtype=np.int32),
        },
        step_bout_metrics=[
            (
                {
                    "num_bouts": np.zeros(2, dtype=np.int32),
                    "mean_bout_duration_s": np.zeros(2, dtype=np.float32),
                    "mean_interbout_interval_s": np.zeros(2, dtype=np.float32),
                },
                empty_bout,
            )
            for _step in steps
        ],
        step_grating_data={
            0: GratingStepData(
                per_frame={},
                per_fish=_exact_table_mapping("grating_per_fish", bundles=bundles),
                time_series={},
                omr=moving_omr,
            )
        },
        step_concentric_data={
            1: ConcentricStepData(
                per_frame={},
                per_fish=_exact_table_mapping("concentric_per_fish", bundles=bundles),
                time_series={},
                radial_omr=radial_omr,
            )
        },
        step_loom_data={2: _loom(fish_ids)},
        global_omr_metrics=_exact_table_mapping(
            "global_omr_per_fish", bundles=bundles, excluded=()
        ),
    )

    assert validate_stimulus_response_v3_run(run) == ()
    paths = {
        f"{table_name}/{field_name}"
        for table_name in run.group_keys()
        for field_name in run[table_name].array_keys()
    }
    assert len(paths) == 310


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda value: value.update({"surprise": np.ones(2, dtype=np.float32)}),
            "unexpected",
        ),
        (
            lambda value: value.__setitem__("coverage", np.ones(2, dtype=np.float64)),
            "dtype",
        ),
        (
            lambda value: value.__setitem__(
                "coverage", np.ones((2, 1), dtype=np.float32)
            ),
            "one-dimensional",
        ),
        (
            lambda value: value.__setitem__("coverage", np.ones(1, dtype=np.float32)),
            "inconsistent",
        ),
    ],
)
def test_mapping_validation_fails_closed(mutator, message: str) -> None:
    mapping = _step_metrics()
    mutator(mapping)
    with pytest.raises(ValueError, match=message):
        validate_mapping(
            "step_per_fish",
            mapping,
            bundles=(),
            excluded_fields=("step_index",),
        )


def test_writer_rejects_fixed_width_text_overflow() -> None:
    with pytest.raises(ValueError, match="maximum is 127"):
        _write(step_name="x" * 129)


def test_run_validator_rejects_recomputed_manifest_tampering() -> None:
    run = _write()
    manifest = copy.deepcopy(run.attrs["stimulus_response_array_schema"])
    manifest["arrays"][0]["logical_contract"]["description"] = "tampered"
    run.attrs["stimulus_response_array_schema"] = manifest
    assert (
        "stimulus-response array manifest is not exact"
        in validate_stimulus_response_v3_run(run)
    )


def test_strict_v3_reader_rejects_legacy_while_compatibility_reader_remains() -> None:
    run = _FakeGroup()
    run.attrs["layout"] = "compact_tabular_v2"
    with pytest.raises(ValueError, match="exact schema and layout"):
        resolve_stimulus_response_v3_tables(run)
