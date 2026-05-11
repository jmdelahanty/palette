"""Compare swim-bout runs across physical Zarr layouts.

This utility is intentionally resolver-based: both inputs are loaded through
``fisheye.analysis.swim_bout_io`` so the comparison validates the logical API
that downstream tools use, not a hard-coded physical tree shape.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import zarr

from fisheye.analysis.swim_bout_io import load_default_swim_bout_tables, load_swim_bout_tables
from fisheye.shared.json_safety import json_attr_safe


DEFAULT_ATOL = 1e-9
DEFAULT_SERIES_ATOL = 1e-6


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    kind: str
    detail: str | None = None
    max_abs_diff: float | None = None


@dataclass(frozen=True)
class RunObjectCounts:
    run_name: str
    zarr_json_count: int | None
    total_path_count: int | None


@dataclass(frozen=True)
class SwimBoutLayoutComparison:
    zarr_path: str
    reference_run: str
    candidate_run: str
    reference_signal: str
    candidate_signal: str
    reference_role: str
    candidate_role: str
    checks_total: int
    checks_failed: int
    checks: list[CheckResult]
    reference_object_counts: RunObjectCounts
    candidate_object_counts: RunObjectCounts

    @property
    def passed(self) -> bool:
        return self.checks_failed == 0


def compare_swim_bout_layouts(
    zarr_path: Path,
    *,
    reference_run: str,
    candidate_run: str,
    speed_level: str | None = None,
    atol: float = DEFAULT_ATOL,
    series_atol: float = DEFAULT_SERIES_ATOL,
) -> SwimBoutLayoutComparison:
    """Compare two swim-bout runs loaded through the logical resolver."""

    zarr_path = zarr_path.expanduser().resolve()
    root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    if speed_level:
        reference = load_swim_bout_tables(root, run_name=reference_run, speed_level=speed_level)
        candidate = load_swim_bout_tables(root, run_name=candidate_run, speed_level=speed_level)
    else:
        reference = load_default_swim_bout_tables(root, run_name=reference_run)
        candidate = load_default_swim_bout_tables(root, run_name=candidate_run)

    checks: list[CheckResult] = []
    _add_scalar_check(
        checks,
        "signal.speed_level",
        reference.signal.speed_level,
        candidate.signal.speed_level,
    )
    _add_scalar_check(checks, "signal.role", reference.signal.role, candidate.signal.role)
    _add_scalar_check(checks, "bouts.count", int(reference.bouts.size), int(candidate.bouts.size))

    _compare_structured_table(
        checks,
        "bouts",
        reference.bouts,
        candidate.bouts,
        exact_fields=(
            "bout_id",
            "start_frame",
            "end_frame",
            "core_start_frame",
            "core_end_frame",
            "duration_frames",
            "core_duration_frames",
            "n_valid_transitions",
            "n_invalid_transitions",
            "gap_censored",
            "core_start_time_interpolated_valid",
            "core_end_time_interpolated_valid",
        ),
        numeric_fields=(
            "duration_s",
            "elapsed_duration_s",
            "observed_duration_s",
            "core_duration_s",
            "path_length_mm",
            "path_length_px",
            "net_displacement_mm",
            "net_displacement_px",
            "mean_speed_mm_s",
            "peak_detection_signal_mm_s",
            "peak_physical_speed_mm_s",
            "valid_transition_fraction",
            "start_time_s",
            "end_time_s",
            "core_start_time_s",
            "core_end_time_s",
            "core_start_time_s_interpolated",
            "core_end_time_s_interpolated",
            "core_duration_s_interpolated",
        ),
        atol=atol,
    )
    _compare_structured_table(
        checks,
        "peak_events",
        reference.peak_events,
        candidate.peak_events,
        exact_fields=(
            "bout_id",
            "peak_index",
            "peak_frame",
            "left_base_index",
            "right_base_index",
            "left_base_frame",
            "right_base_frame",
            "boundary_mode",
            "shape_split_policy",
        ),
        numeric_fields=(
            "peak_time_s",
            "peak_signal_value_mm_s",
            "peak_prominence_mm_s",
            "peak_width_samples",
            "peak_width_s",
            "peak_width_height_mm_s",
            "left_ips",
            "right_ips",
            "left_width_frame_interpolated",
            "right_width_frame_interpolated",
            "left_base_signal_value_mm_s",
            "right_base_signal_value_mm_s",
        ),
        atol=atol,
    )
    _compare_structured_table(
        checks,
        "inter_bout_intervals",
        reference.inter_bout_intervals,
        candidate.inter_bout_intervals,
        exact_fields=(
            "prev_bout_id",
            "next_bout_id",
            "prev_end_frame",
            "next_start_frame",
            "interval_frames",
        ),
        numeric_fields=("prev_end_time_s", "next_start_time_s", "interval_s"),
        atol=atol,
    )
    _compare_series(checks, reference.series, candidate.series, "frame_indices", exact=True)
    _compare_series(
        checks,
        reference.series,
        candidate.series,
        "detection_signal_mm_s",
        exact=False,
        atol=series_atol,
    )

    failed = sum(1 for check in checks if not check.passed)
    return SwimBoutLayoutComparison(
        zarr_path=str(zarr_path),
        reference_run=reference_run,
        candidate_run=candidate_run,
        reference_signal=reference.signal.speed_level,
        candidate_signal=candidate.signal.speed_level,
        reference_role=reference.signal.role,
        candidate_role=candidate.signal.role,
        checks_total=len(checks),
        checks_failed=failed,
        checks=checks,
        reference_object_counts=_run_object_counts(zarr_path, reference_run),
        candidate_object_counts=_run_object_counts(zarr_path, candidate_run),
    )


def _compare_structured_table(
    checks: list[CheckResult],
    label: str,
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    exact_fields: Iterable[str],
    numeric_fields: Iterable[str],
    atol: float,
) -> None:
    _add_scalar_check(checks, f"{label}.count", int(reference.size), int(candidate.size))
    if reference.size != candidate.size:
        return
    for field in exact_fields:
        if _has_field(reference, field) and _has_field(candidate, field):
            _add_array_equal_check(checks, f"{label}.{field}", reference[field], candidate[field])
    for field in numeric_fields:
        if _has_field(reference, field) and _has_field(candidate, field):
            _add_array_close_check(checks, f"{label}.{field}", reference[field], candidate[field], atol=atol)


def _compare_series(
    checks: list[CheckResult],
    reference: Mapping[str, np.ndarray],
    candidate: Mapping[str, np.ndarray],
    name: str,
    *,
    exact: bool,
    atol: float = DEFAULT_ATOL,
) -> None:
    if name not in reference and name not in candidate:
        return
    if name not in reference or name not in candidate:
        checks.append(
            CheckResult(
                name=f"series.{name}",
                passed=False,
                kind="presence",
                detail=f"reference_has={name in reference}, candidate_has={name in candidate}",
            )
        )
        return
    if exact:
        _add_array_equal_check(checks, f"series.{name}", reference[name], candidate[name])
    else:
        _add_array_close_check(checks, f"series.{name}", reference[name], candidate[name], atol=atol)


def _has_field(records: np.ndarray, field: str) -> bool:
    return records.dtype.names is not None and field in records.dtype.names


def _add_scalar_check(checks: list[CheckResult], name: str, reference: Any, candidate: Any) -> None:
    passed = reference == candidate
    checks.append(
        CheckResult(
            name=name,
            passed=bool(passed),
            kind="scalar_equal",
            detail=None if passed else f"reference={reference!r}, candidate={candidate!r}",
        )
    )


def _add_array_equal_check(
    checks: list[CheckResult],
    name: str,
    reference: np.ndarray,
    candidate: np.ndarray,
) -> None:
    passed = bool(np.array_equal(reference, candidate))
    checks.append(
        CheckResult(
            name=name,
            passed=passed,
            kind="array_equal",
            detail=None if passed else _array_mismatch_detail(reference, candidate),
        )
    )


def _add_array_close_check(
    checks: list[CheckResult],
    name: str,
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    atol: float,
) -> None:
    ref = np.asarray(reference, dtype=np.float64)
    cand = np.asarray(candidate, dtype=np.float64)
    same_shape = ref.shape == cand.shape
    passed = bool(same_shape and np.allclose(ref, cand, rtol=0.0, atol=atol, equal_nan=True))
    max_abs_diff: float | None
    if same_shape and ref.size:
        finite_diff = np.abs(ref - cand)
        finite_values = finite_diff[np.isfinite(finite_diff)]
        if finite_values.size:
            max_abs_diff = float(np.max(finite_values))
        else:
            max_abs_diff = 0.0 if passed else None
    else:
        max_abs_diff = 0.0 if same_shape else None
    checks.append(
        CheckResult(
            name=name,
            passed=passed,
            kind="array_close",
            detail=None if passed else _array_mismatch_detail(ref, cand),
            max_abs_diff=max_abs_diff,
        )
    )


def _array_mismatch_detail(reference: np.ndarray, candidate: np.ndarray) -> str:
    return (
        f"reference_shape={reference.shape}, candidate_shape={candidate.shape}, "
        f"reference_head={np.asarray(reference)[:5].tolist()}, "
        f"candidate_head={np.asarray(candidate)[:5].tolist()}"
    )


def _run_object_counts(zarr_path: Path, run_name: str) -> RunObjectCounts:
    run_path = zarr_path / "analysis" / "swim_bout_runs" / run_name
    if not run_path.exists():
        return RunObjectCounts(run_name=run_name, zarr_json_count=None, total_path_count=None)
    return RunObjectCounts(
        run_name=run_name,
        zarr_json_count=sum(1 for _ in run_path.rglob("zarr.json")),
        total_path_count=sum(1 for _ in run_path.rglob("*")),
    )


_json_safe = json_attr_safe


def comparison_to_dict(comparison: SwimBoutLayoutComparison) -> dict[str, Any]:
    return _json_safe(asdict(comparison))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_path", type=Path, help="Palette analysis Zarr archive.")
    parser.add_argument("--reference-run", required=True, help="Reference swim-bout run name, usually v1.")
    parser.add_argument("--candidate-run", required=True, help="Candidate swim-bout run name, usually compact v2.")
    parser.add_argument("--speed-level", default=None, help="Optional speed level to compare instead of each run's default.")
    parser.add_argument("--atol", type=float, default=DEFAULT_ATOL, help="Absolute tolerance for table numeric fields.")
    parser.add_argument("--series-atol", type=float, default=DEFAULT_SERIES_ATOL, help="Absolute tolerance for dense signal series.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    parser.add_argument("--write-json", type=Path, help="Optional path to write comparison JSON.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    comparison = compare_swim_bout_layouts(
        args.zarr_path,
        reference_run=args.reference_run,
        candidate_run=args.candidate_run,
        speed_level=args.speed_level,
        atol=args.atol,
        series_atol=args.series_atol,
    )
    payload = comparison_to_dict(comparison)
    if args.write_json:
        args.write_json.parent.mkdir(parents=True, exist_ok=True)
        args.write_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        _print_human_summary(comparison)
    return 0 if comparison.passed else 1


def _print_human_summary(comparison: SwimBoutLayoutComparison) -> None:
    status = "PASS" if comparison.passed else "FAIL"
    print(f"{status}: {comparison.reference_run} -> {comparison.candidate_run}")
    print(
        f"  signals: {comparison.reference_signal}/{comparison.reference_role} "
        f"vs {comparison.candidate_signal}/{comparison.candidate_role}"
    )
    print(f"  checks: {comparison.checks_total - comparison.checks_failed}/{comparison.checks_total} passed")
    ref_count = comparison.reference_object_counts.zarr_json_count
    cand_count = comparison.candidate_object_counts.zarr_json_count
    print(f"  zarr.json files: reference={ref_count}, candidate={cand_count}")
    if ref_count and cand_count is not None:
        reduction = (1.0 - (cand_count / ref_count)) * 100.0
        print(f"  candidate reduction vs reference: {reduction:.1f}%")
    for check in comparison.checks:
        if not check.passed:
            print(f"  FAIL {check.name}: {check.detail}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
