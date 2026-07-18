"""Compatibility imports for :mod:`fisheye.analysis.chaser_escape_freeze_summary`."""

# ruff: noqa: F401,F403

from fisheye.analysis.chaser_escape_freeze_summary import *  # noqa: F403
from fisheye.analysis.chaser_escape_freeze_summary import (
    ChaserEscapeFreezeSummaryResult as EscapeFreezeCanaryResult,
    _assert_chaser_trace_moves,
    _classify_escape_attempt_by_path,
    _contiguous_true_segments,
    _controller_trial_segments,
    _metric_dtype,
    _select_trial_trigger,
    _trajectory_dtype,
    _trial_dtype,
    _trigger_radius_from_chaser_states,
    _trigger_radius_from_protocol_json,
    build_chaser_escape_freeze_summary_result as build_escape_freeze_canary_result,
    write_chaser_escape_freeze_summary_component as write_escape_freeze_canary_component,
)
