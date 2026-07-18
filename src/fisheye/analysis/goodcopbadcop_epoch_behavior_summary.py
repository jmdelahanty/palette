"""Compatibility imports for :mod:`fisheye.analysis.chaser_epoch_behavior_summary`."""

# ruff: noqa: F401,F403

from fisheye.analysis.chaser_epoch_behavior_summary import *  # noqa: F403
from fisheye.analysis.chaser_epoch_behavior_summary import (
    ChaserEpochBehaviorSummaryResult as GoodCopBadCopEpochBehaviorSummaryResult,
    build_chaser_epoch_behavior_summary_result as build_goodcopbadcop_epoch_behavior_summary_result,
    write_chaser_epoch_behavior_summary_component as write_goodcopbadcop_epoch_behavior_summary_component,
)
