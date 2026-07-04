from __future__ import annotations

import re

from fisheye.shared.batch_logging import (
    make_run_id,
    utc_now,
    utc_now_compact,
    utc_now_date,
    utc_now_z,
)


def test_batch_logging_timestamp_formats_are_named() -> None:
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T.*\+00:00", utc_now())
    assert re.fullmatch(r"\d{8}T\d{6}Z", utc_now_compact())
    assert re.fullmatch(r"\d{8}", utc_now_date())
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", utc_now_z())


def test_make_run_id_uses_compact_timestamp_plus_pid() -> None:
    assert re.fullmatch(r"\d{8}T\d{6}Z_\d+", make_run_id())
