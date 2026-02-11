from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from fisheye.diagnostics import benchmark_random_seek as brs


def test_percentile_interpolates_midpoint() -> None:
    values = [1.0, 2.0, 4.0, 8.0]
    p50 = brs._percentile(values, 0.5)
    assert p50 == 3.0


def test_build_random_positions_respects_range_and_count() -> None:
    positions = brs._build_random_positions(
        total_frames=1000,
        samples=50,
        max_frame_fraction=0.5,
        seed=123,
    )
    assert len(positions) == 50
    assert all(0 <= pos <= 499 for pos in positions)


def test_stats_empty_values() -> None:
    stats = brs._stats([])
    assert stats["count"] == 0
    assert stats["median"] is None
    assert stats["p95"] is None
