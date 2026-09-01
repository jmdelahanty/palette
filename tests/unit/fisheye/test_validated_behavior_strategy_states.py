"""Synthetic-only unit tests for the strategy-state pipeline."""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from fisheye.group_statistics.validated_behavior_strategy_states import (
    DECODER_FEATURE_COLUMNS,
    EPOCH_ROLES,
    FEATURE_COLUMNS,
    ClusterResult,
    StrategyStatesConfig,
    StrategyStatesInputError,
    assemble_feature_matrix,
    build_window_features,
    decompose_displacements,
    default_epoch_windows,
    g_statistic,
    load_ibi_cell_features,
    load_twin_excess_features,
    occupancy_entropy,
    parse_arena,
    punctuated_axis,
    run_loro_decoder,
    stratified_permutation_pvalue,
)


def _feature_sources(recordings: list[str]):
    rows = [
        {"recording_id": rec, "epoch_role": epoch}
        for rec in recordings
        for epoch in EPOCH_ROLES
    ]
    grid = pl.DataFrame(rows)
    rng = np.random.default_rng(0)
    n = grid.height
    twin = grid.with_columns(
        pl.Series("nz_excess", rng.normal(size=n)),
        pl.Series("dist_excess", rng.normal(size=n)),
    )
    wall = grid.with_columns(
        pl.Series("fish_wall_distance_p50_mm", rng.uniform(1, 10, size=n))
    )
    entropy = grid.with_columns(
        pl.Series("occupancy_entropy", rng.uniform(1, 3, size=n))
    )
    bout = grid.with_columns(
        pl.Series("bout_rate_per_min", rng.uniform(10, 60, size=n)),
        pl.Series(
            "mean_abs_bout_net_heading_change_deg", rng.uniform(5, 60, size=n)
        ),
    )
    ibi = grid.with_columns(pl.Series("ibi_gt2s", rng.uniform(0, 0.3, size=n)))
    return twin, wall, entropy, bout, ibi


class TestFeatureAssembly:
    def test_missing_twin_row_fails_closed(self):
        recs = [f"r{i}_arena_1_x" for i in range(4)]
        twin, wall, entropy, bout, ibi = _feature_sources(recs)
        twin_dropped = twin.filter(
            ~(
                (pl.col("recording_id") == recs[0])
                & (pl.col("epoch_role") == "chaser_post")
            )
        )
        with pytest.raises(StrategyStatesInputError, match="nz_excess"):
            assemble_feature_matrix(
                twin=twin_dropped, wall=wall, entropy=entropy, bout=bout, ibi=ibi
            )

    @pytest.mark.parametrize("source", ["wall", "entropy", "bout"])
    def test_missing_required_source_row_fails_closed(self, source):
        recs = [f"r{i}_arena_1_x" for i in range(4)]
        twin, wall, entropy, bout, ibi = _feature_sources(recs)
        sources = {"wall": wall, "entropy": entropy, "bout": bout}
        sources[source] = sources[source].filter(
            ~(
                (pl.col("recording_id") == recs[1])
                & (pl.col("epoch_role") == "chaser_pre")
            )
        )
        with pytest.raises(StrategyStatesInputError):
            assemble_feature_matrix(
                twin=twin,
                wall=sources["wall"],
                entropy=sources["entropy"],
                bout=sources["bout"],
                ibi=ibi,
            )

    def test_missing_ibi_is_median_imputed_with_flag(self):
        recs = [f"r{i}_arena_1_x" for i in range(4)]
        twin, wall, entropy, bout, ibi = _feature_sources(recs)
        ibi_dropped = ibi.filter(
            ~(
                (pl.col("recording_id") == recs[2])
                & (pl.col("epoch_role") == "chaser_post")
            )
        )
        result = assemble_feature_matrix(
            twin=twin, wall=wall, entropy=entropy, bout=bout, ibi=ibi_dropped
        )
        assert result.height == len(recs) * 2
        imputed = result.filter(
            (pl.col("recording_id") == recs[2])
            & (pl.col("epoch_role") == "chaser_post")
        )
        assert imputed["ibi_gt2s_imputed"].to_list() == [True]
        expected_median = float(ibi_dropped["ibi_gt2s"].median())
        assert imputed["ibi_gt2s"][0] == pytest.approx(expected_median)
        assert result["ibi_gt2s_imputed"].sum() == 1

    def test_feature_columns_complete(self):
        recs = [f"r{i}_arena_2_x" for i in range(3)]
        twin, wall, entropy, bout, ibi = _feature_sources(recs)
        result = assemble_feature_matrix(
            twin=twin, wall=wall, entropy=entropy, bout=bout, ibi=ibi
        )
        for column in FEATURE_COLUMNS:
            assert column in result.columns
        assert result["ibi_gt2s_imputed"].sum() == 0


class TestInputLoaders:
    def test_twin_loader_rejects_missing_columns(self, tmp_path):
        path = tmp_path / "twin.parquet"
        pl.DataFrame({"recording_id": ["a"], "provider_role": ["keypoint"]}).write_parquet(path)
        with pytest.raises(StrategyStatesInputError, match="missing required columns"):
            load_twin_excess_features(path)

    def test_ibi_loader_rejects_missing_columns(self, tmp_path):
        path = tmp_path / "ibi.parquet"
        pl.DataFrame({"recording_id": ["a"]}).write_parquet(path)
        with pytest.raises(StrategyStatesInputError, match="missing required columns"):
            load_ibi_cell_features(path)

    def test_twin_loader_filters_and_renames(self, tmp_path):
        path = tmp_path / "twin.parquet"
        pl.DataFrame(
            {
                "recording_id": ["a", "a", "a"],
                "provider_role": ["keypoint", "detection", "keypoint"],
                "epoch_role": ["chaser_pre", "chaser_pre", "chaser_training"],
                "behavior_role": ["aggressive", "aggressive", "aggressive"],
                "near_zone_fraction_valid_excess": [0.1, 0.2, 0.3],
                "distance_p50_mm_excess": [1.0, 2.0, 3.0],
            }
        ).write_parquet(path)
        out = load_twin_excess_features(path)
        assert out.height == 1
        assert out["nz_excess"][0] == pytest.approx(0.1)


class TestEntropy:
    def test_uniform_entropy(self):
        assert occupancy_entropy([0.25] * 4) == pytest.approx(math.log(4))

    def test_unnormalized_fractions_are_normalized(self):
        assert occupancy_entropy([2.0, 2.0]) == pytest.approx(math.log(2))

    def test_concentrated_entropy_zero(self):
        assert occupancy_entropy([0.7]) == pytest.approx(0.0)

    def test_empty_fails(self):
        with pytest.raises(StrategyStatesInputError):
            occupancy_entropy([0.0, -1.0])


class TestGTest:
    def test_independent_table_g_near_zero(self):
        table = np.outer([10.0, 20.0], [5.0, 15.0]) / 50.0 * 50.0 / 50.0
        table = np.outer([30.0, 20.0], [25.0, 25.0]) / 50.0
        assert g_statistic(table) == pytest.approx(0.0, abs=1e-9)

    def test_dependent_table_g_positive(self):
        table = np.array([[20.0, 0.0], [0.0, 20.0]])
        assert g_statistic(table) > 20.0

    def test_constructed_transition_table_permutation(self):
        rng = np.random.default_rng(1)
        pre = np.array([0, 1] * 20)
        post = pre.copy()  # perfect dependence
        arenas = np.array(["1"] * 20 + ["2"] * 20)

        def stat(perm):
            table = np.zeros((2, 2))
            np.add.at(table, (pre, perm), 1.0)
            return g_statistic(table)

        observed, p_value = stratified_permutation_pvalue(
            post, arenas, stat, iterations=500, rng=rng
        )
        assert observed > 20.0
        assert p_value < 0.02

    def test_independent_labels_permutation_p_large(self):
        rng = np.random.default_rng(2)
        pre = np.array([0, 1] * 30)
        post = rng.integers(0, 2, size=60)
        arenas = np.array(["1"] * 30 + ["2"] * 30)

        def stat(perm):
            table = np.zeros((2, 2))
            np.add.at(table, (pre, perm), 1.0)
            return g_statistic(table)

        _, p_value = stratified_permutation_pvalue(
            post, arenas, stat, iterations=500, rng=rng
        )
        assert p_value > 0.05


class TestStratifiedPermutation:
    def test_permutation_preserves_stratum_composition(self):
        rng = np.random.default_rng(3)
        values = np.array(["a", "a", "b", "c", "c", "c"], dtype=object)
        strata = np.array(["s1", "s1", "s1", "s2", "s2", "s2"])
        seen: list[np.ndarray] = []

        def stat(perm):
            seen.append(perm.copy())
            return 0.0

        stratified_permutation_pvalue(
            values, strata, stat, iterations=50, rng=rng
        )
        for perm in seen[1:] if len(seen) > 1 else seen:
            assert sorted(perm[:3]) == ["a", "a", "b"]
            assert sorted(perm[3:]) == ["c", "c", "c"]


class TestDecoder:
    @staticmethod
    def _windows(separable: bool, seed: int = 4):
        rng = np.random.default_rng(seed)
        rows = []
        for rec_index in range(16):
            rec = f"r{rec_index}_arena_{rec_index % 2 + 1}_x"
            for epoch in EPOCH_ROLES:
                shift = (
                    (3.0 if epoch == "chaser_post" else 0.0) if separable else 0.0
                )
                for window_index in range(6):
                    row = {
                        "recording_id": rec,
                        "epoch_role": epoch,
                        "window_index": window_index,
                        "start_frame": window_index * 6000,
                        "end_frame_exclusive": (window_index + 1) * 6000,
                    }
                    for column in DECODER_FEATURE_COLUMNS:
                        row[column] = float(rng.normal() + shift)
                    rows.append(row)
        return pl.DataFrame(rows)

    def test_separable_windows_auc_near_one(self):
        config = StrategyStatesConfig()
        result = run_loro_decoder(
            self._windows(separable=True), config, run_name="test"
        )
        assert result.overall_auc > 0.99

    def test_unseparable_windows_auc_near_half(self):
        config = StrategyStatesConfig()
        result = run_loro_decoder(
            self._windows(separable=False), config, run_name="test"
        )
        assert abs(result.overall_auc - 0.5) < 0.12

    def test_post_window_range_filters(self):
        config = StrategyStatesConfig()
        result = run_loro_decoder(
            self._windows(separable=True),
            config,
            run_name="ranged",
            post_window_range=(0, 2),
        )
        post = result.window_scores.filter(pl.col("epoch_role") == "chaser_post")
        assert post["window_index"].max() == 2
        pre = result.window_scores.filter(pl.col("epoch_role") == "chaser_pre")
        assert pre["window_index"].max() == 5


class TestWindowFeatures:
    def test_bout_and_ibi_window_assignment(self):
        fps = 100.0
        config = StrategyStatesConfig()
        epochs = pl.DataFrame(
            {
                "recording_id": ["r_arena_1_x"],
                "analysis_role": ["chaser_pre"],
                "start_frame": [0],
                "end_frame_exclusive": [12000],  # two 60 s windows
            }
        )
        bouts = pl.DataFrame(
            {
                "recording_id": ["r_arena_1_x"] * 3,
                "start_acquisition_frame_id": [100, 5000, 6500],
                "end_acquisition_frame_id": [200, 5100, 6600],
                "duration_s": [1.0, 1.0, 1.0],
                "path_length_mm": [5.0, 5.0, 5.0],
                "net_displacement_mm": [4.0, 4.0, 4.0],
                "peak_speed_mm_s": [30.0, 30.0, 30.0],
                "tortuosity": [1.2, float("inf"), 1.4],
            }
        )
        windows = build_window_features(
            bouts, epochs, {"r_arena_1_x": fps}, config
        )
        assert windows.height == 2
        first = windows.filter(pl.col("window_index") == 0)
        second = windows.filter(pl.col("window_index") == 1)
        assert first["n_bouts"][0] == 2
        assert second["n_bouts"][0] == 1
        # IBI 1: start at frame 200 -> window 0, 48 s. IBI 2: start at 5100 ->
        # window 0, 14 s.
        assert first["ibi_s_median"][0] == pytest.approx((48.0 + 14.0) / 2.0)
        assert first["ibi_frac_gt_2s"][0] == pytest.approx(1.0)
        assert math.isnan(second["ibi_s_median"][0])
        # finite-only tortuosity median in window 0: median(1.2) == 1.2
        assert first["bout_tortuosity_finite_median"][0] == pytest.approx(1.2)

    def test_default_windows_drop_partial_tail(self):
        rows = [
            {
                "analysis_role": "chaser_pre",
                "start_frame": 0,
                "end_frame_exclusive": 6100 * 2 + 100,
            }
        ]
        windows = default_epoch_windows(rows, 100.0, 60.0)
        assert len(windows) == 2
        assert windows[-1]["end_frame_exclusive"] == 12000


def _cluster_result_from_z(
    z: np.ndarray, labels: np.ndarray, recordings: list[str]
) -> ClusterResult:
    index = pl.DataFrame(
        {
            "recording_id": [rec for rec in recordings for _ in EPOCH_ROLES],
            "epoch_role": [e for _ in recordings for e in EPOCH_ROLES],
        }
    )
    k = int(labels.max()) + 1
    return ClusterResult(
        selected_k=k,
        bic_table=pl.DataFrame({"k": [k], "bic": [0.0]}),
        assignments=index,
        cluster_feature_means=pl.DataFrame(),
        z_matrix=z,
        z_index=index,
        feature_means=np.zeros(z.shape[1]),
        feature_stds=np.ones(z.shape[1]),
        labels=labels,
        posteriors=np.eye(k)[labels],
    )


class TestDirection:
    def test_axis_sign_convention(self):
        n_features = len(FEATURE_COLUMNS)
        entropy_i = FEATURE_COLUMNS.index("occupancy_entropy")
        bout_i = FEATURE_COLUMNS.index("bout_rate_per_min")
        ibi_i = FEATURE_COLUMNS.index("ibi_gt2s")
        rng = np.random.default_rng(5)
        recordings = [f"r{i}_arena_1_x" for i in range(10)]
        z = rng.normal(scale=0.01, size=(20, n_features))
        labels = np.array([0, 1] * 10)
        # Cluster 1 has HIGHER entropy/bout rate and LOWER ibi tail, so the
        # raw (cluster1 - cluster0) axis violates the convention and must flip.
        z[labels == 1, entropy_i] += 2.0
        z[labels == 1, bout_i] += 2.0
        z[labels == 1, ibi_i] -= 2.0
        result = _cluster_result_from_z(z, labels, recordings)
        unit_axis, cluster_a, cluster_b, flipped = punctuated_axis(result)
        assert flipped is True
        assert unit_axis[entropy_i] < 0
        assert unit_axis[bout_i] < 0
        assert unit_axis[ibi_i] > 0
        assert np.linalg.norm(unit_axis) == pytest.approx(1.0)
        assert {cluster_a, cluster_b} == {0, 1}

    def test_decomposition_orthogonality(self):
        n_features = len(FEATURE_COLUMNS)
        rng = np.random.default_rng(6)
        recordings = [f"r{i}_arena_1_x" for i in range(8)]
        z = rng.normal(size=(16, n_features))
        axis = rng.normal(size=n_features)
        axis /= np.linalg.norm(axis)
        result = _cluster_result_from_z(
            z, np.array([0, 1] * 8), recordings
        )
        decomposition = decompose_displacements(result, axis)
        assert decomposition.height == len(recordings)
        for row in decomposition.to_dicts():
            assert row["component_parallel"] ** 2 + row[
                "component_orthogonal"
            ] ** 2 == pytest.approx(row["displacement_norm"] ** 2, rel=1e-9)


class TestArena:
    def test_parse_arena(self):
        assert parse_arena("2026-08-10T17-20-55Z_arena_1_goodbatbadbat") == "1"

    def test_parse_arena_missing_fails(self):
        with pytest.raises(StrategyStatesInputError):
            parse_arena("no_arena_token_here_but_not_matching")
