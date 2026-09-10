from __future__ import annotations

import numpy as np
import pytest

from spine_sim.metrics import (
    SpineMetricInput,
    compute_array_counts,
    integrate_path_resistance,
)


def test_counts_keep_physical_engagement_separate_from_evaluability() -> None:
    result = compute_array_counts(
        (
            SpineMetricInput(True, 0.0, True, True, 2.0, 3.0),
            SpineMetricInput(True, 0.0, None, False, 0.0, 0.0),
            SpineMetricInput(False, None, False, False),
        ),
        gap_tolerance_m=1e-9,
    )
    assert result.n_nominal == 3
    assert result.n_geometric == 2
    assert result.n_contact == 2
    assert result.n_engaged is None
    assert (result.n_engaged_lower, result.n_engaged_upper) == (1, 2)
    assert result.n_evaluable == 2
    assert result.n_active == 1
    assert result.P_sum_N == 2.0
    assert result.P_avg_N == 1.0
    assert result.n_share_normal == 1.0


def test_zero_contact_average_is_undefined_and_unknown_force_propagates() -> None:
    empty = compute_array_counts(
        (SpineMetricInput(False, None, False, False),),
        gap_tolerance_m=0.0,
    )
    assert empty.P_sum_N == 0.0
    assert empty.P_avg_N is None
    unknown = compute_array_counts(
        (SpineMetricInput(True, 0.0, True, True, None, 1.0),),
        gap_tolerance_m=0.0,
    )
    assert unknown.P_sum_N is None
    assert unknown.P_avg_N is None

    with np.errstate(all="raise"):
        underflow = compute_array_counts(
            (SpineMetricInput(True, 0.0, True, True, 1e-200, 1e-200),),
            gap_tolerance_m=0.0,
        )
    assert underflow.n_share_normal is None
    assert underflow.n_share_tangent_positive is None


def test_path_integral_does_not_bridge_invalid_gap_or_fill_it_with_zero() -> None:
    x = np.arange(5, dtype=float)
    result = integrate_path_resistance(
        x,
        np.array([2.0, 2.0, np.nan, -2.0, -2.0]),
        external_normal_preload_N=2.0,
        accepted=np.ones(5, dtype=bool),
        valid=np.array([True, True, False, True, True]),
    )
    assert result.effective_length_m == 2.0
    assert result.full_length_m == 4.0
    assert result.coverage == 0.5
    assert result.J_positive is None
    assert result.J_negative is None
    assert result.J_net is None
    assert result.observed_J_positive == 0.25
    assert result.observed_J_negative == 0.25
    assert result.observed_J_net == 0.0
    assert result.interval_count == 2


def test_path_without_effective_interval_remains_unknown() -> None:
    result = integrate_path_resistance(
        [0.0, 1.0], [1.0, 1.0], external_normal_preload_N=1.0,
        accepted=[True, False], valid=[True, True],
    )
    assert result.J_positive is None
    assert result.coverage == 0.0
    assert not result.complete


def test_macro_signed_normal_and_local_normal_are_distinct() -> None:
    values = [
        SpineMetricInput.from_force(
            wall_force_N=force, contact_normal=normal, geometric=True,
            signed_gap_m=0.0, engagement=True,
        )
        for force, normal in [((-3.0, 0.0, -1.0), (-1.0, 0.0, 0.0)),
                              ((-4.0, 0.0, 2.0), (0.0, 0.0, 1.0))]
    ]
    result = compute_array_counts(values, gap_tolerance_m=0.0)
    assert result.P_sum_N == 1.0
    assert result.N_sum_local_N == 5.0
    assert result.n_share_normal == 1.0
    assert result.n_share_local_normal == pytest.approx(25.0 / 13.0)
    assert result.n_share_tangent_positive == pytest.approx(49.0 / 25.0)
    assert result.load_sharing_index == pytest.approx(2 * np.sqrt(20) / (np.sqrt(10) + np.sqrt(20)))
    assert result.normal_weight_basis == "macro_P_positive"
    assert result.load_sharing_basis == "force_norm"


def test_declared_window_zero_crossing_and_event_jump() -> None:
    crossing = integrate_path_resistance(
        [0.0, 2.0], [2.0, -2.0], external_normal_preload_N=2.0,
        accepted=[True, True], valid=[True, True], window_m=(0.0, 2.0),
        force_quantile=0.25,
    )
    assert crossing.J_positive == pytest.approx(0.25)
    assert crossing.J_negative == pytest.approx(0.25)
    assert crossing.T_quantile_N == pytest.approx(-1.0)
    jumps = integrate_path_resistance(
        [0.0, 1.0, 1.0, 2.0], [2.0, 2.0, -2.0, -2.0],
        external_normal_preload_N=2.0, accepted=[True] * 4, valid=[True] * 4,
        window_m=(0.5, 1.5),
    )
    assert jumps.complete
    assert jumps.J_positive == pytest.approx(0.5)
    assert jumps.J_negative == pytest.approx(0.5)
    terminated = integrate_path_resistance(
        [0.0, 1.0], [2.0, 2.0], external_normal_preload_N=2.0,
        accepted=[True, True], valid=[True, True], window_m=(0.0, 2.0),
    )
    assert terminated.J_positive is None
    assert terminated.observed_J_positive == pytest.approx(0.5)
    assert terminated.coverage == 0.5
