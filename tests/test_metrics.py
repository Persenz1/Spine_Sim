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
    assert result.J_positive == 0.5
    assert result.J_negative == 0.5
    assert result.J_net == 0.0
    assert result.interval_count == 2


def test_path_contract_rejects_missing_effective_interval() -> None:
    with pytest.raises(ValueError, match="no accepted valid"):
        integrate_path_resistance(
            [0.0, 1.0],
            [1.0, 1.0],
            external_normal_preload_N=1.0,
            accepted=[True, False],
            valid=[True, True],
        )
