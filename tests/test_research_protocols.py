from dataclasses import replace

import numpy as np
import pytest

from spine_sim.core.config import BaseCaseSpec
from spine_sim.research_protocols import (
    EstablishmentProtocol, LoadProtocol, ResearchSample, SensorCalibration,
    evaluate_establishment, iter_paired_cases, summarize_establishment,
)


def test_establishment_requires_persistence_and_interpolates_threshold() -> None:
    protocol = EstablishmentProtocol(2.0, 1.0, (0.0, 3.0))
    result = evaluate_establishment(
        [0.0, 1.0, 3.0], [0.0, 4.0, 4.0],
        accepted=[True] * 3, valid=[True] * 3, protocol=protocol,
    )
    assert result.established is True
    assert result.S_req_m == pytest.approx(0.5)
    assert result.confirmed_at_m == pytest.approx(1.5)
    spike = evaluate_establishment(
        [0.0, 1.0, 2.0, 3.0], [0.0, 3.0, 0.0, 0.0],
        accepted=[True] * 4, valid=[True] * 4, protocol=protocol,
    )
    assert spike.established is False
    missing = evaluate_establishment(
        [0.0, 1.0, 2.0, 3.0], [0.0, np.nan, 4.0, 4.0],
        accepted=[True] * 4, valid=[True, False, True, True], protocol=protocol,
    )
    assert missing.established is True
    assert missing.S_req_m is None
    assert missing.observed_S_req_m == 2.0
    assert missing.status == "established_earliest_unknown"


def test_allowable_drop_and_search_deadline() -> None:
    x = [0.0, 0.5, 0.5, 0.75, 0.75, 2.0]
    force = [4.0, 4.0, 0.0, 0.0, 4.0, 4.0]
    protocol = EstablishmentProtocol(2.0, 1.0, (0.0, 2.0))
    strict = evaluate_establishment(x, force, accepted=[True] * 6, valid=[True] * 6, protocol=protocol)
    allowed = evaluate_establishment(
        x, force, accepted=[True] * 6, valid=[True] * 6,
        protocol=replace(protocol, allowed_below_fraction=0.25, max_contiguous_below_m=0.25),
    )
    assert strict.S_req_m == 0.75
    assert allowed.S_req_m == 0.0
    deadline = evaluate_establishment(
        [0.0, 1.0, 1.0, 2.0], [0.0, 0.0, 4.0, 4.0],
        accepted=[True] * 4, valid=[True] * 4,
        protocol=replace(protocol, persistence_distance_m=1.25),
    )
    assert deadline.established is False


def test_incomplete_search_is_not_counted_as_failure() -> None:
    protocol = EstablishmentProtocol(2.0, 1.0, (0.0, 2.0))
    incomplete = evaluate_establishment(
        [0.0, 0.5], [0.0, 0.0], accepted=[True] * 2, valid=[True] * 2, protocol=protocol,
    )
    assert incomplete.established is None
    population = summarize_establishment([incomplete])
    assert population.probability is None
    assert population.probability_lower == 0.0
    assert population.probability_upper == 1.0
    immediate = evaluate_establishment(
        [0.0], [3.0], accepted=[True], valid=[True],
        protocol=replace(protocol, persistence_distance_m=0.0),
    )
    assert immediate.established is True
    assert immediate.S_req_m == 0.0


def test_load_pairing_and_sensor_reference_point_transform() -> None:
    assert LoadProtocol("pressure", 20.0).total_preload_N(area_m2=0.1, n_nominal=8) == 2.0
    assert LoadProtocol("nominal_per_spine", 0.5).total_preload_N(area_m2=0.1, n_nominal=8) == 4.0
    samples = [ResearchSample("surface", 17, 23, (0.1, 0.2), "test")]
    designs = [BaseCaseSpec("guided", "v1", {"design": angle}) for angle in (30, 60)]
    cases = list(iter_paired_cases(designs, samples))
    assert cases[0].parameters["sample"] == cases[1].parameters["sample"]
    assert cases[0].case_id != cases[1].case_id
    calibration = SensorCalibration(
        ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        (0.0, 0.0, 2.0), (1.0, 0.0, 0.0, 0.0, 0.0, 0.0), 1,
    )
    wrench = calibration.wrench_to_backplate([4.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(wrench, [0.0, 3.0, 0.0, -6.0, 0.0, 0.0])
