from __future__ import annotations

import math
import unittest

import numpy as np

from spine_sim.core.errors import ConfigurationError
from spine_sim.core.frames import FrameMetadata, Wrench
from spine_sim.core.units import require_range, to_si


class UnitTests(unittest.TestCase):
    def test_si_and_explicit_units(self) -> None:
        self.assertEqual(to_si(0.004, "length"), 0.004)
        self.assertAlmostEqual(to_si({"value": 4, "unit": "mm"}, "length"), 0.004)
        self.assertAlmostEqual(to_si({"value": 60, "unit": "deg"}, "angle"), math.pi / 3)

    def test_wrong_dimension_and_range_are_rejected(self) -> None:
        with self.assertRaises(ConfigurationError):
            to_si({"value": 1, "unit": "N"}, "length")
        with self.assertRaises(ConfigurationError):
            require_range(-1.0, name="length", minimum=0.0)


class FrameTests(unittest.TestCase):
    def setUp(self) -> None:
        self.wrench = Wrench(
            force_N=(0.0, 1.0, 0.0),
            moment_Nm=(0.0, 0.0, 0.0),
            frame="unit",
            reference_point="O",
            acting_on="plate",
            exerted_by="spine",
        )

    def test_rotation_hand_fixture(self) -> None:
        rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        rotated = self.wrench.rotate(rotation, new_frame="global")
        np.testing.assert_allclose(rotated.force_N, (-1.0, 0.0, 0.0), atol=1e-12)
        self.assertEqual(rotated.interaction_label, "spine_on_plate")

    def test_reference_move_hand_fixture(self) -> None:
        # P→O = +x and F = +y, therefore M_P = +z.
        moved = self.wrench.move_reference((2.0, 0.0, 0.0), new_reference_point="P")
        np.testing.assert_allclose(moved.moment_Nm, (0.0, 0.0, 2.0), atol=1e-12)

    def test_bad_rotation_is_rejected(self) -> None:
        with self.assertRaises(ConfigurationError):
            FrameMetadata(
                name="bad",
                parent=None,
                origin_m=(0, 0, 0),
                rotation_to_parent=((2, 0, 0), (0, 1, 0), (0, 0, 1)),
            )


if __name__ == "__main__":
    unittest.main()
