"""Unit tests validating exact closed-form analytical view factor implementations."""

import numpy as np
from lwrepcoupling.tools.vf_computation.analytical_vf import (
    view_factor_parallel_plates,
    view_factor_perpendicular_plates,
)


# =====================================================================
# 1. Parallel Plates Formula Tests
# =====================================================================
def test_view_factor_parallel_plates_standard_case():
    """
    Verifies that parallel plates yield a physical, bounded float value under standard conditions.
    """
    # Symmetrical configuration: 10m x 3m plates, separated by 5m
    vf = view_factor_parallel_plates(w=10.0, h=3.0, d=5.0)

    assert isinstance(vf, float)
    assert 0.0 < vf < 1.0
    # Expected analytical range for these specific dimensions
    assert 0.15 < vf < 0.25


def test_view_factor_parallel_plates_boundary_guards():
    """
    Ensures zero or negative dimensions trigger safety guards and return 0.0 instead of crashing.
    """
    assert view_factor_parallel_plates(w=10.0, h=3.0, d=0.0) == 0.0
    assert view_factor_parallel_plates(w=10.0, h=3.0, d=-5.0) == 0.0
    assert view_factor_parallel_plates(w=0.0, h=3.0, d=5.0) == 0.0
    assert view_factor_parallel_plates(w=10.0, h=-1.0, d=5.0) == 0.0


def test_view_factor_parallel_plates_asymptotic_limit():
    """
    Asserts that the view factor safely approaches 0.0 as the separation gap approaches infinity.
    """
    vf_near = view_factor_parallel_plates(w=10.0, h=3.0, d=1.0)
    vf_far = view_factor_parallel_plates(w=10.0, h=3.0, d=500.0)

    assert vf_near > vf_far
    assert np.isclose(vf_far, 0.0, atol=1e-4)


# =====================================================================
# 2. Perpendicular Plates Formula Tests
# =====================================================================
def test_view_factor_perpendicular_plates_standard_case():
    """
    Verifies that perpendicular courtyard wall intersections return accurate bounded values.
    """
    # Shared junction edge height of 3.5m, emitting width 6.0m, receiving width 12.0m
    vf = view_factor_perpendicular_plates(w1=6.0, w2=12.0, h=3.5)

    assert isinstance(vf, float)
    assert 0.0 < vf < 1.0


def test_view_factor_perpendicular_plates_boundary_guards():
    """Ensures missing or zero shared junction boundaries safely exit with 0.0."""
    assert view_factor_perpendicular_plates(w1=6.0, w2=12.0, h=0.0) == 0.0
    assert view_factor_perpendicular_plates(w1=6.0, w2=12.0, h=-3.5) == 0.0
    assert view_factor_perpendicular_plates(w1=0.0, w2=12.0, h=3.5) == 0.0
    assert view_factor_perpendicular_plates(w1=6.0, w2=0.0, h=3.5) == 0.0
