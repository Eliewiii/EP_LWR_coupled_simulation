"""Unit tests validating EnergyPlus JSON primitive translation helper functions."""

from lwrepcoupling.tools.ep_models.utils_epjson import to_epboolean
from pyenergyplus.model.model import EPBoolean


def test_to_epboolean_conversion_true():
    """
    Verifies that a Python True primitive maps to a type-compliant EPBoolean.yes string choice.
    """
    assert to_epboolean(True) == EPBoolean.yes


def test_to_epboolean_conversion_false():
    """
    Verifies that a Python False primitive maps to a type-compliant EPBoolean.no string choice.
    """
    assert to_epboolean(False) == EPBoolean.no
