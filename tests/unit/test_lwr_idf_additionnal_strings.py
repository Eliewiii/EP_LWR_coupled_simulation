"""
_summary_
"""

import pytest
from pydantic import ValidationError

# Replace 'your_module' with your actual file name
from src.lwrepcoupling.lwr_idf_additionnal_strings import (
    SurfaceAddStringConfig,
    generate_surface_lwr_idf_additional_string,
)

# ---------------------------------------------------------
# Testing the Inputs and Constraints
# ---------------------------------------------------------


def test_valid_view_factor_sum():
    """Ensure a physically valid combination passes validation."""
    config = SurfaceAddStringConfig(
        surface_name="Zone1_WallNorth",
        cumulated_ext_surf_view_factor=0.5,
        sky_view_factor=0.3,
        ground_view_factor=0.1,
    )
    assert config.surface_name == "Zone1_WallNorth"


def test_invalid_total_view_factor_throws():
    """Ensure a sum greater than 1.0 + Tolerance triggers a ValueError."""
    with pytest.raises(ValueError, match="exceeds 1.0"):
        SurfaceAddStringConfig(
            surface_name="Zone1_WallNorth",
            cumulated_ext_surf_view_factor=0.7,
            sky_view_factor=0.4,  # Sum = 1.1 -> Should crash
        )


def test_negative_view_factor_throws():
    """Pydantic ge=0 check should trap negative physics parameters."""
    with pytest.raises(ValidationError):
        SurfaceAddStringConfig(
            surface_name="Zone1_WallNorth",
            cumulated_ext_surf_view_factor=0.5,
            sky_view_factor=-0.15,
        )


# ---------------------------------------------------------
# Structural Testing (No exact text matching)
# ---------------------------------------------------------


def test_idf_string_structure_with_optionals():
    """Ensure optional fields map cleanly into the final text format when populated."""
    config = SurfaceAddStringConfig(
        surface_name="WallA",
        cumulated_ext_surf_view_factor=0.4,
        sky_view_factor=0.3,
        ground_view_factor=0.2,
    )

    result = generate_surface_lwr_idf_additional_string(config)

    # Check that crucial identifiers are embedded
    assert "WallA_locEnv" in result
    assert "WallA_SurSur" in result

    # Check that the numbers actually made it into the text block
    assert "0.3" in result  # Sky VF
    assert "0.2" in result  # Ground VF


def test_idf_string_structure_with_missing_optionals():
    """Ensure None values correctly translate into safe empty fields for EnergyPlus."""
    config = SurfaceAddStringConfig(
        surface_name="WallB",
        cumulated_ext_surf_view_factor=0.6,
        sky_view_factor=None,
        ground_view_factor=None,
    )

    result = generate_surface_lwr_idf_additional_string(config)

    # Verify the object references exist
    assert "\n  , !- Sky View Factor\n" in result

    # Crucial: Since they were None, verify they don't corrupt the formatting
    # EnergyPlus expects standard empty commas for defaulted fields
    assert "\n  , !- Ground View Factor" in result
