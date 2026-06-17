"""Unit tests for mathematical and geometric solver schema configurations."""

from typing import Any

import pytest
from lwrepcoupling._schemas import SurfaceAddStringConfig
from pydantic import ValidationError

# =====================================================================
# SurfaceAddStringConfig Tests
# =====================================================================


def test_surface_add_string_config_valid() -> None:
    """Verify that a physically sound view factor distribution initializes cleanly."""
    config = SurfaceAddStringConfig(
        surface_name="Zone1_SouthWall",
        cumulated_ext_surf_view_factor=0.4,
        sky_view_factor=0.3,
        ground_view_factor=0.2,
        ground_temperature_schedule="Ground_Temp_Sch",
    )
    assert config.surface_name == "Zone1_SouthWall"
    assert config.ground_temperature_schedule == "Ground_Temp_Sch"


def test_surface_add_string_config_defaults() -> None:
    """Verify that sky and ground view factors gracefully fallback to None/0 values."""
    config = SurfaceAddStringConfig(
        surface_name="Roof_Isolated", cumulated_ext_surf_view_factor=1.0
    )
    assert config.sky_view_factor is None
    assert config.ground_view_factor is None
    assert config.ground_temperature_schedule == ""


def test_surface_add_string_config_invalid_sum_throws() -> None:
    """Verify that total view factor sums exceeding 1.0 trigger a validation error."""
    # Sum = 0.6 (Ext) + 0.3 (Sky) + 0.2 (Ground) = 1.1 -> Must fail
    with pytest.raises(ValidationError, match="exceeds 1.0"):
        SurfaceAddStringConfig(
            surface_name="Corrupted_Surface",
            cumulated_ext_surf_view_factor=0.6,
            sky_view_factor=0.3,
            ground_view_factor=0.2,
        )


@pytest.mark.parametrize(
    "field_name, invalid_value",
    [
        ("surface_name", ""),  # Min length constraint
        ("cumulated_ext_surf_view_factor", -0.1),  # Must be gt=0
        ("cumulated_ext_surf_view_factor", 1.05),  # Must be le=1
        ("sky_view_factor", -0.1),  # Must be ge=0
        ("ground_view_factor", 1.2),  # Must be le=1
    ],
)
def test_surface_add_string_field_constraints(field_name: str, invalid_value: Any) -> None:
    """Verify standard field-level physical boundary validation rules."""
    valid_payload = {
        "surface_name": "Wall",
        "cumulated_ext_surf_view_factor": 0.5,
        "sky_view_factor": 0.2,
        "ground_view_factor": 0.2,
    }
    valid_payload[field_name] = invalid_value

    with pytest.raises(ValidationError):
        SurfaceAddStringConfig(**valid_payload)
