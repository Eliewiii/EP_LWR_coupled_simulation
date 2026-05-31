"""Unit tests for EnergyPlus IDF additional string macro injection utilities."""

import pytest
from lwrepcoupling._schemas import SurfaceAddStringConfig
from lwrepcoupling._utils import generate_idfs_additional_strings
from pydantic import ValidationError

# =====================================================================
# 1. Testing Input Boundaries & Constraints (Pydantic Layer)
# =====================================================================


def test_valid_view_factor_sum() -> None:
    """Ensure a physically valid combination passes verification gates cleanly."""
    config = SurfaceAddStringConfig(
        surface_name="Zone1_WallNorth",
        cumulated_ext_surf_view_factor=0.5,
        sky_view_factor=0.3,
        ground_view_factor=0.1,
    )
    assert config.surface_name == "Zone1_WallNorth"
    assert config.cumulated_ext_surf_view_factor == 0.5


def test_invalid_total_view_factor_throws() -> None:
    """Ensure a total sum breaking energy conservation (>1.0 + Tol) triggers a ValueError."""
    # Sum = 0.7 + 0.4 = 1.1 -> Must raise ValueError via @model_validator
    with pytest.raises(ValueError, match="exceeds 1.0"):
        SurfaceAddStringConfig(
            surface_name="Zone1_WallNorth",
            cumulated_ext_surf_view_factor=0.7,
            sky_view_factor=0.4,
        )


def test_negative_view_factor_throws() -> None:
    """Verify that negative physics fields are rejected by Field constraints."""
    with pytest.raises(ValidationError):
        # sky_view_factor must be >= 0
        SurfaceAddStringConfig(
            surface_name="Zone1_WallNorth",
            cumulated_ext_surf_view_factor=0.5,
            sky_view_factor=-0.15,
        )


# =====================================================================
# 2. Structural & Generation Testing (API Surface Layer)
# =====================================================================


def test_generate_idfs_additional_strings_with_optionals() -> None:
    """Verify that optional lists translate cleanly into the final EnergyPlus text block."""
    surfaces = ["WallA", "WallB"]
    srd_vfs = [0.4, 0.3]
    sky_vfs = [0.3, 0.2]
    ground_vfs = [0.2, 0.1]

    result = generate_idfs_additional_strings(
        outdoor_surface_names=surfaces,
        srd_vf_list=srd_vfs,
        sky_vf_list=sky_vfs,
        ground_vf_list=ground_vfs,
    )

    # 1. Verify basic type contract
    assert isinstance(result, str)

    # 2. Verify macro block components are injected for both surfaces
    for name in surfaces:
        assert f"{name}_locEnv" in result
        assert f"{name}_SurSur" in result
        assert f"{name}_ConSur" in result
        assert f"{name}_ConSur_TemSch" in result

    # 3. Structural text validation to prove numbers mapped correctly
    assert "0.4" in result  # WallA external view factor
    assert "0.3" in result  # WallA sky view factor
    assert "0.2" in result  # WallA ground view factor


def test_generate_idfs_additional_strings_with_missing_optionals() -> None:
    """Verify that None/empty optional parameters yield safe defaulted fields in the IDF."""
    surfaces = ["WallIsolated"]
    srd_vfs = [0.6]

    # Leave sky and ground lists empty to simulate defaulted environments
    result = generate_idfs_additional_strings(
        outdoor_surface_names=surfaces,
        srd_vf_list=srd_vfs,
        sky_vf_list=None,
        ground_vf_list=None,
    )

    # EnergyPlus uses standard trailing empty commas to indicate defaulted fields.
    # Prove that missing optionals preserve formatting slots safely.
    assert ", !- Sky Temperature Schedule Name" in result
    assert ", !- Ground Temperature Schedule Name" in result
