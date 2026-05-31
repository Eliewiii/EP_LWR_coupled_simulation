"""Unit tests for data ingestion, file paths, and matrix solver bounds."""

from pathlib import Path
from typing import Any

import pytest
from lwrepcoupling import BuildingInput, InversionConfig, SimulationInputs
from pydantic import ValidationError

# =====================================================================
# InversionConfig Tests
# =====================================================================


def test_inversion_config_defaults() -> None:
    """Verify standard default mathematical criteria initialization states."""
    config = InversionConfig()
    assert config.tol == 1e-5
    assert config.num_workers == 1


def test_inversion_config_cpu_count_fallback() -> None:
    """Verify that num_workers=0 automatically resolves to systemic core counts."""
    config = InversionConfig(num_workers=0)
    assert config.num_workers >= 1  # Confirms fallback system triggered correctly


@pytest.mark.parametrize(
    "field, invalid_val",
    [
        ("tol", 1e-9),  # Below ge bounds
        ("maxiter", 1500),  # Above le bounds
        ("rtol", 1e-4),  # Above le bounds
        ("num_workers", 65),  # Out of scaling limit
    ],
)
def test_inversion_config_invalid_bounds(field: str, invalid_val: Any) -> None:
    """Verify that mathematical execution parameters strictly block dangerous boundary values."""
    with pytest.raises(ValidationError):
        InversionConfig(**{field: invalid_val})


# =====================================================================
# BuildingInput Tests
# =====================================================================


def test_building_input_valid_extension(tmp_path: Path) -> None:
    """Verify clean instantiation of layout files using strict .idf formats."""
    valid_idf = tmp_path / "geometry.idf"
    valid_idf.touch()

    building = BuildingInput(building_id="b_0", idf_path=valid_idf, outdoor_surface_names=["s1"])
    assert building.building_id == "b_0"


def test_building_input_invalid_extension_throws(tmp_path: Path) -> None:
    """Verify that non-.idf layout assets are rejected by the field validator boundary."""
    bad_idf = tmp_path / "geometry.txt"
    bad_idf.touch()

    with pytest.raises(ValidationError, match="Expected a '.idf' file"):
        BuildingInput(building_id="b_0", idf_path=bad_idf, outdoor_surface_names=["s1"])


# =====================================================================
# SimulationInputs Tests
# =====================================================================


@pytest.fixture
def valid_inputs_payload(tmp_path: Path) -> dict:
    """Generates a complete layout dictionary payload for structural workspace validation."""
    # Setup folders
    ws = tmp_path / "ws"
    ws.mkdir()

    # Touch assets
    weather = tmp_path / "weather.epw"
    weather.touch()

    matrices = {}
    for prefix in ["vf", "eps", "rho", "tau"]:
        ext = ".npz"
        m_file = tmp_path / f"m_{prefix}{ext}"
        m_file.touch()
        matrices[f"{prefix}_matrix_path"] = m_file

    return {
        "workspace_dir": ws,
        "epw_path": weather,
        "num_ts_per_h": 4,
        **matrices,
        "buildings": [],
    }


def test_simulation_inputs_valid(valid_inputs_payload) -> None:
    """Verify successful parsing of pristine workspace initialization inputs."""
    inputs = SimulationInputs(**valid_inputs_payload)
    assert inputs.num_total_surfaces == 0


def test_simulation_inputs_invalid_weather_throws(valid_inputs_payload, tmp_path: Path) -> None:
    """Verify that climate sets missing strict .epw file layouts crash safely."""
    bad_weather = tmp_path / "weather.csv"
    bad_weather.touch()
    valid_inputs_payload["epw_path"] = bad_weather

    with pytest.raises(ValidationError, match="Expected a '.epw' file"):
        SimulationInputs(**valid_inputs_payload)


@pytest.mark.parametrize(
    "matrix_field", ["vf_matrix_path", "eps_matrix_path", "rho_matrix_path", "tau_matrix_path"]
)
def test_simulation_inputs_invalid_matrix_extensions_throw(
    valid_inputs_payload, tmp_path: Path, matrix_field: str
) -> None:
    """Verify batch field validator safely traps raw unverified format storage types."""
    bad_matrix = tmp_path / "corrupted_array.mat"
    bad_matrix.touch()
    valid_inputs_payload[matrix_field] = bad_matrix

    with pytest.raises(ValidationError, match="Invalid matrix file format"):
        SimulationInputs(**valid_inputs_payload)
