"""Unit tests for parallel loop coordination and runtime manifest contracts."""

from pathlib import Path

import pytest
from lwrepcoupling._schemas import (
    CompiledBuildingState,
    EpSimulationRuntimeConfig,
    SimulationManifest,
)
from lwrepcoupling._schemas.runtime_models import SynchronizerBarrier
from pydantic import ValidationError

# =====================================================================
# Mock Objects for Multiprocessing Structural Protocols
# =====================================================================


class MockMultiprocessingBarrier:
    """A minimal mock matching the SynchronizerBarrier Protocol interface."""

    def wait(self, timeout: float | None = None) -> int:
        return 0

    def abort(self) -> None:
        pass


@pytest.fixture
def valid_manifest_data(tmp_path: Path) -> dict:
    """Generates a geometrically pristine data payload for a simulation manifest."""
    return {
        "workspace_dir": tmp_path / "sim_workspace",
        "epw_file_name": "climate.epw",
        "num_ts_per_h": 4,
        "num_total_surfaces": 5,
        "save_resolution_matrix": False,
        "enable_lwr_coupling": True,
        "compiled_buildings": [
            CompiledBuildingState(
                building_id="b_0",
                building_index=0,
                num_surfaces=2,
                surface_index_min=0,
                surface_index_max=1,
                outdoor_surface_names=["s1", "s2"],
            ),
            CompiledBuildingState(
                building_id="b_1",
                building_index=1,
                num_surfaces=3,
                surface_index_min=2,
                surface_index_max=4,
                outdoor_surface_names=["s3", "s4", "s5"],
            ),
        ],
    }


# =====================================================================
# CompiledBuildingState Tests
# =====================================================================


def test_compiled_building_state_pure_formulas() -> None:
    """Verify static naming and location-derivation string math configurations."""
    assert CompiledBuildingState.short_id(3) == "b_3"
    assert CompiledBuildingState.mtx_file_name(3) == "b_3_mtx.npy"

    mock_runs_dir = Path("/tmp/workspace/runs")
    assert CompiledBuildingState.derive_output_dir(mock_runs_dir, 2) == Path(
        "/tmp/workspace/runs/b_2"
    )
    assert CompiledBuildingState.derive_idf_path(mock_runs_dir, 2) == Path(
        "/tmp/workspace/runs/b_2/b_2.idf"
    )


def test_compiled_building_state_instance_delegation() -> None:
    """Verify live state object instance methods delegate cleanly down to class formulas."""
    state = CompiledBuildingState(
        building_id="b_0",
        building_index=0,
        num_surfaces=1,
        surface_index_min=0,
        surface_index_max=0,
        outdoor_surface_names=["surf"],
    )
    mock_runs_dir = Path("/tmp/workspace/runs")
    assert state.get_output_dir(mock_runs_dir) == Path("/tmp/workspace/runs/b_0")
    assert state.get_idf_path(mock_runs_dir) == Path("/tmp/workspace/runs/b_0/b_0.idf")


# =====================================================================
# SimulationManifest Alignment Tests
# =====================================================================


def test_manifest_valid_construction(valid_manifest_data) -> None:
    """Verify clean instantiation and path-property resolution logic loops."""
    manifest = SimulationManifest(**valid_manifest_data)
    assert manifest.num_buildings == 2
    assert manifest.runs_dir == valid_manifest_data["workspace_dir"] / "runs"
    assert manifest.epw_path == valid_manifest_data["workspace_dir"] / "climate.epw"


def test_manifest_sequence_integrity_gap_throws(valid_manifest_data) -> None:
    """Ensure list position mismatches break execution before boot loops."""
    # Corrupt order: list index 1 holds building_index 0 instead of 1
    valid_manifest_data["compiled_buildings"][1].building_index = 0
    with pytest.raises(ValidationError, match="Critical manifest alignment error"):
        SimulationManifest(**valid_manifest_data)


def test_manifest_surface_count_mismatch_throws(valid_manifest_data) -> None:
    """Ensure conflicts between global surface tracking arrays and item counts trigger faults."""
    # Total sum of surfaces in building list is 5. Declare a corrupt system dimension of 99.
    valid_manifest_data["num_total_surfaces"] = 99
    with pytest.raises(ValidationError, match="The total number of outdoor surfaces tracked"):
        SimulationManifest(**valid_manifest_data)


# =====================================================================
# Serialization & Path Self-Healing Integration Tests
# =====================================================================


def test_manifest_io_and_self_healing_lifecycle(tmp_path: Path, valid_manifest_data) -> None:
    """Verify moving a workspace directory root automatically triggers path adjustments on load."""
    original_ws = tmp_path / "old_workspace_anchor"
    original_ws.mkdir()
    valid_manifest_data["workspace_dir"] = original_ws

    manifest = SimulationManifest(**valid_manifest_data)
    manifest.write_to_disk()
    assert manifest.json_path.is_file()

    # Simulate filesystem drift (e.g., pipeline deployment tracking)
    moved_ws = tmp_path / "new_relocated_workspace"
    original_ws.rename(moved_ws)
    moved_json_file = moved_ws / SimulationManifest.MANIFEST_FILE_NAME

    # Read payload back using the load gateway
    healed_manifest = SimulationManifest.load_and_adapt(moved_json_file)
    assert healed_manifest.workspace_dir == moved_ws
    assert healed_manifest.runs_dir == moved_ws / "runs"


# =====================================================================
# EpSimulationRuntimeConfig Tests
# =====================================================================


def test_runtime_config_arbitrary_types_allowed(tmp_path: Path) -> None:
    """Verify that duck-typed multiprocessing primitives pass verification gates via ConfigDict."""
    b_state = CompiledBuildingState(
        building_id="b_0",
        building_index=0,
        num_surfaces=1,
        surface_index_min=0,
        surface_index_max=0,
        outdoor_surface_names=["s"],
    )
    mock_barrier = MockMultiprocessingBarrier()

    config = EpSimulationRuntimeConfig(
        building_state=b_state,
        epw_path=tmp_path / "climate.epw",
        num_ts_per_h=4,
        runs_dir=tmp_path / "runs",
        num_buildings=1,
        num_total_surfaces=1,
        shared_memory_temperatures_name="mem_temp",
        shared_memory_timesteps_name="mem_ts",
        synch_point_barrier=mock_barrier,
        enable_lwr_coupling=True
    )
    assert isinstance(config.synch_point_barrier, SynchronizerBarrier)
