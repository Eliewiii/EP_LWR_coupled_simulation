"""Unit tests for the EpLwrSimulationManager workspace orchestration and lifecycle gatekeeper."""

import unittest.mock as mock
from pathlib import Path

import numpy as np
import pytest
from lwrepcoupling._schemas import (
    BuildingInput,
    CompiledBuildingState,
    InversionConfig,
    SimulationInputs,
    SimulationManifest,
)
from lwrepcoupling.ep_coupled_simulation_manager import EpLwrSimulationManager
from lwrepcoupling.exceptions import SimulationCrashError, WorkspaceConflictError
from scipy.sparse import csr_matrix, save_npz


@pytest.fixture
def mock_manifest(tmp_path: Path) -> SimulationManifest:
    """Provides a valid, isolated manifest data contract."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "haifa.epw").touch()

    return SimulationManifest(
        workspace_dir=ws,
        # energyplus_dir removed to support native pyenergyplus bindings
        epw_file_name="haifa.epw",
        num_ts_per_h=4,
        num_total_surfaces=4,
        save_resolution_matrix=False,
        enable_lwr_coupling=True,
        compiled_buildings=[
            CompiledBuildingState(
                building_id="b_0",
                building_index=0,
                num_surfaces=4,
                surface_index_min=0,
                surface_index_max=3,
                outdoor_surface_names=["s1", "s2", "s3", "s4"],
            )
        ],
    )


@pytest.fixture
def valid_inputs(tmp_path: Path) -> SimulationInputs:
    """Provides valid, mocked raw simulation inputs for workspace compilation testing."""
    ws = tmp_path / "target_workspace"
    ws.mkdir()

    # Generate mock sparse matrices
    vf_path = tmp_path / "vf.npz"
    save_npz(vf_path, csr_matrix(np.eye(4)))

    dummy_matrix = tmp_path / "dummy.npy"
    np.save(dummy_matrix, np.ones(4))

    weather = tmp_path / "weather.epw"
    weather.touch()

    # Create a legitimate mock layout file to satisfy Pydantic's FilePath validator
    dummy_idf = tmp_path / "geometry.idf"
    dummy_idf.touch()

    return SimulationInputs(
        workspace_dir=ws,
        # energyplus_dir removed to match native pyenergyplus integration
        epw_path=weather,
        num_ts_per_h=4,
        vf_matrix_path=vf_path,
        eps_matrix_path=dummy_matrix,
        rho_matrix_path=dummy_matrix,
        tau_matrix_path=dummy_matrix,
        inversion_parameters=InversionConfig(num_workers=1),
        # Provide a matching building entry so the matrix shape check matches total
        # outdoor surfaces (4)
        buildings=[
            BuildingInput(
                building_id="b_0",
                idf_path=dummy_idf,
                outdoor_surface_names=["s1", "s2", "s3", "s4"],
            )
        ],
        save_resolution_matrix=False,
    )


# =====================================================================
# Property and Initialization Tests
# =====================================================================


def test_manager_properties(mock_manifest) -> None:
    """
    Verify that manager attributes correctly delegate downstream to the manifest data contract.
    """
    manager = EpLwrSimulationManager(mock_manifest)
    assert manager.workspace_dir == mock_manifest.workspace_dir
    assert manager.epw_path == mock_manifest.epw_path
    assert manager.num_ts_per_h == 4
    assert manager.num_buildings == 1
    assert manager.num_total_surfaces == 4
    assert len(manager.buildings) == 1


# =====================================================================
# Workspace Creation and Exclusivity Validation Gates
# =====================================================================


def test_make_target_workspace_dir_clean_creation(tmp_path: Path) -> None:
    """Verify the manager generates a brand-new workspace directory if it doesn't exist."""
    target = tmp_path / "new_sim_workspace"
    assert not target.exists()

    EpLwrSimulationManager._make_target_workspace_dir(target, overwrite=False)
    assert target.exists()


def test_make_target_workspace_dir_occupied_no_overwrite_throws(tmp_path: Path) -> None:
    """Verify that an occupied directory throws a conflict error when overwrite is disabled."""
    target = tmp_path / "occupied_workspace"
    target.mkdir()
    (target / "foreign_user_file.txt").touch()

    with pytest.raises(WorkspaceConflictError, match="already exists and contains data"):
        EpLwrSimulationManager._make_target_workspace_dir(target, overwrite=False)


def test_verify_workspace_exclusivity_breach_throws(tmp_path: Path) -> None:
    """Verify that unexpected foreign files trigger an exclusivity security block."""
    target = tmp_path / "breached_workspace"
    target.mkdir()

    # Create legitimate files
    (target / SimulationManifest.MANIFEST_FILE_NAME).touch()
    # Drop a rogue file that breaks fingerprint rules
    (target / "malicious_script.sh").touch()

    with pytest.raises(WorkspaceConflictError, match="Workspace exclusivity breach"):
        EpLwrSimulationManager._verify_workspace_exclusivity(target)


def test_verify_workspace_exclusivity_multiple_epw_throws(tmp_path: Path) -> None:
    """Verify that hosting multiple climate assets triggers a non-deterministic fault code."""
    target = tmp_path / "multi_weather_workspace"
    target.mkdir()

    (target / "haifa_winter.epw").touch()
    (target / "haifa_summer.epw").touch()  # Secondary file breaches uniqueness bounds

    with pytest.raises(WorkspaceConflictError, match="Multiple climate profiles discovered"):
        EpLwrSimulationManager._verify_workspace_exclusivity(target)


# =====================================================================
# Isolated Process Spawning & Exit Code Analysis
# =====================================================================


def test_run_in_new_process_vanished_process_throws(tmp_path: Path) -> None:
    """Verify a SimulationCrashError is raised if a process terminates without an exit code."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / SimulationManifest.MANIFEST_FILE_NAME).touch()

    # Patch get_context dynamically at the manager level to keep OS isolation handling clean
    with (
        mock.patch("lwrepcoupling._schemas.runtime_models.SimulationManifest.load_and_adapt"),
        mock.patch(
            "lwrepcoupling.ep_coupled_simulation_manager.EpLwrSimulationManager._verify_workspace_integrity"
        ),
        mock.patch("lwrepcoupling.ep_coupled_simulation_manager.get_context") as mock_get_context,
    ):
        mock_context = mock.Mock()
        mock_process = mock.Mock()
        mock_process.exitcode = None  # Force unexpected missing exit status code

        mock_context.Process.return_value = mock_process
        mock_get_context.return_value = mock_context

        with pytest.raises(
            SimulationCrashError, match="process vanished without returning an exit code"
        ):
            EpLwrSimulationManager.run_in_new_process_from_workspace_dir(ws, debug_mode=False)


@pytest.mark.parametrize(
    "exit_code, expected_msg",
    [
        (1, "encountered an unhandled exception and crashed"),
        (-9, "strongly indicates an Out-Of-Memory / OOM termination"),
        (-15, "killed by operating system signal"),
    ],
)
def test_run_in_new_process_error_codes(tmp_path: Path, exit_code: int, expected_msg: str) -> None:
    """Verify that OS crash signals and bad exit codes are mapped to descriptive crash faults."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / SimulationManifest.MANIFEST_FILE_NAME).touch()

    with (
        mock.patch("lwrepcoupling._schemas.runtime_models.SimulationManifest.load_and_adapt"),
        mock.patch(
            "lwrepcoupling.ep_coupled_simulation_manager.EpLwrSimulationManager._verify_workspace_integrity"
        ),
        mock.patch("lwrepcoupling.ep_coupled_simulation_manager.get_context") as mock_get_context,
    ):
        mock_context = mock.Mock()
        mock_process = mock.Mock()
        mock_process.exitcode = exit_code

        mock_context.Process.return_value = mock_process
        mock_get_context.return_value = mock_context

        with pytest.raises(SimulationCrashError, match=expected_msg):
            EpLwrSimulationManager.run_in_new_process_from_workspace_dir(ws, debug_mode=False)
