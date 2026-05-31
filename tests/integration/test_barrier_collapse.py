"""Integration test verifying real barrier collapse under authentic multiprocessing conditions

by providing incomplete physical runtime assets to a specific private worker loop.
"""

import logging
from pathlib import Path

import numpy as np
import pytest
from lwrepcoupling._schemas import CompiledBuildingState, SimulationManifest
from lwrepcoupling.ep_coupled_simulation_manager import EpLwrSimulationManager

logger = logging.getLogger(__name__)


@pytest.mark.integration
def test_real_parallel_processes_abort_barrier_on_missing_matrix_file(
    tmp_path: Path, caplog
) -> None:
    """Verify that a real background process failure due to a missing disk asset triggers

    a real barrier abort, safely unblocking surviving processes that are waiting.
    """
    # 1. Manually craft an authentic, partially complete manifest with 2 buildings
    ws = tmp_path / "hybrid_workspace"
    runs_dir = SimulationManifest.derive_runs_dir(workspace_dir=ws)
    runs_dir.mkdir(parents=True)

    building_0_state = CompiledBuildingState(
        building_id="b_healthy",
        building_index=0,
        num_surfaces=2,
        surface_index_min=0,
        surface_index_max=1,
        outdoor_surface_names=["s1", "s2"],
    )
    building_1_state = CompiledBuildingState(
        building_id="b_doomed",
        building_index=1,
        num_surfaces=2,
        surface_index_min=2,
        surface_index_max=3,
        outdoor_surface_names=["s3", "s4"],
    )

    manifest = SimulationManifest(
        workspace_dir=ws,
        epw_file_name="haifa.epw",
        num_ts_per_h=4,
        num_total_surfaces=4,
        save_resolution_matrix=False,
        compiled_buildings=[building_0_state, building_1_state],
    )

    # 2. Setup only the specific runtime directories needed to host the NumPy array
    dir_building_0 = building_0_state.get_output_dir(runs_dir=runs_dir)
    dir_building_0.mkdir()

    # Create a valid, loadable NumPy array ONLY for building 0
    dummy_matrix = np.zeros((2, 4))
    np.save(building_0_state.get_sub_mtx_path(runs_dir), dummy_matrix)

    # Note: We deliberately do NOT save a matrix file inside building 1's folder!
    # This guarantees building 1 naturally encounters a FileNotFoundError on startup.

    # 3. Instantiate the real manager using our customized data contract
    manager = EpLwrSimulationManager(manifest)

    # 4. Execute the private parallel orchestration loop directly
    with caplog.at_level(logging.INFO):
        exit_status = manager._run_lwr_coupled_simulation()

    # =====================================================================
    # VERIFICATIONS
    # =====================================================================

    # Assert #1: The real supervisor loop catches the background crash and returns code 1
    assert exit_status == 1

    # Assert #2: Check caplog to see the authentic step-by-step history trace
    logs = [record.message for record in caplog.records]

    # Confirm the main supervisor thread caught the genuine death of building process [1]
    assert any("CRITICAL: Building process [1] with ID b_doomed died" in log for log in logs)

    # Confirm the supervisor broke the barrier to rescue building process [0] from its deadlock
    assert any("Aborting synch_point_barrier to release surviving processes" in log for log in logs)
