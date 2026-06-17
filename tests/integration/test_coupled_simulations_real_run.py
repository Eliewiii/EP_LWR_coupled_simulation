"""Integration system tests executing authentic, un-mocked parallel EnergyPlus simulation loops

across the three baseline case study configurations.
"""

from pathlib import Path

import pytest
from lwrepcoupling.ep_coupled_simulation_manager import EpLwrSimulationManager
from lwrepcoupling.tools.case_study.radiation_benchmark import (
    generate_solo_box_case_study,
    generate_solo_u_case_study,
    generate_three_box_case_study,
)

# Strictly isolate these heavy runtime engine tests under the system and simulation markers
pytestmark = [pytest.mark.simulation]

# Resolve the project root dynamically to point to your repository's sample weather file
# Resolve the project root dynamically (parents[2] hits the repo root 'ep_lwr')
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_EPW_PATH = PROJECT_ROOT / "tests" / "data" / "weather_sample.epw"


@pytest.fixture
def real_weather_file():
    """
    Ensures the sample environmental profile is present before initiating un-mocked solver threads.
    """
    if not SAMPLE_EPW_PATH.exists():
        pytest.skip(
            f"Authentic system run skipped: Weather asset missing at '{SAMPLE_EPW_PATH}'. "
            f"Verify that the sample weather file exists inside your repository's tests/data/ "
            f"directory."
        )
    return SAMPLE_EPW_PATH


# =====================================================================
# UN-MOCKED MULTI-PROCESS ENGINE INTEGRATION RUNNERS
# =====================================================================


def test_unmocked_system_solo_box_simulation(tmp_path: Path, real_weather_file: Path) -> None:
    """Compiles and runs a real parallel simulation loop for Configuration 1 (Standalone Box)."""
    input_workspace = tmp_path / "real_solo_box_inputs"
    compilation_sandbox = tmp_path / "real_solo_box_workspace"

    # 1. Orchestrate case configurations natively (writes real structural assets to disk)
    sim_inputs = generate_solo_box_case_study(
        output_workspace=input_workspace,
        epw_file_path=real_weather_file,
        width=10.0,
        length=10.0,
        height=3.0,
    )

    # Enforce directory separation to pass the manager's strict exclusivity checks
    sim_inputs.workspace_dir = compilation_sandbox

    # 2. Build the unified simulation workspace manifest layout
    manager = EpLwrSimulationManager.compile_and_initialize_workspace(
        inputs=sim_inputs, overwrite=True
    )

    # 3. Fire up the actual multiprocess runtime worker engine
    # Bypasses all mocks—spawns an authentic child process running pyenergyplus C++ loops
    exit_code = manager._run_lwr_coupled_simulation()

    # Assert that the engine executed, synchronized, and completed with full success
    assert exit_code == 0

    # Confirm physical simulation files were actually dropped in the sandbox runs directory
    assert (compilation_sandbox / "runs" / "b_0" / "eplusout.err").exists()


def test_unmocked_system_solo_u_shape_simulation(tmp_path: Path, real_weather_file: Path) -> None:
    """Compiles and runs a real parallel simulation loop for Configuration 2 (U-Shape Courtyard).

    Verifies the real-time matrix slice lookups handle shared memory concurrently without crashing.
    """
    input_workspace = tmp_path / "real_solo_u_inputs"
    compilation_sandbox = tmp_path / "real_solo_u_workspace"

    sim_inputs = generate_solo_u_case_study(
        output_workspace=input_workspace, epw_file_path=real_weather_file
    )

    sim_inputs.workspace_dir = compilation_sandbox

    manager = EpLwrSimulationManager.compile_and_initialize_workspace(
        inputs=sim_inputs, overwrite=True
    )

    exit_code = manager._run_lwr_coupled_simulation()
    assert exit_code == 0
    assert (compilation_sandbox / "runs" / "b_0" / "eplusout.err").exists()


def test_unmocked_system_three_box_simulation(tmp_path: Path, real_weather_file: Path) -> None:
    """
    Compiles and runs a real parallel simulation loop for Configuration 3
    (Three Parallel Buildings).

    Enforces that three entirely separate process contexts can simultaneously share the OS
    memory segments (`SharedMemory`) and accurately trip the twin synchronizer barriers.
    """
    input_workspace = tmp_path / "real_three_box_inputs"
    compilation_sandbox = tmp_path / "real_three_box_workspace"

    sim_inputs = generate_three_box_case_study(
        output_workspace=input_workspace, epw_file_path=real_weather_file, separation_distance=5.0
    )

    sim_inputs.workspace_dir = compilation_sandbox

    manager = EpLwrSimulationManager.compile_and_initialize_workspace(
        inputs=sim_inputs, overwrite=True
    )

    # Ensure all three separate sandbox run profiles are fully compiled and configured
    assert manager.num_buildings == 3

    exit_code = manager._run_lwr_coupled_simulation()
    assert exit_code == 0

    # Cross-verify that individual process sandboxes generated distinct file endpoints
    for b_idx in range(3):
        err_log_path = compilation_sandbox / "runs" / f"b_{b_idx}" / "eplusout.err"
        assert err_log_path.exists()
