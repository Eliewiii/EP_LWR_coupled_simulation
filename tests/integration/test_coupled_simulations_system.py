"""Integration system tests validating workspace compilation and parallel process lifecycles

across the three target urban radiation geometry configurations.
"""

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from lwrepcoupling.ep_coupled_simulation_manager import EpLwrSimulationManager
from lwrepcoupling.tools.case_study.radiation_benchmark import (
    generate_solo_box_case_study,
    generate_solo_u_case_study,
    generate_three_box_case_study,
)

logger = logging.getLogger(__name__)

# Enforce system and simulation categorization for this entire test module
pytestmark = [pytest.mark.system]


@pytest.fixture
def dummy_epw_path(tmp_path: Path) -> Path:
    """Creates a mock weather asset to fulfill path verification constraints."""
    epw_file = tmp_path / "mock_haifa_climate.epw"
    epw_file.write_text(
        "! Fake weather records payload for simulation validation", encoding="utf-8"
    )
    return epw_file


# =====================================================================
# SYSTEM CONFIGURATION INTEGRATION RUNNERS (ISOLATED RUN DIRECTORIES)
# =====================================================================


@patch("lwrepcoupling.tools.ep_models.generate_model.EnergyPlusAPI")
def test_system_configuration_1_solo_box_execution(
    mock_api, tmp_path: Path, dummy_epw_path: Path
) -> None:
    """Validates the execution loop for Configuration 1 (Solo Box Building).

    Ensures the orchestrator input workspace and compiler output sandbox remain isolated.
    """
    # Create distinct folders to prevent exclusivity perimeter crashes
    input_workspace = tmp_path / "config_1_inputs"
    compilation_sandbox = tmp_path / "config_1_compiled_workspace"

    # 1. Invoke high-level orchestration builder to dump initial geometries and arrays
    sim_inputs = generate_solo_box_case_study(
        output_workspace=input_workspace, epw_file_path=dummy_epw_path
    )

    # Point the compilation target vector explicitly to a pristine, untouched folder
    sim_inputs.workspace_dir = compilation_sandbox

    # 2. Compile the production workspace structure cleanly
    manager = EpLwrSimulationManager.compile_and_initialize_workspace(
        inputs=sim_inputs, overwrite=True
    )

    # 3. Execute parallel worker orchestration loop
    with patch("lwrepcoupling._ep_simulation_instance.EnergyPlusAPI") as mock_worker_api:
        mock_instance = MagicMock()
        mock_instance.runtime.run_energyplus.return_value = 0
        mock_worker_api.return_value = mock_instance

        exit_code = manager._run_lwr_coupled_simulation()
        assert exit_code == 0


@patch("lwrepcoupling.tools.ep_models.generate_model.EnergyPlusAPI")
def test_system_configuration_2_solo_u_shape_execution(
    mock_api, tmp_path: Path, dummy_epw_path: Path
) -> None:
    """Validates the execution loop for Configuration 2 (Solo U-Shape Courtyard Building)."""
    input_workspace = tmp_path / "config_2_inputs"
    compilation_sandbox = tmp_path / "config_2_compiled_workspace"

    sim_inputs = generate_solo_u_case_study(
        output_workspace=input_workspace, epw_file_path=dummy_epw_path
    )
    sim_inputs.workspace_dir = compilation_sandbox

    manager = EpLwrSimulationManager.compile_and_initialize_workspace(
        inputs=sim_inputs, overwrite=True
    )

    with patch("lwrepcoupling._ep_simulation_instance.EnergyPlusAPI") as mock_worker_api:
        mock_instance = MagicMock()
        mock_instance.runtime.run_energyplus.return_value = 0
        mock_worker_api.return_value = mock_instance

        exit_code = manager._run_lwr_coupled_simulation()
        assert exit_code == 0


@patch("lwrepcoupling.tools.ep_models.generate_model.EnergyPlusAPI")
def test_system_configuration_3_three_box_buildings_execution(
    mock_api, tmp_path: Path, dummy_epw_path: Path
) -> None:
    """Validates the execution loop for Configuration 3 (Three Parallel Box Buildings)."""
    input_workspace = tmp_path / "config_3_inputs"
    compilation_sandbox = tmp_path / "config_3_compiled_workspace"

    sim_inputs = generate_three_box_case_study(
        output_workspace=input_workspace, epw_file_path=dummy_epw_path
    )
    sim_inputs.workspace_dir = compilation_sandbox

    manager = EpLwrSimulationManager.compile_and_initialize_workspace(
        inputs=sim_inputs, overwrite=True
    )
    assert manager.num_buildings == 3

    with patch("lwrepcoupling._ep_simulation_instance.EnergyPlusAPI") as mock_worker_api:
        mock_instance = MagicMock()
        mock_instance.runtime.run_energyplus.return_value = 0
        mock_worker_api.return_value = mock_instance

        exit_code = manager._run_lwr_coupled_simulation()
        assert exit_code == 0
