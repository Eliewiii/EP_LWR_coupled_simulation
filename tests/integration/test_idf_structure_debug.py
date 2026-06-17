"""Integration debug suite validating raw text compliance of the recursive IDF exporter

and caching structural assets to disk for manual inspection.
"""

from pathlib import Path

import pytest
from lwrepcoupling.ep_coupled_simulation_manager import EpLwrSimulationManager
from lwrepcoupling.tools.case_study.radiation_benchmark import (
    generate_solo_box_case_study,
    generate_solo_u_case_study,
    generate_three_box_case_study,
)

# Apply standard tagging layers
pytestmark = [pytest.mark.debug]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEBUG_OUTPUT_DIR = PROJECT_ROOT / ".debug_artifacts"
SAMPLE_EPW_PATH = PROJECT_ROOT / "tests" / "data" / "weather_sample.epw"


@pytest.fixture(scope="module", autouse=True)
def setup_debug_environment():
    """Initializes a persistent local directory for caching generated test IDFs."""
    DEBUG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    yield


@pytest.fixture
def real_weather_file():
    """Ensures weather assets exist prior to executing layout generation."""
    if not SAMPLE_EPW_PATH.exists():
        pytest.skip("Sample weather file missing.")
    return SAMPLE_EPW_PATH


def test_debug_export_solo_box_structure(tmp_path, real_weather_file):
    """Generates and evaluates the standalone box building text alignment rules."""
    # Write to a subfolder within tmp_path for the orchestrator input step
    inputs = generate_solo_box_case_study(
        output_workspace=tmp_path / "inputs", epw_file_path=real_weather_file
    )
    generated_idf = inputs.buildings[0].idf_path

    # Persistent cache copy step for manual inspection
    debug_destination = DEBUG_OUTPUT_DIR / "debug_solo_box.idf"
    debug_destination.write_text(generated_idf.read_text(encoding="utf-8"), encoding="utf-8")

    content = debug_destination.read_text(encoding="utf-8")

    # ------------------------------------------------=================
    # STRUCTURAL COMPLIANCE ASSERTIONS
    # ------------------------------------------------=================
    # Check 1: Verify the file doesn't contain un-wrapped Pydantic object syntax strings
    assert "BaseModel" not in content
    assert "__dict__" not in content

    # Check 2: Verify that every declared block ends with a terminal semicolon
    # Find all instances of a class definition and ensure a semicolon terminates the block
    blocks = content.split("\n\n")
    for block in blocks:
        cleaned = block.strip()
        if cleaned and not cleaned.startswith("!"):
            # The last non-comment character before the block end must be a semicolon
            assert ";" in cleaned, f"Block missing terminal semicolon:\n{cleaned}"


def test_debug_export_solo_u_shape_vertices(tmp_path, real_weather_file):
    """Validates vertex flattening logic inside the solo U-shape courtyard context."""
    inputs = generate_solo_u_case_study(
        output_workspace=tmp_path / "inputs", epw_file_path=real_weather_file
    )
    generated_idf = inputs.buildings[0].idf_path

    debug_destination = DEBUG_OUTPUT_DIR / "debug_solo_u.idf"
    debug_destination.write_text(generated_idf.read_text(encoding="utf-8"), encoding="utf-8")

    content = debug_destination.read_text(encoding="utf-8")

    # Check 3: Verify that the class name mapping logic translated camelCase correctly
    assert "BuildingSurfaceDetailed" not in content
    assert "BuildingSurface:Detailed" in content

    # Check 4: Verify coordinate formatting for vertex boundaries
    # Ensure lines containing vertices don't accidentally preserve dictionary braces like
    # {"vertex_x_coordinate": ...}
    assert "vertex_x_coordinate" not in content
    assert "{" not in content


def test_debug_export_three_box_alignment(tmp_path, real_weather_file):
    """Caches and validates all three parallel building blocks side-by-side."""
    inputs = generate_three_box_case_study(
        output_workspace=tmp_path / "inputs",
        epw_file_path=real_weather_file,
        separation_distance=6.0,
    )

    for idx, bldg in enumerate(inputs.buildings):
        debug_destination = DEBUG_OUTPUT_DIR / f"debug_three_box_bldg_{idx}_{bldg.building_id}.idf"
        debug_destination.write_text(bldg.idf_path.read_text(encoding="utf-8"), encoding="utf-8")

        content = debug_destination.read_text(encoding="utf-8")
        assert "Zone," in content
        assert bldg.building_id in content


# =====================================================================
# PERSISTENT UN-MOCKED SYSTEM RUNS FOR LOG ANALYSIS
# =====================================================================


def test_debug_real_run_solo_box(real_weather_file: Path) -> None:
    """Executes an un-mocked standalone box simulation inside the persistent debug folder."""
    input_workspace = DEBUG_OUTPUT_DIR / "real_solo_box_inputs"
    compilation_sandbox = DEBUG_OUTPUT_DIR / "real_solo_box_workspace"

    # 1. Orchestrate case configurations natively
    sim_inputs = generate_solo_box_case_study(
        output_workspace=input_workspace,
        epw_file_path=real_weather_file,
        width=10.0,
        length=10.0,
        height=3.0,
    )
    sim_inputs.workspace_dir = compilation_sandbox

    # 2. Compile the production workspace structure
    manager = EpLwrSimulationManager.compile_and_initialize_workspace(
        inputs=sim_inputs, overwrite=True
    )

    # 3. Fire up the un-mocked parallel engine loop
    exit_code = manager._run_lwr_coupled_simulation()

    # We assert exit_code == 0 so pytest alerts us, but the files stay behind for inspection!
    assert exit_code == 0
    assert (compilation_sandbox / "runs" / "b_0" / "eplusout.err").exists()


def test_debug_real_run_solo_u_shape(real_weather_file: Path) -> None:
    """Executes an un-mocked U-shape simulation inside the persistent debug folder."""
    input_workspace = DEBUG_OUTPUT_DIR / "real_solo_u_inputs"
    compilation_sandbox = DEBUG_OUTPUT_DIR / "real_solo_u_workspace"

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


def test_debug_real_run_three_box(real_weather_file: Path) -> None:
    """Executes an un-mocked parallel 3-building loop inside the persistent debug folder."""
    input_workspace = DEBUG_OUTPUT_DIR / "real_three_box_inputs"
    compilation_sandbox = DEBUG_OUTPUT_DIR / "real_three_box_workspace"

    sim_inputs = generate_three_box_case_study(
        output_workspace=input_workspace, epw_file_path=real_weather_file, separation_distance=5.0
    )
    sim_inputs.workspace_dir = compilation_sandbox

    manager = EpLwrSimulationManager.compile_and_initialize_workspace(
        inputs=sim_inputs, overwrite=True
    )

    exit_code = manager._run_lwr_coupled_simulation()
    assert exit_code == 0

    for b_idx in range(3):
        err_log_path = compilation_sandbox / "runs" / f"b_{b_idx}" / "eplusout.err"
        assert err_log_path.exists()
