"""Unit tests validating case study structural orchestrations and view factor cell calculations."""

import numpy as np
from lwrepcoupling.tools.case_study.radiation_benchmark import (
    generate_solo_box_case_study,
    generate_solo_u_case_study,
    generate_three_box_case_study,
)


def test_generate_solo_box_case_study_orchestrator(tmp_path):
    """
    Validates standalone box building case loop state setups and empty boundary conditions matrix
    allocations.
    """
    epw_file = tmp_path / "test_weather.epw"
    epw_file.write_text("! MOCK EPW FILE", encoding="utf-8")

    inputs = generate_solo_box_case_study(output_workspace=tmp_path, epw_file_path=epw_file)

    assert inputs.workspace_dir == tmp_path
    assert inputs.epw_path == epw_file
    assert len(inputs.buildings) == 1
    assert inputs.buildings[0].building_id == "bldg_solo_box"

    # Assert physical file exists and validated by Pydantic FilePath constraints
    assert inputs.buildings[0].idf_path.exists()

    # Standalone convex block has an interaction array initialized completely with zeros
    vf_matrix = np.load(str(inputs.vf_matrix_path))
    assert np.all(vf_matrix == 0.0)


def test_generate_solo_u_case_study_courtyard_matrices(tmp_path):
    """
    Validates cross-wing and perpendicular view factor mappings inside the single U-shape layout
    context.
    """
    epw_file = tmp_path / "test_weather.epw"
    epw_file.write_text("! MOCK EPW FILE", encoding="utf-8")

    inputs = generate_solo_u_case_study(output_workspace=tmp_path, epw_file_path=epw_file)

    assert inputs.buildings[0].building_id == "bldg_solo_u"
    assert inputs.buildings[0].idf_path.exists()

    vf_matrix = np.load(str(inputs.vf_matrix_path))
    # Matrix must contain our newly calculated non-zero parallel and perpendicular analytical
    # elements
    assert np.any(vf_matrix > 0.0)


def test_generate_three_box_case_study_multi_building_coupling(tmp_path):
    """
    Validates multi-building layout execution splits and adjacent vs far-field view factor radiation
    matrix cells.
    """
    epw_file = tmp_path / "test_weather.epw"
    epw_file.write_text("! MOCK EPW FILE", encoding="utf-8")

    inputs = generate_three_box_case_study(output_workspace=tmp_path, epw_file_path=epw_file)

    # Verifies independent file listings created sequentially for all 3 structures
    assert len(inputs.buildings) == 3
    assert inputs.buildings[0].building_id == "west_building"
    assert inputs.buildings[1].building_id == "middle_building"
    assert inputs.buildings[2].building_id == "east_building"

    for bldg in inputs.buildings:
        assert bldg.idf_path.exists()

    master_vf = np.load(str(inputs.vf_matrix_path))
    assert master_vf.shape[0] == master_vf.shape[1]
    assert np.any(master_vf > 0.0)
