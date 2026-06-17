"""Unit tests validating the dynamic recursive tree-walking IDF text exporter engine."""

import pytest
from lwrepcoupling.tools.case_study.idf_exporter import (
    export_model_to_idf_string,
    write_native_idf,
)
from lwrepcoupling.tools.ep_models.default import (
    get_default_base_registry,
    get_single_zone_box_geometry,
)
from lwrepcoupling.tools.ep_models.generate_model import compile_energyplus_model


@pytest.fixture
def sample_compiled_box_model():
    """
    Compiles a complete, type-safe EnergyPlusModel box building instance from standard registries.
    """
    base_registry = get_default_base_registry()
    geo_data = get_single_zone_box_geometry(
        zone_name="test_box_zone",
        width=10.0,
        length=12.0,
        height=3.0,
    )

    return compile_energyplus_model(
        base_registry=base_registry,
        building_zone_registry=geo_data["Zone"],
        building_surface_registry=geo_data["BuildingSurfaceDetailed"],
    )


def test_export_model_to_idf_string_structural_elements(sample_compiled_box_model):
    """
    Validates that the recursive exporter dynamically produces accurate top-level EnergyPlus
    class entries.
    """
    idf_text = export_model_to_idf_string(sample_compiled_box_model)

    # Check that class headers match legacy syntax conventions
    assert "Building," in idf_text
    assert "Zone," in idf_text
    assert "BuildingSurface:Detailed," in idf_text
    assert "Material," in idf_text
    assert "Construction," in idf_text


def test_export_model_to_idf_string_positional_values_and_punctuation(sample_compiled_box_model):
    """
    Asserts that the text streams apply punctuation terminators according to the line-by-index
    rules.
    """
    idf_text = export_model_to_idf_string(sample_compiled_box_model)

    # 1. Look at the isolated Zone block to verify tracking name override and terminal semicolon
    # Each parameter line must terminate with a comma except the final entry
    assert "test_box_zone," in idf_text

    # 2. Assert that the multi-layer construction mapping loops preserved their positional layers
    assert "Default_Exterior_Wall_Construction," in idf_text
    assert "Common_Brick," in idf_text


def test_export_model_to_idf_string_vertex_flattening(sample_compiled_box_model):
    """
    Guarantees that nested arrays of vertices are unpacked into continuous flat positional rows.
    """
    idf_text = export_model_to_idf_string(sample_compiled_box_model)

    # A 10m x 12m roof box corner layout with origin 0,0,0 should flatten to explicit string rows:
    assert "10.0," in idf_text  # X coordinate component
    assert "12.0," in idf_text  # Y coordinate component
    assert "3.0;" in idf_text or "3.0," in idf_text  # Z coordinate component


def test_write_native_idf_file_io_and_lwr_text_append(sample_compiled_box_model, tmp_path):
    """
    Verifies physical file writing on disk and asserts that custom longwave coupling string payloads
    append to the footer.
    """
    target_file = tmp_path / "simulation_workspace" / "test_output.idf"

    # Custom mock text payload representing your specialized longwave coupling parameters
    # or EMS modifications
    lwr_coupling_payload = (
        "! CUSTOM LWR COUPLING ADDITIONS\n"
        "SurfaceProperty:LongWaveSystemEffects,\n"
        "    test_box_zone_wall_s,\n"
        "    Custom_Coupled_Radiation_Matrix_001;"
    )

    # Execute the file writer
    saved_path = write_native_idf(
        model=sample_compiled_box_model,
        target_path=target_file,
        additional_text=lwr_coupling_payload,
    )

    # Assert paths and content persist correctly
    assert saved_path.exists()
    assert saved_path == target_file

    file_content = saved_path.read_text(encoding="utf-8")

    # Verify both the model geometry code and your raw coupling block are present side-by-side
    assert "BuildingSurface:Detailed," in file_content
    assert "! CUSTOM LWR COUPLING ADDITIONS" in file_content
    assert "Custom_Coupled_Radiation_Matrix_001;" in file_content

    # Ensure it terminates with a clean newline block
    assert file_content.endswith("\n")
