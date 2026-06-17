"""Unit tests isolating native file serialization and scientific matrix writing operations."""

import numpy as np
from lwrepcoupling.tools.case_study.io_handlers import (
    save_and_convert_to_idf,
    serialize_numpy_matrices,
)
from lwrepcoupling.tools.ep_models.default import (
    get_default_base_registry,
    get_single_zone_box_geometry,
)
from lwrepcoupling.tools.ep_models.generate_model import compile_energyplus_model


def test_serialize_numpy_matrices_saving_loops(tmp_path):
    """Verifies that numerical tracking arrays are accurately mapped to unique storage endpoints."""
    vf = np.eye(3)
    eps = np.array([0.9, 0.9, 0.9])
    rho = np.array([0.1, 0.1, 0.1])
    tau = np.zeros(3)

    paths = serialize_numpy_matrices(
        target_dir=tmp_path,
        prefix="test_case",
        vf_matrix=vf,
        eps_vector=eps,
        rho_vector=rho,
        tau_vector=tau,
    )

    assert paths["vf"].exists()
    assert paths["eps"].exists()
    assert paths["rho"].exists()
    assert paths["tau"].exists()

    loaded_vf = np.load(str(paths["vf"]))
    assert np.array_equal(loaded_vf, vf)


def test_save_and_convert_to_idf_native_generation(tmp_path):
    """
    Asserts that model records are compiled cleanly into physical text assets on disk without
    shell dependencies.
    """
    base_registry = get_default_base_registry()
    geo_data = get_single_zone_box_geometry(zone_name="io_box")
    ep_model = compile_energyplus_model(
        base_registry=base_registry,
        building_zone_registry=geo_data["Zone"],
        building_surface_registry=geo_data["BuildingSurfaceDetailed"],
    )

    # Execute native writer
    idf_out_path = save_and_convert_to_idf(
        ep_model=ep_model,
        target_dir=tmp_path,
        base_filename="bldg_io_test",
        lwr_coupling_text="! APPENDED COUPLING DATA",
    )

    assert idf_out_path == tmp_path / "bldg_io_test.idf"
    assert idf_out_path.exists()

    content = idf_out_path.read_text(encoding="utf-8")
    assert "BuildingSurface:Detailed," in content
    assert "! APPENDED COUPLING DATA" in content
