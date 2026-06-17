"""Unit tests isolating surface boundary filters and radiative material property extraction."""

import numpy as np
from lwrepcoupling.tools.case_study.geometry_parsers import (
    derive_surface_radiative_properties,
    extract_outdoor_surfaces,
)
from lwrepcoupling.tools.ep_models.default import get_default_base_registry
from pyenergyplus.model.model import BuildingSurfaceDetailed, OutsideBoundaryCondition, SurfaceType


def test_extract_outdoor_surfaces_filtering():
    """Validates that surfaces pool filters out interior boundaries and retains outdoors names."""
    mock_surfaces_pool = {
        "wall_exposed": BuildingSurfaceDetailed(
            surface_type=SurfaceType.wall,
            construction_name="Const",
            zone_name="Zone1",
            outside_boundary_condition=OutsideBoundaryCondition.outdoors,
            number_of_vertices=4,
            vertices=[],
        ),
        "wall_partition": BuildingSurfaceDetailed(
            surface_type=SurfaceType.wall,
            construction_name="Const",
            zone_name="Zone1",
            outside_boundary_condition=OutsideBoundaryCondition.adiabatic,
            number_of_vertices=4,
            vertices=[],
        ),
    }

    outdoor_list = extract_outdoor_surfaces(mock_surfaces_pool)
    assert outdoor_list == ["wall_exposed"]


def test_derive_surface_radiative_properties_extraction():
    """
    Asserts tracking of construction outside layers to evaluate longwave emissivity and
    reflectivity vectors.
    """
    base_registry = get_default_base_registry()

    # Common_Brick is standard exterior layer for Default_Exterior_Wall_Construction
    # Let's inspect its baseline property settings
    brick_material = base_registry.material["Common_Brick"]
    expected_eps = getattr(brick_material, "thermal_absorptance", 0.9)
    expected_rho = 1.0 - expected_eps

    mock_surfaces_pool = {
        "bldg_wall_s": BuildingSurfaceDetailed(
            surface_type=SurfaceType.wall,
            construction_name="Default_Exterior_Wall_Construction",
            zone_name="Zone1",
            outside_boundary_condition=OutsideBoundaryCondition.outdoors,
            number_of_vertices=4,
            vertices=[],
        )
    }

    eps_v, rho_v, tau_v = derive_surface_radiative_properties(
        outdoor_surfaces=["bldg_wall_s"],
        surfaces_pool=mock_surfaces_pool,
        base_registry=base_registry,
    )

    assert np.isclose(eps_v[0], expected_eps)
    assert np.isclose(rho_v[0], expected_rho)
    assert np.isclose(tau_v[0], 0.0)  # Longwave transmissivity remains zero (opaque solid)
