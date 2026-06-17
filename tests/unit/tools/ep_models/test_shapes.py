"""
Unit tests validating structural macro shape extrusions, boundary rules, and parallel layout
geometry allocations.
"""

from lwrepcoupling.tools.ep_models.shapes import generate_cube_zone, generate_u_shape_zone
from pyenergyplus.model.model import BuildingSurfaceDetailed, OutsideBoundaryCondition, SurfaceType


def test_generate_cube_zone_enclosure_count_and_boundary_conditions():
    """Validates that a generated cube contains exactly 6 surfaces with compliant boundary rules."""
    cube_data = generate_cube_zone(
        zone_name="cube_bldg",
        width=10.0,
        length=12.0,
        height=3.5,
        wall_construction="Wall_Assembly",
        roof_construction="Roof_Assembly",
        floor_construction="Floor_Assembly",
    )

    assert "Zone" in cube_data
    assert "BuildingSurfaceDetailed" in cube_data
    assert "cube_bldg" in cube_data["Zone"]

    surfaces = cube_data["BuildingSurfaceDetailed"]
    assert len(surfaces) == 6

    # Verify specialized bottom ground surface interface contract
    floor = surfaces["cube_bldg_floor"]
    assert isinstance(floor, BuildingSurfaceDetailed)
    assert floor.surface_type == SurfaceType.floor
    assert floor.outside_boundary_condition == OutsideBoundaryCondition.ground
    assert floor.construction_name == "Floor_Assembly"

    # Verify standard atmospheric vertical exposure element contract
    wall_e = surfaces["cube_bldg_wall_e"]
    assert isinstance(wall_e, BuildingSurfaceDetailed)
    assert wall_e.surface_type == SurfaceType.wall
    assert wall_e.outside_boundary_condition == OutsideBoundaryCondition.outdoors
    assert wall_e.construction_name == "Wall_Assembly"


def test_generate_u_shape_zone_polyline_facades():
    """
    Validates the extrusion of a single-zone U-shaped footprint, producing exactly 10 discrete
    surfaces.
    """
    u_data = generate_u_shape_zone(
        zone_name="courtyard_core",
        w_total=20.0,
        l_total=20.0,
        w_wing=5.0,
        l_courtyard=10.0,
        height=4.0,
    )

    surfaces = u_data["BuildingSurfaceDetailed"]

    # 1 Slab Floor + 1 Ceiling Roof + 8 Perimeter Footprint Facade Walls = 10 Surfaces Total
    assert len(surfaces) == 10

    # Ensure all 8 wall keys are generated deterministically via tracking loops
    for i in range(8):
        surface_key = f"courtyard_core_wall_{i}"
        assert surface_key in surfaces
        surface = surfaces[surface_key]
        assert isinstance(surface, BuildingSurfaceDetailed)
