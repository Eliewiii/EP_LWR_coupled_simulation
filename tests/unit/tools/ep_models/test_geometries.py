"""
Unit tests validating primitive surface builders and in-place coordinate translation matrix
transformations.
"""

import pytest
from lwrepcoupling.tools.ep_models.geometries import (
    create_surface,
    create_zone,
    translate_zone_geometry,
)
from pyenergyplus.model.model import (
    OutsideBoundaryCondition,
    SunExposure,
    SurfaceType,
    WindExposure,
)


def test_create_surface_instantiation():
    """
    Validates that create_surface builds a fully populated, validated BuildingSurfaceDetailed
    Pydantic object.
    """
    coords = [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 10.0, 0.0], [0.0, 10.0, 0.0]]
    surface_map = create_surface(
        name="test_wall",
        surface_type=SurfaceType.wall,
        construction_name="Concrete_Const",
        zone_name="zone_1",
        vertices_coords=coords,
        outside_boundary_condition=OutsideBoundaryCondition.outdoors,
        sun_exposure=SunExposure.sun_exposed,
        wind_exposure=WindExposure.wind_exposed,
    )

    assert "test_wall" in surface_map
    surf_obj = surface_map["test_wall"]

    assert surf_obj.surface_type == SurfaceType.wall
    assert surf_obj.number_of_vertices == 4
    if surf_obj.vertices is not None:
        assert surf_obj.vertices[1].vertex_x_coordinate == 10.0
        assert surf_obj.vertices[1].vertex_y_coordinate == 0.0
        assert surf_obj.vertices[1].vertex_z_coordinate == 0.0


def test_create_zone_instantiation():
    """
    Validates that create_zone maps origin coordinates accurately to a localized Zone container.
    """
    zone_map = create_zone("living_room", x_origin=1.5, y_origin=2.0, z_origin=0.0)

    assert "living_room" in zone_map
    zone_obj = zone_map["living_room"]
    assert zone_obj.x_origin == 1.5
    assert zone_obj.y_origin == 2.0
    assert zone_obj.z_origin == 0.0


def test_translate_zone_geometry_shifts_coordinates():
    """
    Asserts that vector translations correctly update both the zone frame and individual surface
    vertices in place.
    """
    # Setup standard mocked data mapping dictionary structure
    coords = [[0.0, 0.0, 3.0], [5.0, 0.0, 3.0], [5.0, 5.0, 3.0], [0.0, 5.0, 3.0]]
    model_data = {
        "Zone": create_zone("zone_a", x_origin=0.0, y_origin=0.0, z_origin=0.0),
        "BuildingSurfaceDetailed": create_surface(
            "roof_a", SurfaceType.roof, "Roof_Const", "zone_a", coords
        ),
    }

    # Shift globally: dx=10.0, dy=-5.0, dz=1.5
    translate_zone_geometry(model_data, zone_name="zone_a", dx=10.0, dy=-5.0, dz=1.5)

    # 1. Check Zone Origin Mutation
    zone = model_data["Zone"]["zone_a"]
    assert zone.x_origin == 10.0
    assert zone.y_origin == -5.0
    assert zone.z_origin == 1.5

    # 2. Check Structural Vertex Transformations
    surf = model_data["BuildingSurfaceDetailed"]["roof_a"]
    # Check first vertex shift: [0.0, 0.0, 3.0] -> [10.0, -5.0, 4.5]
    assert surf.vertices[0].vertex_x_coordinate == 10.0
    assert surf.vertices[0].vertex_y_coordinate == -5.0
    assert surf.vertices[0].vertex_z_coordinate == 4.5


def test_translate_zone_geometry_missing_zone_raises_key_error():
    """
    Guarantees that attempting to shift coordinates of an unregistered zone name throws
    a proper KeyError.
    """
    model_data = {"Zone": {}, "BuildingSurfaceDetailed": {}}
    with pytest.raises(KeyError, match="Zone 'unregistered_building' not found"):
        translate_zone_geometry(
            model_data, zone_name="unregistered_building", dx=1.0, dy=0.0, dz=0.0
        )
