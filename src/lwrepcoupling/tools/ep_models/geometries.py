"""Module for generic surface processing and 3D coordinate translation."""

from typing import Any, Dict, List

from pyenergyplus.model.model import (
    BuildingSurfaceDetailed,
    OutsideBoundaryCondition,
    SunExposure,
    SurfaceType,
    Vertice,
    WindExposure,
    Zone,
)


def create_surface(
    name: str,
    surface_type: SurfaceType,
    construction_name: str,
    zone_name: str,
    vertices_coords: List[List[float]],
    outside_boundary_condition: OutsideBoundaryCondition = OutsideBoundaryCondition.outdoors,
    sun_exposure: SunExposure = SunExposure.sun_exposed,
    wind_exposure: WindExposure = WindExposure.wind_exposed,
) -> Dict[str, BuildingSurfaceDetailed]:
    """
    Instantiates a Pydantic BuildingSurfaceDetailed object wrapped in a dict with its name.
    """
    # Map the list of list coordinates to the list[Vertice] structure
    vertices_list = [
        Vertice(vertex_x_coordinate=v[0], vertex_y_coordinate=v[1], vertex_z_coordinate=v[2])
        for v in vertices_coords
    ]

    # Build and validate the Pydantic model object
    surface_obj = BuildingSurfaceDetailed(
        surface_type=surface_type,
        construction_name=construction_name,
        zone_name=zone_name,
        outside_boundary_condition=outside_boundary_condition,
        sun_exposure=sun_exposure,
        wind_exposure=wind_exposure,
        number_of_vertices=len(vertices_coords),
        vertices=vertices_list,
    )

    return {name: surface_obj}


def create_zone(
    name: str, x_origin: float = 0.0, y_origin: float = 0.0, z_origin: float = 0.0
) -> Dict[str, Zone]:
    """
    Instantiates a Pydantic Zone object wrapped in a dict with its name.
    """
    zone_obj = Zone(x_origin=x_origin, y_origin=y_origin, z_origin=z_origin)
    return {name: zone_obj}


def translate_zone_geometry(
    model_data: dict[str, dict[str, Any]], zone_name: str, *, dx: float, dy: float, dz: float
) -> None:
    """
    Translates a specific zone and all its associated surfaces by (dx, dy, dz) in place.
    Modifies both the Zone origin and the individual surface vertices.
    """

    # 1. Translate the Zone Origin
    zones_collection = model_data.get("Zone", {})
    if zone_name in zones_collection:
        zone_obj = zones_collection[zone_name]

        # Guard against None values if they were defaulted in the model
        zone_obj.x_origin = (zone_obj.x_origin or 0.0) + dx
        zone_obj.y_origin = (zone_obj.y_origin or 0.0) + dy
        zone_obj.z_origin = (zone_obj.z_origin or 0.0) + dz
    else:
        raise KeyError(f"Zone '{zone_name}' not found in the provided model data.")

    # 2. Find and Translate all matching BuildingSurfaceDetailed vertices
    surfaces_collection = model_data.get("BuildingSurfaceDetailed", {})

    for _, surf_obj in surfaces_collection.items():
        # Only modify surfaces that belong to our target zone
        if surf_obj.zone_name == zone_name:
            if surf_obj.vertices:
                for vertex in surf_obj.vertices:
                    vertex.vertex_x_coordinate += dx
                    vertex.vertex_y_coordinate += dy
                    vertex.vertex_z_coordinate += dz
