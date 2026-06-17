"""Module for compiling geometric shapes using localized coordinate offsets."""

from pyenergyplus.model.model import (
    BuildingSurfaceDetailed,
    OutsideBoundaryCondition,
    SunExposure,
    SurfaceType,
    WindExposure,
    Zone,
)

from .geometries import create_surface, create_zone


def generate_cube_zone(
    zone_name: str,
    width: float,
    length: float,
    height: float,
    *,
    wall_construction: str = "Generic_Wall_Construction",
    roof_construction: str = "Generic_Roof_Construction",
    floor_construction: str = "Generic_Floor_Construction",
    origin: list[float] | None = None,
) -> dict[str, dict[str, Zone] | dict[str, BuildingSurfaceDetailed]]:
    """Generates a single zone box with specific constructions for walls, roof, and floor."""
    if origin is None:
        origin = [0.0, 0.0, 0.0]
    x0, y0, z0 = origin
    x1, y1, z1 = x0 + width, y0 + length, z0 + height

    model_data = {
        "Zone": create_zone(zone_name, x_origin=x0, y_origin=y0, z_origin=z0),
        "BuildingSurfaceDetailed": {},
    }

    # Added the specific construction variables directly into the mapped structural specs
    surfaces_definition = {
        f"{zone_name}_floor": (
            SurfaceType.floor,
            floor_construction,
            [[x0, y1, z0], [x0, y0, z0], [x1, y0, z0], [x1, y1, z0]],
            OutsideBoundaryCondition.ground,
            SunExposure.no_sun,
            WindExposure.no_wind,
        ),
        f"{zone_name}_roof": (
            SurfaceType.roof,
            roof_construction,
            [[x0, y0, z1], [x0, y1, z1], [x1, y1, z1], [x1, y0, z1]],
            OutsideBoundaryCondition.outdoors,
            SunExposure.sun_exposed,
            WindExposure.wind_exposed,
        ),
        f"{zone_name}_wall_s": (
            SurfaceType.wall,
            wall_construction,
            [[x0, y0, z1], [x1, y0, z1], [x1, y0, z0], [x0, y0, z0]],
            OutsideBoundaryCondition.outdoors,
            SunExposure.sun_exposed,
            WindExposure.wind_exposed,
        ),
        f"{zone_name}_wall_e": (
            SurfaceType.wall,
            wall_construction,
            [[x1, y0, z1], [x1, y1, z1], [x1, y1, z0], [x1, y0, z0]],
            OutsideBoundaryCondition.outdoors,
            SunExposure.sun_exposed,
            WindExposure.wind_exposed,
        ),
        f"{zone_name}_wall_n": (
            SurfaceType.wall,
            wall_construction,
            [[x1, y1, z1], [x0, y1, z1], [x0, y1, z0], [x1, y1, z0]],
            OutsideBoundaryCondition.outdoors,
            SunExposure.sun_exposed,
            WindExposure.wind_exposed,
        ),
        f"{zone_name}_wall_w": (
            SurfaceType.wall,
            wall_construction,
            [[x0, y1, z1], [x0, y0, z1], [x0, y0, z0], [x0, y1, z0]],
            OutsideBoundaryCondition.outdoors,
            SunExposure.sun_exposed,
            WindExposure.wind_exposed,
        ),
    }

    for s_name, (s_type, s_const, verts, bc, sun, wind) in surfaces_definition.items():
        surf_dict = create_surface(
            name=s_name,
            surface_type=s_type,
            construction_name=s_const,
            zone_name=zone_name,
            vertices_coords=verts,
            outside_boundary_condition=bc,
            sun_exposure=sun,
            wind_exposure=wind,
        )
        model_data["BuildingSurfaceDetailed"].update(surf_dict)

    return model_data


def generate_u_shape_zone(
    zone_name: str,
    w_total: float,
    l_total: float,
    w_wing: float,
    l_courtyard: float,
    height: float,
    *,
    wall_construction: str = "Generic_Wall_Construction",
    roof_construction: str = "Generic_Roof_Construction",
    floor_construction: str = "Generic_Floor_Construction",
    origin: list[float] | None = None,
) -> dict[str, dict[str, Zone] | dict[str, BuildingSurfaceDetailed]]:
    """
    Generates a U-shaped building zone envelope with separate envelope construction definitions.
    """
    if origin is None:
        origin = [0.0, 0.0, 0.0]
    x0, y0, z0 = origin
    z1 = z0 + height

    # 2D Polylines counter-clockwise loop mapping
    p = [
        [x0, y0],
        [x0 + w_total, y0],
        [x0 + w_total, y0 + l_total],
        [x0 + w_total - w_wing, y0 + l_total],
        [x0 + w_total - w_wing, y0 + l_total - l_courtyard],
        [x0 + w_wing, y0 + l_total - l_courtyard],
        [x0 + w_wing, y0 + l_total],
        [x0, y0 + l_total],
    ]

    model_data = {
        "Zone": create_zone(zone_name, x_origin=x0, y_origin=y0, z_origin=z0),
        "BuildingSurfaceDetailed": {},
    }

    # Floor (Uses floor_construction)
    floor_verts = [[pt[0], pt[1], z0] for pt in reversed(p)]
    model_data["BuildingSurfaceDetailed"].update(
        create_surface(
            name=f"{zone_name}_floor",
            surface_type=SurfaceType.floor,
            construction_name=floor_construction,
            zone_name=zone_name,
            vertices_coords=floor_verts,
            outside_boundary_condition=OutsideBoundaryCondition.ground,
            sun_exposure=SunExposure.no_sun,
            wind_exposure=WindExposure.no_wind,
        )
    )

    # Roof (Uses roof_construction)
    roof_verts = [[pt[0], pt[1], z1] for pt in p]
    model_data["BuildingSurfaceDetailed"].update(
        create_surface(
            name=f"{zone_name}_roof",
            surface_type=SurfaceType.roof,
            construction_name=roof_construction,
            zone_name=zone_name,
            vertices_coords=roof_verts,
            outside_boundary_condition=OutsideBoundaryCondition.outdoors,
            sun_exposure=SunExposure.sun_exposed,
            wind_exposure=WindExposure.wind_exposed,
        )
    )

    # 8 Extruded Vertical Walls (Uses wall_construction)
    for i in range(8):
        pt_start = p[i]
        pt_end = p[(i + 1) % 8]

        wall_verts = [
            [pt_start[0], pt_start[1], z1],
            [pt_end[0], pt_end[1], z1],
            [pt_end[0], pt_end[1], z0],
            [pt_start[0], pt_start[1], z0],
        ]

        model_data["BuildingSurfaceDetailed"].update(
            create_surface(
                name=f"{zone_name}_wall_{i}",
                surface_type=SurfaceType.wall,
                construction_name=wall_construction,
                zone_name=zone_name,
                vertices_coords=wall_verts,
                outside_boundary_condition=OutsideBoundaryCondition.outdoors,
                sun_exposure=SunExposure.sun_exposed,
                wind_exposure=WindExposure.wind_exposed,
            )
        )

    return model_data
