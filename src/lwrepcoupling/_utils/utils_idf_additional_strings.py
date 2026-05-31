"""
Functions to add strings to the IDF file for LWR coupling.
It includes the following addons for each outdoor surface:
- one SurfaceProperty:LocalEnvironment
- one SurfaceProperty:SurroundingSurfaces (associated to the SurfaceProperty:LocalEnvironment)
  that contains:
    - an identifier
    - the name of the surface
    - a sky view factor
    - a sky temperature schedule (left empty, EP will compute it well on its own)
    - a ground view factor (if one surface only, if more detailed use the dedicated
        SurfaceProperty:GroundSurfaces)
    - a ground temperature schedule (left empty, EP will compute it well on its own, but can be used
        to force a temperature)
    - the "combined" one context surface name
    - the cumulated view factor to the context surface
    - the temperature schedule of the context surface
- one temperature schedule for the context surface (to be updated at each time step)
- one actuator for the context surface temperature schedule (to be called each time step)
"""

from typing import Optional

from .._schemas import SurfaceAddStringConfig


def generate_idfs_additional_strings(
    outdoor_surface_names: list[str],
    srd_vf_list: list[float],
    sky_vf_list: Optional[list[float]] = None,
    ground_vf_list: Optional[list[float]] = None,
) -> str:
    """Generate the additional strings to add to the IDF file for the LWR coupling.
    Consist of the generation of a SurfaceProperty:LocalEnvironment and a
    SurfaceProperty:SurroundingSurfaces for each outdoor surface, with temperature schedule for the
    surrounding surfaces to be actuated to account for the LWR.

    Args:
        outdoor_surface_names (list[str]): list of names of the outdoor surfaces of the building
        srd_vf_list (list[float]): list of the total surrounding view factors for each
            outdoor surface
        sky_vf_list (Optional[list[float]], optional): list of the sky view factors
            for each outdoor surface. Default to None
        ground_vf_list (Optional[list[float]], optional): list of the ground view factors
            for each outdoor surface. Defaults to None.

    Returns:
        str: Additional string to add to the idf

    Raises:
        ValueError : if there are inconsistency in the sum of VF on surfaces

    """
    additional_strings = ""
    for i, surface_name in enumerate(outdoor_surface_names):
        add_string_config = SurfaceAddStringConfig(
            surface_name=surface_name,
            cumulated_ext_surf_view_factor=srd_vf_list[i],
            sky_view_factor=sky_vf_list[i] if sky_vf_list else None,
            ground_view_factor=ground_vf_list[i] if ground_vf_list else None,
        )
        additional_strings += _generate_surface_lwr_idf_additional_string(add_string_config)

    return additional_strings


def _generate_surface_lwr_idf_additional_string(add_string_config: SurfaceAddStringConfig) -> str:
    """
    Generate the additional IDF string for a given surface based on the provided configuration.
    Args:
        add_string_config: Configuration object containing all necessary parameters for string
        generation.
    :return: A formatted string to be appended to the IDF file for LWR coupling
    """
    surface_name = add_string_config.surface_name

    surface_property_local_environment_name = (
        surface_name + "_locEnv"
    )  # I guess to change to be lighter
    surface_property_surrounding_surface_name = surface_name + "_SurSur"
    context_surface_name = surface_name + "_ConSur"
    context_surface_temperature_schedule = name_surrounding_surface_temperature_schedule(
        surface_name
    )

    additional_string = _write_surface_temperature_schedule(
        schedule_name=context_surface_temperature_schedule
    )
    additional_string += _write_surface_property_surrounding_surfaces(
        surface_property_surrounding_surface_name=surface_property_surrounding_surface_name,
        context_surface_name=context_surface_name,
        cumulated_view_factor=str(add_string_config.cumulated_ext_surf_view_factor),
        context_surface_temperature_schedule=context_surface_temperature_schedule,
        sky_view_factor=str(add_string_config.sky_view_factor)
        if add_string_config.sky_view_factor is not None
        else "",
        ground_view_factor=str(add_string_config.ground_view_factor)
        if add_string_config.ground_view_factor is not None
        else "",
        ground_temperature_schedule=add_string_config.ground_temperature_schedule,
    )
    additional_string += _write_surface_property_local_environment(
        surface_name=surface_name,
        surface_property_local_environment_name=surface_property_local_environment_name,
        surface_property_surrounding_surface_name=surface_property_surrounding_surface_name,
    )
    return additional_string


def name_surrounding_surface_temperature_schedule(surface_name: str) -> str:
    """
    Name the temperature schedule of the surrounding surface.
    :param surface_name: name of the surface
    :return: name of the temperature schedule
    """
    return surface_name + "_ConSur_TemSch"


def _write_surface_temperature_schedule(schedule_name: str, init_temperature: float = 20.0) -> str:
    """

    :return:
    """
    schedule_str = (
        f"\n"
        f"Schedule:Constant,\n"
        f"  {schedule_name}, !- Name\n"
        f"  Temperature, !- Schedule Type Limits Name\n"
        f"  {init_temperature};  !- Hourly Value\n"
    )
    return schedule_str


def _write_surface_property_surrounding_surfaces(
    surface_property_surrounding_surface_name: str,
    context_surface_name: str,
    cumulated_view_factor: str,
    context_surface_temperature_schedule: str,
    sky_view_factor: str = "",
    ground_view_factor: str = "",
    ground_temperature_schedule: str = "",
) -> str:
    """

    :return:
    """
    surface_property_str = (
        f"\n"
        f"SurfaceProperty:SurroundingSurfaces,\n"
        f"  {surface_property_surrounding_surface_name}, !- Name\n"
        f"  {sky_view_factor}, !- Sky View Factor\n"
        f"  , !- Sky Temperature Schedule Name\n"
        f"  {ground_view_factor}, !- Ground View Factor\n"
        f"  {ground_temperature_schedule}, !- Ground Temperature Schedule Name\n"
        f"  {context_surface_name}, !- Surrounding Surface 1 Name\n"
        f"  {cumulated_view_factor}, !- Surrounding Surface 1 View Factor\n"
        f"  {context_surface_temperature_schedule}; !- Surrounding Surface 1 Temperature "
        f"Schedule Name\n"
    )

    return surface_property_str


def _write_surface_property_local_environment(
    surface_name: str,
    surface_property_local_environment_name: str,
    surface_property_surrounding_surface_name: str,
    ground_surface_object_name: str = "",
) -> str:
    """

    :param surface_name:
    :param surface_property_surrounding_surface_name:
    :return:
    """
    surface_property_str = (
        f"\n"
        f"SurfaceProperty:LocalEnvironment,\n"
        f"  {surface_property_local_environment_name}, !- Name\n"
        f"  {surface_name}, !- Exterior Surface Name\n"
        f"  , !- Sunlit Fraction Schedule Name\n"
        f"  {surface_property_surrounding_surface_name}, !- Surrounding Surfaces Object Name\n"
        f"  , !- Outdoor Air Node Name\n"
        f"  {ground_surface_object_name}; !- Ground Surfaces Object Name\n"
    )
    return surface_property_str
