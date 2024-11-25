"""
Functions to add strings to the IDF file for LWR coupling.
It includes the following addons for each outdoor surface:
- one SurfaceProperty:LocalEnvironment
- one SurfaceProperty:SurroundingSurfaces (associated to the SurfaceProperty:LocalEnvironment) that contains:
    - an identifier
    - the name of the surface
    - a sky view factor
    - a sky temperature schedule (left empty, EP will compute it well on its own)
    - a ground view factor (if one surface only, if more detailed us the dedicated SurfaceProperty:GroundSurfaces)
    - a ground temperature schedule (left empty, EP will compute it well on its own, but can be used to force a temperature)
    - the "combined" one conetxt surface name
    - the cumulated view factor to the context surface
    - the temperature schedule of the context surface
- one temperature schedule for the conetxt surface (to be updated at each time step)
- one actuator for the context surface temperature schedule (to be called each time step)
"""


def add_additional_strings_to_idf(idf_string: str, path_idf: str) -> str:
    """

    :param idf_string:
    :param path_idf:
    :return:
    """


def generate_surface_lwr_idf_additional_string(surface_name: str,
                                               cumulated_ext_surf_view_factor: float,
                                               sky_view_factor: float = '',
                                               ground_view_factor: float = '',
                                               ground_temperature_schedule: str = '') -> str:
    """

    :param surface_name:
    :param sky_view_factor:
    :param ground_view_factor: View factor to the ground, if not used, set to 0.
    :param cumulated_ext_surf_view_factor: cumulated view factors of all the surrounding surfaces.
    :param ground_temperature_schedule: schedule name for the ground temperature, to be left empty if not used.
        If used, it will be common to all surfaces, and thus created once, outside of this function.
        For more detailed ground temperature, use the dedicated SurfaceProperty:GroundSurfaces, with multiple surfaces
        and schedules, or one ground surface for each surface, with averaged and weighted temperature the same way as
        regular context surface. Not implemented here as no ground temperature model was developed.
    :return:
    """

    surface_property_local_environment_name = surface_name + "_locEnv"  # I guess to change to be lighter
    surface_property_surrounding_surface_name = surface_name + "_SurSur"
    context_surface_name = surface_name + "_ConSur"
    context_surface_temperature_schedule = name_surrounding_surface_temperature_schedule(surface_name)

    additional_string = _write_surface_temperature_schedule(
        schedule_name=context_surface_temperature_schedule)
    additional_string += _write_surface_property_surrounding_surfaces(
        surface_property_surrounding_surface_name=surface_property_surrounding_surface_name,
        context_surface_name=context_surface_name,
        cumulated_view_factor=cumulated_ext_surf_view_factor,
        context_surface_temperature_schedule=context_surface_temperature_schedule,
        sky_view_factor=sky_view_factor,
        ground_view_factor=ground_view_factor,
        ground_temperature_schedule=ground_temperature_schedule)
    additional_string += _write_surface_property_local_environment(surface_name=surface_name,
                                                                   surface_property_local_environment_name=surface_property_local_environment_name,
                                                                   surface_property_surrounding_surface_name=surface_property_surrounding_surface_name)
    return additional_string


def name_surrounding_surface_temperature_schedule(surface_name: str) -> str:
    """
    Name the temperature schedule of the surrounding surface.
    :param surface_name: name of the surface
    :return: name of the temperature schedule
    """
    return surface_name + "_ConSur_TemSch"


def _write_surface_temperature_schedule(schedule_name: str, init_temperature: float = 20.) -> str:
    """

    :return:
    """
    schedule_str = f"\n" \
                   f"Schedule:Constant,\n" \
                   f"  {schedule_name}, !- Name\n" \
                   f"  Temperature, !- Schedule Type Limits Name\n" \
                   f"  {init_temperature};  !- Hourly Value\n"
    return schedule_str


def _write_surface_property_surrounding_surfaces(surface_property_surrounding_surface_name: str,
                                                 context_surface_name: str, cumulated_view_factor: float,
                                                 context_surface_temperature_schedule: str,
                                                 sky_view_factor: float = '',
                                                 ground_view_factor: float = '',
                                                 ground_temperature_schedule: str = '') -> str:
    """

    :return:
    """
    surface_property_str = f"\n" \
                           f"SurfaceProperty:SurroundingSurfaces,\n" \
                           f"  {surface_property_surrounding_surface_name}, !- Name\n" \
                           f"  {sky_view_factor}, !- Sky View Factor\n" \
                           f"  , !- Sky Temperature Schedule Name\n" \
                           f"  {ground_view_factor}, !- Ground View Factor\n" \
                           f"  {ground_temperature_schedule}, !- Ground Temperature Schedule Name\n" \
                           f"  {context_surface_name}, !- Surrounding Surface 1 Name\n" \
                           f"  {cumulated_view_factor}, !- Surrounding Surface 1 View Factor\n" \
                           f"  {context_surface_temperature_schedule}; !- Surrounding Surface 1 Temperature Schedule Name\n"

    return surface_property_str


def _write_surface_property_local_environment(surface_name: str, surface_property_local_environment_name: str,
                                              surface_property_surrounding_surface_name: str,
                                              ground_surface_object_name: str = '') -> str:
    """

    :param surface_name:
    :param surface_property_surrounding_surface_name:
    :return:
    """
    surface_property_str = f"\n" \
                           f"SurfaceProperty:LocalEnvironment,\n" \
                           f"  {surface_property_local_environment_name}, !- Name\n" \
                           f"  {surface_name}, !- Exterior Surface Name\n" \
                           f"  , !- Sunlit Fraction Schedule Name\n" \
                           f"  {surface_property_surrounding_surface_name}, !- Surrounding Surfaces Object Name\n" \
                           f"  , !- Outdoor Air Node Name\n" \
                           f"  {ground_surface_object_name}; !- Ground Surfaces Object Name\n"
    return surface_property_str
