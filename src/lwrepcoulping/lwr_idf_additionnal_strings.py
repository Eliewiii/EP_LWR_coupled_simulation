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


def _write_surface_lwr_idf_additional_string(surface_name: str, sky_view_factor: float, ground_view_factor: float,
                                             cumulated_view_factor: float,
                                             ground_temperature_schedule: str = '') -> str:
    """

    :param surface_name:
    :param sky_view_factor:
    :param ground_view_factor:
    :param cumulated_view_factor:
    :param ground_temperature_schedule: schedule name for the ground temperature, to be left empty if not used.
        If used, it will be common to all surfaces, and thus created once, outside of this function.
        For more detailed ground temperature, use the dedicated SurfaceProperty:GroundSurfaces, with multiple surfaces
        and schedules, or one ground surface for each surface, with averaged and weighted temperature the same way as
        regular context surface. Not implemented here as no ground temperature model was developed.
    :return:
    """

    surface_property_local_environment_name = surface_name + "_local_environment"  # I guess to change to be lighter
    surface_property_surrounding_surface_name = surface_name + "_surrounding"
    context_surface_name = surface_name + "_context_surface"
    context_surface_temperature_schedule = surface_name + "_context_surface_temperature_schedule"
    context_surface_temperature_schedule_actuator = surface_name + "_context_surface_temperature_schedule_actuator"

    additional_string = _write_surface_property_surrounding_surfaces(surface_name=surface_name,
                                                                     surface_property_surrounding_surface_name=surface_property_surrounding_surface_name,
                                                                     sky_view_factor=sky_view_factor,
                                                                     ground_view_factor=ground_view_factor,
                                                                     context_surface_name=context_surface_name,
                                                                     cumulated_view_factor=cumulated_view_factor,
                                                                     context_surface_temperature_schedule=context_surface_temperature_schedule,
                                                                     ground_temperature_schedule=ground_temperature_schedule)
    additional_string += '\n'


def _write_actuator(schedule_name: str) -> str:
    """

    :return:
    """


def _write_surface_temperature_schedule(schedule_name: str, init_temperature: float) -> str:
    """

    :return:
    """
    schedule_str = f"\n" \
                   f"Schedule:Compact ," \
                   f"{schedule_name}, !- Name" \
                   f"{init_temperature} , !- Schedule Type Limits Name" \
                   f"Through: 12/31 , !- Field 1" \
                   f"For: AllDays , !- Field 2" \
                   f"Until: 24:00 , 15.0; !- Field 3"
    return schedule_str


def _write_surface_property_surrounding_surfaces(surface_name: str, surface_property_surrounding_surface_name: str,
                                                 sky_view_factor: float, ground_view_factor: float,
                                                 context_surface_name: str, cumulated_view_factor: float,
                                                 context_surface_temperature_schedule: str,
                                                 ground_temperature_schedule: str = '') -> str:
    """

    :return:
    """
    surface_property_str = f"\n" \
                           f"SurfaceProperty:SurroundingSurfaces," \
                           f"{surface_property_surrounding_surface_name}, !- Name" \
                           f"{sky_view_factor}, !- Sky View Factor" \
                           f", !- Sky Temperature Schedule Name" \
                           f"{ground_view_factor}, !- Ground View Factor" \
                           f"{ground_temperature_schedule}, !- Ground Temperature Schedule Name" \
                           f"{context_surface_name}, !- Context Surface Name" \
                           f"{cumulated_view_factor}, !- Cumulated View Factor to Context Surface" \
                           f"{context_surface_temperature_schedule}; !- Context Surface Temperature Schedule Name"

    return surface_property_str


def _write_surface_property_local_environment(surface_name: str, surface_property_local_environment_name: str,
                                              surface_property_surrounding_surface_name: str) -> str:
    """

    :param surface_name:
    :param surface_property_surrounding_surface_name:
    :return:
    """
    surface_property_str = f"\n" \
                           f"SurfaceProperty:LocalEnvironment," \
                           f"{surface_property_local_environment_name}, !- Name" \
                           f"{surface_name}; !- Exterior Surface Name" \
                           f"ExtShadingSch:Zn001:Wall001 , !- Sunlit Fraction Schedule Name" \
                           f"{surface_property_surrounding_surface_name} , !- Surrounding Surfaces Object Name" \
                           f"OutdoorAirNode :0001 , !- Outdoor Air Node Name" \
                           f"GndSurfs:South; !- Ground Surfaces Object Name"
    return surface_property_str
