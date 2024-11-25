"""

"""
import os

from src.lwrepcoupling.lwr_idf_additionnal_strings import generate_surface_lwr_idf_additional_string, \
    _write_surface_temperature_schedule, _write_surface_property_surrounding_surfaces, \
    _write_surface_property_local_environment


class TestLwrIdfAdditionnalStrings:

    def test_write_surface_temperature_schedule(self):
        schedule_name = "Test_surrounding_surface_temp"
        init_temperature = 20
        schedule_str = _write_surface_temperature_schedule(schedule_name, init_temperature)
        print(schedule_str)

    def test_write_surface_property_surrounding_surfaces(self):
        surface_property_surrounding_surface_name = "Test_surrounding_surface"
        sky_view_factor = 0.5
        ground_view_factor = 0.5
        context_surface_name = "Test_context_surface"
        cumulated_view_factor = 1
        context_surface_temperature_schedule = "Test_context_surface_temperature_schedule"
        ground_temperature_schedule = "Test_ground_temperature_schedule"
        additional_string = _write_surface_property_surrounding_surfaces(
            surface_property_surrounding_surface_name=surface_property_surrounding_surface_name,
            sky_view_factor=sky_view_factor,
            ground_view_factor=ground_view_factor,
            context_surface_name=context_surface_name,
            cumulated_view_factor=cumulated_view_factor,
            context_surface_temperature_schedule=context_surface_temperature_schedule,
            ground_temperature_schedule=ground_temperature_schedule)
        print(additional_string)

    def test_write_surface_property_local_environment(self):
        surface_name = "Test_surface"
        surface_property_local_environment_name = "Test_surface_local_environment"
        surface_property_surrounding_surface_name = "Test_surrounding_surface"
        ground_surface_object_name = "Test_ground_surface"
        additional_string = _write_surface_property_local_environment(
            surface_name=surface_name,
            surface_property_local_environment_name=surface_property_local_environment_name,
            surface_property_surrounding_surface_name=surface_property_surrounding_surface_name,
            ground_surface_object_name=ground_surface_object_name)
        print(additional_string)

    def test_write_surface_lwr_idf_additional_string(self):
        surface_name = "Test_surface"
        sky_view_factor = 0.2
        ground_view_factor = 0.2
        cumulated_ext_surf_view_factor = 0.5
        additional_string = generate_surface_lwr_idf_additional_string(
            surface_name=surface_name,
            sky_view_factor=sky_view_factor,
            ground_view_factor=ground_view_factor,
            cumulated_ext_surf_view_factor=cumulated_ext_surf_view_factor)
        print(additional_string)
