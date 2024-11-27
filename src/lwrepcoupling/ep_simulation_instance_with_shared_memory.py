"""
Class to represent an instance of a simulation of the EP model, generating the idf additional strings,
managing the handlers etc.
"""

import numpy as np
import os
import shutil
from copy import deepcopy
from multiprocessing import shared_memory
from typing import List

from .pyenergyplus.api import EnergyPlusAPI
from .lwr_idf_additionnal_strings import generate_surface_lwr_idf_additional_string, \
    name_surrounding_surface_temperature_schedule


class EpSimulationInstance:

    def __init__(self, identifier: str, path_idf: str, path_output_dir: str, simulation_index: int):
        """
        Constructor of the class.
        :param identifier: str, The identifier of the building/simulation
        :param path_idf: str, The path to the IDF file of the building
        :param path_output_dir: str, The path to the output directory of the simulation
        :param simulation_index: int, The index of the simulation in the manager
        """
        # Identifier
        self._identifier = identifier
        # EP API and state
        self._api = None
        self._state = None
        # Paths to the files and directories
        self._path_original_idf = path_idf
        self._path_simulated_idf = None
        self._path_output_dir = path_output_dir

        # Geometry
        self._outdoor_surface_name_list = []
        self._outdoor_surface_surrounding_surface_vf_dict = {}
        self._outdoor_surface_sky_vf_dict = {}
        self._outdoor_surface_ground_vf_dict = {}
        # todo: potentially need to add the emissivity or other related parameters

        # Handlers
        self._schedule_name_dict = {}
        self._schedule_actuator_handle_dict = {}
        self._surface_temp_handler_dict = {}
        self._surrounding_surface_temperature_schedule_temperature_handler_dict = {}

        # LWR data
        self._vf_epsilon_matrix = None
        self._vf_matrix = None

        # Synchronization attributes
        self._simulation_index = simulation_index  # Index of the simulation, to synchronize the simulation and
        self._surface_index_min = None
        self._surface_index_max = None

        # Test variables to check
        self._test_temperature_list = []
        self._test_surrounding_surface_temperature_list = []
        self._test_current_time_list = []

    # -----------------------------------------------------#
    # --------------------- Properties --------------------#
    # -----------------------------------------------------#

    @property
    def num_outdoor_surfaces(self):
        return len(self._outdoor_surface_name_list)

    # -----------------------------------------------------#
    # ------------------ Init Method ----------------------#
    # -----------------------------------------------------#

    def set_outdoor_surfaces_and_view_factors(self, outdoor_surface_name_list: List[str],
                                              outdoor_surface_surrounding_surface_vf_dict: dict,
                                              outdoor_surface_sky_vf_dict: dict,
                                              outdoor_surface_ground_vf_dict: dict,
                                              manager_num_outdoor_surfaces: int):
        """
        Set the outdoor surfaces and the view factors for the simulation.
        :param outdoor_surface_name_list: list, List of the names of the outdoor surfaces
        :param outdoor_surface_surrounding_surface_vf_dict: dict, Dictionary of the cumulative view factors between the outdoor
        surfaces and the surrounding surfaces
        :param outdoor_surface_sky_vf_dict: dict, Dictionary of the view factors between the outdoor surfaces and the sky
        :param outdoor_surface_ground_vf_dict: dict, Dictionary of the view factors between the outdoor surfaces and the ground
        :param manager_num_outdoor_surfaces: int, The number of outdoor surfaces in the manager to set the index boundaries
        for the access to the shared memory
        :return:
        """
        # Set surfaces
        self._outdoor_surface_name_list = outdoor_surface_name_list
        self._outdoor_surface_surrounding_surface_vf_dict = outdoor_surface_surrounding_surface_vf_dict
        self._outdoor_surface_sky_vf_dict = outdoor_surface_sky_vf_dict
        self._outdoor_surface_ground_vf_dict = outdoor_surface_ground_vf_dict
        # Set the index boundaries for the access to the shared memory
        self._surface_index_min = manager_num_outdoor_surfaces
        self._surface_index_max = manager_num_outdoor_surfaces + len(outdoor_surface_name_list) - 1

    def set_vf_matrices(self, vf_matrix, vf_eps_matrix):
        """

        :param vf_matrix:
        :param vf_eps_matrix:
        :return:
        """

        # Todo :

    def adjust_idf(self):
        """
        Make a copy of the idf in the directory of the simulation and add the additional strings for the LWR coupling.
        It also generates the schedule names for the surrounding surface temperature.
        """
        # Generate the additional strings
        additional_strings = self.generate_idfs_additional_strings()
        # Make a copy of the original IDF file to the output directory
        self._path_simulated_idf = os.path.join(self._path_output_dir, f"in.idf")
        shutil.copy(self._path_original_idf, self._path_simulated_idf)
        # Add the additional strings to the IDF file
        with open(self._path_simulated_idf, 'a') as file:
            file.write(additional_strings)

    def generate_idfs_additional_strings(self):
        """
        Generate the additional strings to add to the IDF file for the LWR coupling.
        Consist of the generation of a SurfaceProperty:LocalEnvironment and a SurfaceProperty:SurroundingSurfaces for each
        outdoor surface, with temperature schedule for the surrounding surfaces to be actuated to account for the LWR.
        :return:
        """
        additional_strings = ""
        for surface_name in self._outdoor_surface_name_list:
            additional_strings += generate_surface_lwr_idf_additional_string(
                surface_name=surface_name,
                cumulated_ext_surf_view_factor=self._outdoor_surface_surrounding_surface_vf_dict[
                    surface_name],
                sky_view_factor=self._outdoor_surface_sky_vf_dict[surface_name],
                ground_view_factor=self._outdoor_surface_ground_vf_dict[surface_name]
            )
            # Add the schedule name to the dictionary
            self._schedule_name_dict[surface_name] = name_surrounding_surface_temperature_schedule(
                surface_name)
        return additional_strings

    # -----------------------------------------------------#
    # ----------- EnergyPlus API Preparation --------------#
    # -----------------------------------------------------#

    def request_variables_before_running_simulation(self):
        """
        Request the variables to access the surface temperature of the outdoor surfaces during the simulation.
        """
        for surface_name in self._outdoor_surface_name_list:
            self._api.exchange.request_variable(self._state, "SURFACE OUTSIDE FACE TEMPERATURE",
                                                surface_name)

    def request_additional_variables_before_running_simulation_for_testing(self):
        """
        Request the variables to access the schedule values of the surrounding surface temperature during the simulation.
        For testing purposes only as it is not needed for the LWR computation.
        """
        for surface_name in self._outdoor_surface_name_list:
            self._api.exchange.request_variable(self._state, "Schedule Value",
                                                self._schedule_name_dict[surface_name])

    def initialize_actuator_handler_callback_function(self,state):
        """
        Initialize the actuator handlers for the surrounding surface temperature schedules.
        Should be run at the end of the warmup period.
        """
        for surface_name in self._outdoor_surface_name_list:
            schedule_actuator_handle = self._api.exchange.get_actuator_handle(state,
                                                                              "Schedule:Constant",
                                                                              "Schedule Value",
                                                                              self._schedule_name_dict[
                                                                                  surface_name])
            if schedule_actuator_handle == -1:
                raise ValueError(
                    f"Failed to create actuator for schedule {self._schedule_name_dict[surface_name]}")
            else:
                self._schedule_actuator_handle_dict[surface_name] = schedule_actuator_handle

    def init_surface_temperature_handlers_call_back_function(self,state):
        """
        Initialize the handlers to access the surface temperatures of the outdoor surfaces.
        Should be run at the end of the warmup period.
        """
        for surface_name in self._outdoor_surface_name_list:
            self._surface_temp_handler_dict[surface_name] = self._api.exchange.get_variable_handle(
                state,
                "SURFACE OUTSIDE FACE TEMPERATURE",
                surface_name)

    def init_surrounding_surface_schedule_handlers_call_back_function_for_testing(self,state):
        """
        Initialize the handlers to access the schedule values of the surrounding surface temperatures.
        Should be run at the end of the warmup period.
        For testing purposes only as it is not needed for the LWR computation.
        """
        for surface_name in self._outdoor_surface_name_list:
            self._surrounding_surface_temperature_schedule_temperature_handler_dict[
                surface_name] = self._api.exchange.get_variable_handle(state, "Schedule Value",
                                                                       self._schedule_name_dict[surface_name])

    def get_surface_temperature_of_all_outdoor_surfaces_in_kelvin(self) -> List[float]:
        """
        Reads the surface temperature of all the outdoor surfaces and store them in a list.
        :return: list,  List of surface temperatures
        """
        surface_temperatures_list = []
        for surface_name in self._outdoor_surface_name_list:
            surface_temperatures_list.append(
                self._api.exchange.get_variable_value(self._state,
                                                      self._surface_temp_handler_dict[
                                                          surface_name]) + 273.15)  # convert to Kelvin
        return surface_temperatures_list

    # def write_surface_temperatures_in_file(self, current_time):
    #     """
    #
    #     :return:
    #     """
    #     surface_temperatures_list = self.get_surface_temperature_of_all_outdoor_surfaces()
    #     # write down the surface temperatures in a file with tmp extension to avoid reading it before it is fully written
    #     with open(os.path.join(self._path_output_dir, f"{current_time}_{self.simulation_index}.tmp"),
    #               'a') as file:
    #         file.write(f"{current_time} {surface_temperatures_list}\n")
    #     # Rename the file to remove the .tmp extension
    #     os.rename(os.path.join(self._path_output_dir, f"{current_time}_{self.simulation_index}.tmp"),
    #               os.path.join(self._path_output_dir, f"{current_time}_{self.simulation_index}.txt"))

    # def coupled_simulation_callback_function(self, state, shared_array, shared_memory_lock,
    #                                          synch_point_barrier):
    #     """
    #     Function to run at the end (or beginning) of each time step, to update the schedule values and surrounding surface temperatures.
    #     :return:
    #     """
    #     # Todo implement teh rest of the function with the LWR logic
    #
    #     # Get current simulation time (in hours)
    #     current_time = self._api.exchange.current_sim_time(state)
    #     # Get the surface temperatures of all the surfaces
    #     surface_temperatures_list = self.get_surface_temperature_of_all_outdoor_surfaces()
    #     # write down the surface temperatures the shared memory
    #     with shared_memory_lock:
    #         # todo: need to indicated properly the start and end index of the shared memory
    #         # Here we are writing the list as a slice of the shared memory
    #         shared_array[index:index + len(float_list)] = np.array(
    #             surface_temperatures_list) ** 4  # directly give the temperatures power 4
    #         print(f"Process {index} wrote data at index {index}")
    #
    #     # wait for the other building to write down its surface temperatures
    #     synch_point_barrier.wait()
    #     # can directly use the numpy array to compute the LWR
    #     # update the surrounding surface temperature schedules with the proper "mean radiant temperature" values
    #
    #     # Delete the file with the surface temperatures
    #     # os.remove(os.path.join(self._path_output_dir,f"{current_time}_{self.simulation_index}.txt"))

    # -----------------------------------------------------#
    # ------------ Main Callback Function -----------------#
    # -----------------------------------------------------#

    def coupled_simulation_callback_function_test(self, state, shared_array, shared_memory_lock,
                                                  synch_point_barrier):
        """
        Function to run at the end (or beginning) of each time step, to update the schedule values and surrounding surface temperatures.
        This function is a test version that will not perform the LWR computation but will write the surface temperatures and update the schedules
        to test the synchronization of the shared memory and the barrier.
        :return:
        """
        # prevent from runnning the function if the actuator handlers are not initialized (at warmup)
        if self._schedule_actuator_handle_dict == {}:
            return

        # Need to make sure all the simulation are at the same time step, as EnergyPlus might adjust the time step if needed.
        # Need to check the timee step and make the simulation wait if needed with a second shared memory and barrier
        # Could be overcomed by using a smaller time step, but it needs to be tested

        # Get the surface temperatures of all the surfaces
        surface_temperatures_list = self.get_surface_temperature_of_all_outdoor_surfaces_in_kelvin()

        # write down the surface temperatures the shared memory
        with shared_memory_lock:
            # todo: need to indicated properly the start and end index of the shared memory
            # Here we are writing the list as a slice of the shared memory
            shared_array[self._surface_index_min:self._surface_index_max + 1] = np.array(
                # +1 to include the last index
                surface_temperatures_list) ** 4  # directly give the temperatures power 4

        # wait for the other building to write down its surface temperatures
        synch_point_barrier.wait()
        # Compute a somewhat mean surface temperature, to test the synchronization, only one surface per building will be used
        mean_surface_temperature = np.mean(shared_array ** (1 / 4)) - 273.15  # convert back to Celsius
        # Set the surrounding surface temperature to the average of the surface temperatures
        for surface_name in self._outdoor_surface_name_list:
            self._api.exchange.set_actuator_value(state,
                                                  self._schedule_actuator_handle_dict[
                                                      surface_name], mean_surface_temperature)
            # get the current value of the schedule
            current_value = self._api.exchange.get_variable_value(state,
                                                                  self._surrounding_surface_temperature_schedule_temperature_handler_dict[
                                                                      surface_name])
            # assert current_value == mean_surface_temperature, f"Error: the schedule value is not properly set, current value = {current_value}, expected value = {mean_surface_temperature}"

        # -- For testing --#
        # Get current simulation time (in hours)
        self._test_current_time_list.append(self._api.exchange.current_sim_time(state))
        self._test_temperature_list.append(deepcopy(np.array(surface_temperatures_list)-273.15))
        self._test_surrounding_surface_temperature_list.append(mean_surface_temperature)
        # -----------------#

    # -----------------------------------------------------#
    # ------------ Run Simulation Function ----------------#
    # -----------------------------------------------------#

    def run_ep_simulation(self, shared_memory_name:str, shared_memory_array_size:int, shared_memory_lock,
                          synch_point_barrier, path_epw:str, path_energyplus_dir: str):
        """
        Run the EnergyPlus simulation with the shared memory and the synchronization objects.
        :param shared_memory_name: str, The name of the shared memory to access the surface temperatures
        :param shared_memory_array_size: int, The size of the shared memory array
        :param shared_memory_lock: Lock, The lock to limit writing access to the shared memory
        :param synch_point_barrier: Barrier, The barrier to synchronize processes
        :param path_epw: str, The path to the EPW file
        :param path_energyplus_dir: str, The path to the EnergyPlus directory
        :return: self: EpSimulationInstance, The instance of the simulation to update the one in the manager
        """
        # Point to the shared memory
        shm = shared_memory.SharedMemory(name=shared_memory_name)
        shared_array = np.ndarray(shared_memory_array_size, dtype=np.float64, buffer=shm.buf)

        # initialize the EnergyPlus API and simulation state
        self._api = EnergyPlusAPI(running_as_python_plugin=True, path_to_ep_folder=path_energyplus_dir)
        self._state = self._api.state_manager.new_state()

        # request the variables to access schedule and surface temperature values during the simulation
        self.request_variables_before_running_simulation()

        # -- For testing --#
        self.request_additional_variables_before_running_simulation_for_testing()

        # -----------------#

        # Make wrapper for the main callback function
        def simulation_callback_function(state):
            """
            Wrapper for the main callback function to pass arguments (which are not allowed in the callback function).
            :param state:
            :return:
            """
            # return self.coupled_simulation_callback_function(state, shared_array, shared_memory_lock,
            #                                                  synch_point_barrier)
            return self.coupled_simulation_callback_function_test(state, shared_array, shared_memory_lock,
                                                                  synch_point_barrier)

        # Set the callback functions to run at the various moment of the simulation
        self._api.runtime.callback_begin_new_environment(self._state,
                                                                         self.initialize_actuator_handler_callback_function)
        self._api.runtime.callback_begin_new_environment(self._state,
                                                                         self.init_surface_temperature_handlers_call_back_function)

        # -- For testing --#
        self._api.runtime.callback_begin_new_environment(self._state,
                                                                         self.init_surrounding_surface_schedule_handlers_call_back_function_for_testing)
        # -----------------#

        self._api.runtime.callback_end_zone_timestep_after_zone_reporting(self._state,
                                                                            simulation_callback_function)  # todo: might be change to the end of the timestep

        # Run the EnergyPlus simulation
        self._api.runtime.run_energyplus(
            self._state,
            [
                '-r',  # Run annual simulation
                '-w', path_epw,  # Weather file
                '-d', self._path_output_dir,  # Output directory
                self._path_simulated_idf  # Input IDF file
            ]
        )

        return self._test_current_time_list, self._test_temperature_list, self._test_surrounding_surface_temperature_list
