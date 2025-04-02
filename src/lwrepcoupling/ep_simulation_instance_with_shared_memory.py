"""
Class to represent an instance of a simulation of the EP model, generating the idf additional strings,
managing the handlers etc.
"""

import os
import shutil
import pickle

import numpy as np

from copy import deepcopy
from multiprocessing import shared_memory
from typing import List, Optional

from .pyenergyplus.api import EnergyPlusAPI
from .lwr_idf_additionnal_strings import generate_surface_lwr_idf_additional_string, \
    name_surrounding_surface_temperature_schedule


class EpSimulationInstance:

    def __init__(self, identifier: str, path_idf: str, path_output_dir: str):
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
        self._outdoor_surface_surrounding_surface_vf_list = []
        self._outdoor_surface_sky_vf_list = []
        self._outdoor_surface_ground_vf_list = []
        # todo: potentially need to add the emissivity or other related parameters

        # Handlers
        self._schedule_name_list = []
        self._schedule_actuator_handle_list = []
        self._surface_temp_handler_list = []
        self._surrounding_surface_temperature_schedule_temperature_handler_list = []

        # flags
        self._warmup_started = False
        self._warmup_done = False

        # LWR data
        self._resolution_mtx = None

        # Synchronization attributes
        self._simulation_index = None
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
    def num_outdoor_surfaces(self) -> int:
        return len(self._outdoor_surface_name_list)

    # -----------------------------------------------------#
    # -------------- Export and Import --------------------#
    # -----------------------------------------------------#

    def to_pkl(self, path_folder: str, file_name: Optional[str] = None,
               get_path_only: bool = False) -> str:
        """
        Save the instance to a pickle file.
        :param path_folder: str, The folder path where the pickle file will be saved.
        :param file_name: Optional[str], The name of the pickle file.
        :param get_path_only: bool, If True, return the path of the pickle file only; otherwise, save the instance.
        :return: str, The path of the pickle file.
        """
        if file_name is None:
            file_name = f"ep_sim_instance_{self._identifier}.pkl"
        path_pkl_file = os.path.join(path_folder, file_name)

        if not get_path_only:
            try:
                with open(path_pkl_file, 'wb') as f:
                    pickle.dump(self, f)
            except IOError as e:
                raise RuntimeError(f"Error saving pickle file: {e}")

        return path_pkl_file

    @staticmethod
    def from_pkl(path_pkl_file: str) -> "EpSimulationInstance":
        """
        Load an EpSimulationInstance object from a pickle file.
        :param path_pkl_file: str, The path of the pickle file.
        :return: EpSimulationInstance, The loaded instance.
        """
        try:
            with open(path_pkl_file, 'rb') as f:
                return pickle.load(f)
        except IOError as e:
            raise RuntimeError(f"Error loading pickle file: {e}")

    # -----------------------------------------------------#
    # ---- Generate object from simulation manager --------#
    # -----------------------------------------------------#

    @classmethod
    def init_and_preprocess_to_pkl(cls, identifier: str, simulation_index: int, path_idf: str,
                                   path_output_dir: str, min_surface_index: int, max_surface_index: int,
                                   outdoor_surface_id_list: List[str], resolution_mtx: np.ndarray,
                                   srd_vf_list: List[float], sky_vf_list: Optional[List[float]] = None,
                                   ground_vf_list: Optional[List[float]] = None) -> str:
        """
        Initialize an instance, preprocess data, and save it as a pickle file.
        :param identifier: str, The identifier of the building/simulation.
        :param simulation_index: int, The index of the simulation in the manager.
        :param path_idf: str, The path to the IDF file of the building.
        :param path_output_dir: str, The path to the output directory of the simulation.
        :param min_surface_index: int, The minimum index of the surfaces in shared memory.
        :param max_surface_index: int, The maximum index of the surfaces in shared memory.
        :param outdoor_surface_id_list: List[str], The list of outdoor surface names.
        :param resolution_mtx: np.ndarray, The resolution matrix for the simulation.
        :param srd_vf_list: List[float], The list of view factors to surrounding surfaces.
        :param sky_vf_list: Optional[List[float]], The list of view factors to the sky.
        :param ground_vf_list: Optional[List[float]], The list of view factors to the ground.
        :return: str, The path to the saved pickle file.
        """
        ep_simulation_instance_obj = cls(identifier=identifier, path_idf=path_idf,
                                         path_output_dir=path_output_dir)
        ep_simulation_instance_obj._set_simulation_index(simulation_index)
        ep_simulation_instance_obj._set_min_and_max_surface_index(min_surface_index, max_surface_index)
        ep_simulation_instance_obj._set_outdoor_surfaces_and_view_factors(outdoor_surface_id_list,
                                                                          srd_vf_list, sky_vf_list,
                                                                          ground_vf_list)
        ep_simulation_instance_obj.set_vf_matrices(resolution_mtx)
        ep_simulation_instance_obj.adjust_idf()
        return ep_simulation_instance_obj.to_pkl(path_output_dir)

    # -----------------------------------------------------#
    # ------------------ Init Method ----------------------#
    # -----------------------------------------------------#

    def _set_simulation_index(self, simulation_index: int):
        """
        Set the simulation index.
        :param simulation_index: int, The index of the simulation in the manager.
        """
        self._simulation_index = simulation_index

    def _set_min_and_max_surface_index(self, min_surface_index, max_surface_index):
        """
        Set the minimum and maximum index of the surfaces in the shared memory.
        :param min_surface_index: int, The minimum index of the surfaces in the shared memory
        :param max_surface_index: int, The maximum index of the surfaces in the shared memory
        :return:
        """
        # todo : add some checks
        self._surface_index_min = min_surface_index
        self._surface_index_max = max_surface_index

    def _set_outdoor_surfaces_and_view_factors(self, outdoor_surface_name_list: List[str],
                                               outdoor_surface_surrounding_surface_vf_list: List[float],
                                               outdoor_surface_sky_vf_list: Optional[List[float]],
                                               outdoor_surface_ground_vf_list: Optional[List[float]]):
        """
        Set the outdoor surfaces and the view factors for the simulation.
        Ensures all input lists have the same length.
        :param outdoor_surface_name_list: List[str], The list of the names of the outdoor surfaces.
        :param outdoor_surface_surrounding_surface_vf_list: List[float], The list of the view factors to the surrounding surfaces.
        :param outdoor_surface_sky_vf_list: Optional[List[float]], The list of the view factors to the sky.
        :param outdoor_surface_ground_vf_list: Optional[List[float]], The list of the view factors to the ground.
        """
        self._outdoor_surface_name_list = outdoor_surface_name_list
        self._outdoor_surface_surrounding_surface_vf_list = outdoor_surface_surrounding_surface_vf_list
        self._outdoor_surface_sky_vf_list = outdoor_surface_sky_vf_list or []
        self._outdoor_surface_ground_vf_list = outdoor_surface_ground_vf_list or []

    def set_vf_matrices(self, resolution_mxt):
        """

        :param resolution_mxt:
        """
        self._resolution_mtx = resolution_mxt

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
        for i, surface_name in enumerate(self._outdoor_surface_name_list):
            additional_strings += generate_surface_lwr_idf_additional_string(
                surface_name=surface_name,
                cumulated_ext_surf_view_factor=self._outdoor_surface_surrounding_surface_vf_list[i],
                # sky_view_factor=self._outdoor_surface_sky_vf_list[i],
                # ground_view_factor=self._outdoor_surface_ground_vf_list[i]
            )
            # Add the schedule name to the dictionary
            self._schedule_name_list.append(name_surrounding_surface_temperature_schedule(
                surface_name))
        return additional_strings

    # -----------------------------------------------------#
    # -----------       LWR Resolution       --------------#
    # -----------------------------------------------------#

    def compute_srd_mean_radiant_temperatures_in_c(self, temperature_p4_vector: np.ndarray):
        """

        :param temperature_vector:
        :return:
        """

        return (np.power(temperature_p4_vector.T[
                         self._surface_index_min:self._surface_index_max + 1] - self._resolution_mtx @ temperature_p4_vector.T,
                         1 / 4) - 273.15).tolist()

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
        for i, surface_name in enumerate(self._outdoor_surface_name_list):
            self._api.exchange.request_variable(self._state, "Schedule Value",
                                                self._schedule_name_list[i])

    def initialize_actuator_handler_callback_function(self, state):
        """
        Initialize the actuator handlers for the surrounding surface temperature schedules.
        Should be run at the end of the warmup period.
        """
        for i, surface_name in enumerate(self._outdoor_surface_name_list):
            schedule_actuator_handle = self._api.exchange.get_actuator_handle(state,
                                                                              "Schedule:Constant",
                                                                              "Schedule Value",
                                                                              self._schedule_name_list[
                                                                                  i])
            if schedule_actuator_handle == -1:
                raise ValueError(
                    f"Failed to create actuator for schedule {self._schedule_name_list[i]}")
            else:
                self._schedule_actuator_handle_list.append(schedule_actuator_handle)

    def init_surface_temperature_handlers_call_back_function(self, state):
        """
        Initialize the handlers to access the surface temperatures of the outdoor surfaces.
        Should be run at the end of the warmup period.
        """
        for i, surface_name in enumerate(self._outdoor_surface_name_list):
            self._surface_temp_handler_list.append(self._api.exchange.get_variable_handle(
                state,
                "SURFACE OUTSIDE FACE TEMPERATURE",
                surface_name))

    def init_surrounding_surface_schedule_handlers_call_back_function_for_testing(self, state):
        """
        Initialize the handlers to access the schedule values of the surrounding surface temperatures.
        Should be run at the end of the warmup period.
        For testing purposes only as it is not needed for the LWR computation.
        """
        for i, surface_name in enumerate(self._outdoor_surface_name_list):
            self._surrounding_surface_temperature_schedule_temperature_handler_list.append(
                self._api.exchange.get_variable_handle(state, "Schedule Value",
                                                       self._schedule_name_list[i]))

    def get_surface_temperature_of_all_outdoor_surfaces_in_kelvin(self) -> List[float]:
        """
        Reads the surface temperature of all the outdoor surfaces and store them in a list.
        :return: list,  List of surface temperatures
        """
        surface_temperatures_list = []
        for i, surface_name in enumerate(self._outdoor_surface_name_list):
            surface_temperatures_list.append(
                self._api.exchange.get_variable_value(self._state,
                                                      self._surface_temp_handler_list[
                                                          i]) + 273.15)  # convert to Kelvin
        return surface_temperatures_list


    # -----------------------------------------------------#
    # ------------ Main Callback Function -----------------#
    # -----------------------------------------------------#

    # def all_instances_synch_for_warmup(self,shared_array_timestep,sim_dt):
    #     """
    #
    #     :param shared_array_timestep:
    #     :param sim_dt:
    #     :return:
    #     """
    #     def is_identical_or_first_time_step()
    #     if

    def coupled_simulation_callback_function(self, state, shared_array,shared_array_timestep, shared_memory_lock,
                                             synch_point_barrier):
        """
        Function to run at the end (or beginning) of each time step, to update the schedule values and surrounding surface temperatures.
        This function is a test version that will not perform the LWR computation but will write the surface temperatures and update the schedules
        to test the synchronization of the shared memory and the barrier.
        :return:
        """

        # prevent from runnning the function if the actuator handlers are not initialized (at warmup)
        if not self._schedule_actuator_handle_list:
            return
        current_time = self._api.exchange.current_sim_time(state)

        if not self._warmup_started:
            self._warmup_started = True
            return
        if not self._warmup_done:
            if np.isclose(current_time, 0.15, rtol=1e-05, atol=1e-05):
                self._warmup_done = True
            else:
                return

        # current_time = api.exchange.current_sim_time(state)

        # Get the surface temperatures of all the surfaces
        surface_temperatures_list = self.get_surface_temperature_of_all_outdoor_surfaces_in_kelvin()

        synch_point_barrier.wait()

        # write down the surface temperatures the shared memory
        with shared_memory_lock:
            # todo: need to indicated properly the start and end index of the shared memory
            # Here we are writing the list as a slice of the shared memory
            np.copyto(shared_array[self._surface_index_min:self._surface_index_max + 1],
                      np.array(surface_temperatures_list) ** 4)  # directly give the temperatures power 4

            np.copyto(shared_array_timestep[self._simulation_index:self._simulation_index+1],
                      np.array([current_time]))


        synch_point_barrier.wait()

        if self._simulation_index == 0:
            print(shared_array_timestep)
        # Compute a somewhat mean surface temperature, to test the synchronization, only one surface per building will be used
        list_srd_mean_radiant_temperature_in_c = self.compute_srd_mean_radiant_temperatures_in_c(
            temperature_p4_vector=shared_array)  # convert back to Celsius
        # Set the surrounding surface temperature to the average of the surface temperatures
        for i, srd_mrt in enumerate(list_srd_mean_radiant_temperature_in_c):
            self._api.exchange.set_actuator_value(state,
                                                  self._schedule_actuator_handle_list[i], srd_mrt)


    # -----------------------------------------------------#
    # ------------ Run Simulation Function ----------------#
    # -----------------------------------------------------#

    @classmethod
    def run_coupled_simulation_from_ep_instance(cls, path_ep_instance_pkl: str, **kwargs):
        """

        :param path_ep_instance_pkl:
        :param path_file_resolution_mtx_npz:
        :param kwargs:
        :return:
        """
        # Load the ep object
        ep_sim_inst_obj = cls.from_pkl(path_pkl_file=path_ep_instance_pkl)
        # Run the simulation
        ep_sim_inst_obj.run_ep_simulation(**kwargs)

    def run_ep_simulation(self, shared_memory_name: str, shared_memory_timestep_name: str,
                          shared_memory_array_size: int, shared_memory_lock,
                          synch_point_barrier, num_building:int, path_epw: str, path_energyplus_dir: str):
        """
        Run the EnergyPlus simulation with the shared memory and the synchronization objects.
        :param shared_memory_name: str, The name of the shared memory to access the surface temperatures
        :param shared_memory_timestep_name: str, The name of the shared memory to access the timesteps
        :param shared_memory_array_size: int, The size of the shared memory array
        :param shared_memory_lock: Lock, The lock to limit writing access to the shared memory
        :param synch_point_barrier: Barrier, The barrier to synchronize processes
        :param num_building: number of buildings simulated
        :param path_epw: str, The path to the EPW file
        :param path_energyplus_dir: str, The path to the EnergyPlus directory
        :return: self: EpSimulationInstance, The instance of the simulation to update the one in the manager
        """
        # Point to the shared memory for surface temperature access
        shm = shared_memory.SharedMemory(name=shared_memory_name)
        shared_array = np.ndarray(shared_memory_array_size, dtype=np.float64, buffer=shm.buf)

        shm_timestep = shared_memory.SharedMemory(name=shared_memory_timestep_name)
        shared_array_timestep = np.ndarray(num_building, dtype=np.float64, buffer=shm_timestep.buf)

        # initialize the EnergyPlus API and simulation state
        self._api = EnergyPlusAPI(running_as_python_plugin=True, path_to_ep_folder=path_energyplus_dir)
        self._state = self._api.state_manager.new_state()

        # request the variables to access schedule and surface temperature values during the simulation
        self.request_variables_before_running_simulation()

        # Make wrapper for the main callback function
        def simulation_callback_function(state):
            """
            Wrapper for the main callback function to pass arguments (which are not allowed in the callback function).
            :param state:
            """
            return self.coupled_simulation_callback_function(state, shared_array, shared_array_timestep,
                                                             shared_memory_lock, synch_point_barrier)

        # Set the callback functions to run at the various moment of the simulation
        self._api.runtime.callback_begin_new_environment(self._state,
                                                         self.initialize_actuator_handler_callback_function)
        self._api.runtime.callback_begin_new_environment(self._state,
                                                         self.init_surface_temperature_handlers_call_back_function)
        self._api.runtime.callback_end_zone_timestep_after_zone_reporting(self._state,
                                                                          simulation_callback_function)

        synch_point_barrier.wait()

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

        return 0
