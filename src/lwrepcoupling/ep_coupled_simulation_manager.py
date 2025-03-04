"""
Class to manage the couped long-wave radiation (LWR) simulation with EnergyPlus among multiple buildings.
"""
import json
import os
import shutil

from scipy.sparse import csr_matrix
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import shared_memory, Manager
from typing import List

import numpy as np

from .ep_simulation_instance_with_shared_memory import EpSimulationInstance
from lwrepcoupling.utils.utils_resolution_matrices import check_matrices,compute_total_vf


class EpLwrSimulationManager:
    # Constants
    CONFIG_BUILDING_ID_LIST = "building_id_list"
    CONFIG_BUILDING_ID = "building_id"
    CONFIG_PATH_IDF = "path_idf"
    CONFIG_LIST_OUTDOOR_SURFACE_NAME = "list_outdoor_surface_name"
    CONFIG_NUM_OUTDOOR_SURFACES = "num_outdoor_surfaces"
    CONFIG_NUM_OUTDOOR_SURFACES_PER_BUILDING = "num_outdoor_surfaces_per_building"

    # -----------------------------------------------------#
    # -------------------- Initialization------------------#
    # -----------------------------------------------------#

    def __init__(self, path_output_dir: str, path_epw: str, path_energyplus_dir: str):
        """
        Initialize the EnergyPlus simulation manager for the coupled long-wave radiation (LWR) simulation.
        :param path_output_dir: Path to the output directory, where the simulation will be run. It can be either
         an existing empty directory or a new directory that will be created.
        :param path_epw: Path to the EPW (weather) file for the simulation.
        :param path_energyplus_dir: Path to the EnergyPlus directory, where the EnergyPlus executable is located.
        """
        # Simulation
        self._building_id_list = []
        self._ep_simulation_instance_dict = {}
        # Paths to the output directory, EPW file, and EnergyPlus directory
        self._path_output_dir = None
        self._path_epw = None
        self._path_energyplus_dir = None
        self._set_paths_attributes(path_output_dir, path_epw, path_energyplus_dir)
        self._x_epsilon_matrix = None

    def _set_paths_attributes(self, path_output_dir: str, path_epw: str, path_energyplus_dir: str):
        """
        Set the paths to the output directory, EPW file, and EnergyPlus directory.
        :param path_output_dir:
        :param path_epw:
        :param path_energyplus_dir:
        """
        # Check if the EPW file exists
        if not os.path.exists(path_epw):
            raise FileNotFoundError(f"EPW file not found at {path_epw}")
        self._path_epw = path_epw
        # Check if energyplus directory exists
        if not os.path.exists(path_energyplus_dir):
            raise FileNotFoundError(f"EnergyPlus directory not found at {path_energyplus_dir}.\n"
                                    f"Check if the path is correct. You might have a different verison of EnergyPlus.")
        self._path_energyplus_dir = path_energyplus_dir
        # Check if the output directory exists
        if not os.path.exists(path_output_dir):
            os.makedirs(path_output_dir)
            print(f"Output directory created at {path_output_dir}")
        # Check if not empty
        elif not os.listdir(path_output_dir):
            pass
        else:
            raise FileExistsError(
                f"Output directory already exists at {path_output_dir} and is not empty. Please empty it or choose another directory.")
        self._path_output_dir = path_output_dir

    # -----------------------------------------------------#
    # --------------------- Properties --------------------#
    # -----------------------------------------------------#

    @property
    def num_building(self):
        return len(self._building_id_list)

    @property
    def num_outdoor_surfaces(self):
        return sum([ep_simulation_instance.num_outdoor_surfaces for ep_simulation_instance in
                    self._ep_simulation_instance_dict.values()])

    # -----------------------------------------------------#
    # --------------------- Config file -------------------#
    # -----------------------------------------------------#
    def make_config_file(self, path_dir_config: str, list_building_id: List[str],
                         list_path_idf_file: List[str],
                         list_of_list_outdoor_surface_name: List[List[str]]):
        """
        Make a config file for the LWR-EP coupling simulation manager
        :param path_dir_config: str, the path to the directory of the config file
        :param list_path_idf_file: [str], the list of path to the idf files
        :param list_building_id: [str], the list of building IDs
        :param list_of_list_outdoor_surface_name: [[str]], the list of list of outdoor surface names
        """
        # Check if the directory exists
        if not os.path.exists(path_dir_config):
            raise FileNotFoundError(f"The directory {path_dir_config} does not exist")
        # Check if idf files exist
        for path_idf in list_path_idf_file:
            if not os.path.exists(path_idf):
                raise FileNotFoundError(f"The idf file {path_idf} does not exist")
        # Create the config file
        config_dict = {
            building_id: {
                self.CONFIG_BUILDING_ID: building_id,
                self.CONFIG_PATH_IDF: path_idf,
                self.CONFIG_LIST_OUTDOOR_SURFACE_NAME: list_outdoor_surface_name,
                self.CONFIG_NUM_OUTDOOR_SURFACES_PER_BUILDING: len(list_outdoor_surface_name)
            }
            for building_id, path_idf, list_outdoor_surface_name in
            zip(list_building_id, list_path_idf_file, list_of_list_outdoor_surface_name)
        }

        config_dict[self.CONFIG_BUILDING_ID_LIST] = list_building_id
        config_dict[self.CONFIG_NUM_OUTDOOR_SURFACES] = sum(
            [len(list_outdoor_surface_name) for list_outdoor_surface_name in
             list_of_list_outdoor_surface_name])

        path_config_file = os.path.join(path_dir_config, "ep_lwr_sim_man_config.json")
        with open(path_config_file, 'w') as f:
            json.dump(config_dict, f)

        return path_config_file

    @staticmethod
    def load_config_file(path_config_file: str):
        """

        :param path_config_file:
        :return:
        """
        # Check if the file exists
        if not os.path.exists(path_config_file):
            raise FileNotFoundError(f"The config file {path_config_file} does not exist")
        # Load the config file
        with open(path_config_file, 'r') as f:
            config_dict = json.load(f)

        return config_dict

    # -----------------------------------------------------#
    # --------------------- Methods -----------------------#
    # -----------------------------------------------------#

    def set_up_lwr_coupled_simulation(self, path_config_file: str, vf_matrix: csr_matrix,
                                      eps_matrix: csr_matrix, rho_matrix: csr_matrix, tau_matrix: csr_matrix):
        """
        Set up the coupled long-wave radiation (LWR) simulation with EnergyPlus for all buildings.
        """
        #### Config file
        config_dict = self.load_config_file()

        #### Check if the size of the matrices fits the number of outdoor surfaces
        check_matrices(vf_matrix, eps_matrix, rho_matrix, tau_matrix)
        if vf_matrix.shape[0] != self.num_outdoor_surfaces:
            raise ValueError("The size of the matrix must fit the number of outdoor surfaces.")

        #### Process the matrices

        # get sum of view factors for each surface
        vf_tot_list = compute_total_vf(vf_matrix)


        # get final matrix

        ### Initialize buildings

    def add_building(self, building_id: str, path_idf: str, outdoor_surface_name_list: List[str],
                     outdoor_surface_surrounding_surface_vf_dict: dict, outdoor_surface_sky_vf_dict: dict,
                     outdoor_surface_ground_vf_dict: dict, vf_matrix=None, vf_eps_matrix=None):
        """
        Add a building to the simulation manager.
        :param building_id: The building ID.
        :param path_idf: The path to the IDF file for the building.
        :param outdoor_surface_name_list: List of outdoor surface names in the IDF file.
        :param outdoor_surface_surrounding_surface_vf_dict: Dictionary of view factors from outdoor surfaces to other
        outdoor surfaces.
        :param outdoor_surface_sky_vf_dict: Dictionary of view factors from outdoor surfaces to the sky.
        :param outdoor_surface_ground_vf_dict: Dictionary of view factors from outdoor surfaces to the ground.
        :param vf_matrix: The view factor matrix for the building.
        :param vf_eps_matrix: The view factor matrix for the building with emissivity values.
        """
        # Create an EnergyPlus simulation instance for the building
        building_output_dir = os.path.join(self._path_output_dir, f"building_{building_id}")
        ep_simulation_instance = EpSimulationInstance(
            identifier=building_id,
            path_idf=path_idf,
            path_output_dir=building_output_dir,
            simulation_index=self.num_building
        )
        # Make the folder for the building
        if not os.path.exists(building_output_dir):
            os.makedirs(building_output_dir)
        else:
            shutil.rmtree(building_output_dir)
            os.makedirs(building_output_dir)

        ep_simulation_instance.set_outdoor_surfaces_and_view_factors(
            outdoor_surface_name_list=outdoor_surface_name_list,
            outdoor_surface_surrounding_surface_vf_dict=outdoor_surface_surrounding_surface_vf_dict,
            outdoor_surface_sky_vf_dict=outdoor_surface_sky_vf_dict,
            outdoor_surface_ground_vf_dict=outdoor_surface_ground_vf_dict,
            manager_num_outdoor_surfaces=self.num_outdoor_surfaces
        )
        # Generate a copy of the IDF with additional strings for the LWR computation
        ep_simulation_instance.adjust_idf()
        # Set LWR VF matrices
        ep_simulation_instance.set_vf_matrices(vf_matrix, vf_eps_matrix)
        # Add the building to the simulation manager
        self._building_id_list.append(building_id)
        # Todo: save tp pkl instead
        path_pkl_file = ep_simulation_instance.to_pkl(
            path_folder=self._path_output_dir)  # change path to output
        self._ep_simulation_instance_dict[building_id] = path_pkl_file

    def add_y_matrix(self, f_star_matrix, f_star_epsilon_matrix, epsilon_matrix):
        """
        """
        # Check if the matrices are of the same size
        if f_star_matrix.shape != f_star_epsilon_matrix.shape or f_star_matrix.shape != epsilon_matrix.shape:
            raise ValueError("The matrices must be of the same size.")
        # Check if the matrices are square
        if f_star_matrix.shape[0] != f_star_matrix.shape[1]:
            raise ValueError("The matrices must be square.")
        # Check if the size of the matrix fits the number of outdoor surfaces
        if f_star_matrix.shape[0] != self.num_outdoor_surfaces:
            raise ValueError("The size of the matrix must fit the number of outdoor surfaces.")

        # Make intermediate matrices
        vf_tot_list = [1 - sum(f_star_matrix[i, :]) for i in range(f_star_matrix.shape[0])]
        f_vf_tot_epsilon_matrix_inv = np.zeros((f_star_matrix.shape[0], f_star_matrix.shape[0]))
        for i in range(f_star_matrix.shape[0]):
            try:
                f_vf_tot_epsilon_matrix_inv[i, i] = 1 / (vf_tot_list[i] * epsilon_matrix[i][i])
            except ZeroDivisionError:
                f_vf_tot_epsilon_matrix_inv[i, i] = 0
                """ Set to zero if the denominator is zero, in the equation it is equivalen to say the surface 
                is not receiving  any LWR from the other surfaces"""

    def run_lwr_coupled_simulation(self):
        """
        Run the coupled long-wave radiation (LWR) simulation with EnergyPlus for all buildings in parallel and synchronously.
        :return:
        """

        # To do: Make sure it's the proper size
        shared_memory_array_size = self.num_outdoor_surfaces

        # Run the simulation under a Manager context to share memory, locks, and barriers
        with Manager() as manager:

            # Initialize a lock to limit writing access to shared memory
            shared_memory_lock = manager.Lock()  # Todo: Check if it's necessary as no overlapping writing is done
            # Initialize a barrier to synchronize processes, when called with .wait() all processes will wait until all
            # processes have reached the barrier
            synch_point_barrier = manager.Barrier(self.num_building)
            # Create shared memory for float64 data (enough for all processes' lists)
            shm = shared_memory.SharedMemory(create=True,
                                             size=shared_memory_array_size * np.float64().itemsize)

            # Run the EnergyPlus simulations in parallel for all buildings, monitored by the EnergyPlus API
            results_list = []
            try:
                num_workers = self.num_building  # One process per building, as they should all be run in parallel
                # Start tasks
                with ProcessPoolExecutor(max_workers=num_workers) as executor:
                    futures = [
                        executor.submit(
                            ep_simulation_instance.run_ep_simulation,
                            shared_memory_name=shm.name,
                            shared_memory_array_size=shared_memory_array_size,
                            shared_memory_lock=shared_memory_lock,
                            synch_point_barrier=synch_point_barrier,
                            path_epw=self._path_epw,
                            path_energyplus_dir=self._path_energyplus_dir
                        )
                        for ep_simulation_instance in self._ep_simulation_instance_dict.values()
                    ]
                    # Wait for all processes to complete
                    for future in futures:
                        try:
                            results_list.append(future.result())
                        except Exception as e:
                            print(f"Task generated an exception: {e}")


            finally:
                # Cleanup
                shm.close()
                shm.unlink()

        return results_list
