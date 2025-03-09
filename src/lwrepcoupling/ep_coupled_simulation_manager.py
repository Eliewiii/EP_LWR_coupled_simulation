"""
Class to manage the couped long-wave radiation (LWR) simulation with EnergyPlus among multiple buildings.
"""
import json
import os
import shutil
import logging
import pickle

from scipy.sparse import csr_matrix
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import shared_memory, Manager
from typing import List

import numpy as np

from .ep_simulation_instance_with_shared_memory import EpSimulationInstance
from .utils import read_csr_matrices_from_npz, compute_resolution_matrices, check_matrices,check_inversion_parameters, create_dir


class EpLwrSimulationManager:
    # Constants
    CONFIG_FILE_NAME = "ep_lwr_sim_man_config.json"
    CONFIG_BUILDING_ID_LIST = "building_id_list"
    CONFIG_BUILDING_ID = "building_id"
    CONFIG_PATH_IDF = "path_idf"
    CONFIG_LIST_KEY_OUTDOOR_SURFACE_ID = "list_outdoor_surface_name"
    CONFIG_NUM_OUTDOOR_SURFACES = "num_outdoor_surfaces"
    CONFIG_NUM_OUTDOOR_SURFACES_PER_BUILDING = "num_outdoor_surfaces_per_building"
    # Config matrices
    CONFIG_KEY_PATH_VF_MTX = "vf_mtx"
    CONFIG_KEY_PATH_EPS_MTX = "eps_mtx"
    CONFIG_KEY_PATH_RHO_MTX = "rho_mtx"
    CONFIG_KEY_PATH_TAU_MTX = "tau_mtx"
    # Config matrix inversion parameters
    CONFIG_KEY_INV_PARA = "inv_parameters"
    # Sim
    CONFIG_KEY_PATH_OUT_DIR = "path_output"
    CONFIG_KEY_PATH_EP = "path_folder_EnergyPlus"
    CONFIG_KEY_PATH_EPW = "path_epw"

    EP_SIM_PARA_DICT_UNIT_DEFAULT = {
        "path_idf": None
    }

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
        self._building_id_to_path_pkl_dict = {}
        # Paths to the output directory, EPW file, and EnergyPlus directory
        self._path_output_dir = None
        self._path_epw = None
        self._path_energyplus_dir = None

        self._set_paths_attributes(path_output_dir, path_epw, path_energyplus_dir)
        #
        self._num_outdoor_surfaces = 0


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


    def _add_building_to_dict(self,building_id:str,path_to_pkl:str):
        """

        :param building_id:
        :param path_to_pkl:
        :return:
        """

        self._building_id_list.append(building_id)
        self._building_id_to_path_pkl_dict[building_id] = path_to_pkl


    # -----------------------------------------------------#
    # --------------------- Properties --------------------#
    # -----------------------------------------------------#

    @property
    def path_output_dir(self) -> str:
        """Read-only property for the output directory path."""
        return self._path_output_dir

    @property
    def path_epw(self) -> str:
        """Read-only property for the EPW file path."""
        return self._path_epw

    @property
    def path_energyplus_dir(self) -> str:
        """Read-only property for the EnergyPlus directory path."""
        return self._path_energyplus_dir

    @property
    def num_building(self):
        return len(self._building_id_list)

    @property
    def num_outdoor_surfaces(self):
        return self._num_outdoor_surfaces

    # -----------------------------------------------------#
    # -------------- Export and Import --------------------#
    # -----------------------------------------------------#

    def to_pkl(self, path_folder: str, file_name: str = None,
               get_path_only: bool = False) -> str:
        """
        Save the instance to a pickle file.
        :param path_folder: str, the folder path where the pickle file will be saved.
        :param file_name: str, the name of the pickle file.
        :param get_path_only: bool, if True, return the path of the pickle file only, if False, save the instance to the
        pickle file and return the path.
        """
        if file_name is None:
            file_name = f"ep_lwr_simulation_manager.pkl"
        path_pkl_file = os.path.join(path_folder, file_name)
        if not get_path_only:
            with open(path_pkl_file, 'wb') as f:
                pickle.dump(self, f)
        return path_pkl_file

    @staticmethod
    def from_pkl(path_pkl_file) -> "EpLwrSimulationManager":
        """
        Load a RadiativeSurfaceManager object from a pickle file.
        :param path_pkl_file: str, the path of the pickle file.
        :return: RadiativeSurfaceManager, the RadiativeSurfaceManager object.
        """
        with open(path_pkl_file, 'rb') as f:
            ep_simulation_instance = pickle.load(f)
        return ep_simulation_instance

    # -----------------------------------------------------#
    # --------------------- Config file -------------------#
    # -----------------------------------------------------#
    @classmethod
    def make_config_file(cls, path_dir_config: str, path_dir_outputs: str,
                         path_epw_file: str, path_energyplus_dir: str,
                         list_building_id: List[str], list_path_idf_file: List[str],
                         list_of_list_outdoor_surface_name: List[List[str]],
                         path_vf_mtx_crs_npz: str, path_eps_mtx_crs_npz: str,
                         path_rho_mtx_crs_npz: str, path_tau_mtx_crs_npz: str,
                         **kwargs) -> str:
        """
        Creates a JSON configuration file for the LWR-EP coupled simulation manager.

        This method ensures that all required input files and directories exist before
        generating the configuration file. It also validates and integrates optional
        parameters related to the GMRES-based matrix inversion method.

        :param path_dir_config: Path to the directory where the config file will be saved.
        :param path_dir_outputs: Path to the directory where the simulation outputs will be stored.
        :param path_epw_file: Path to the EPW weather file required for the EnergyPlus simulation.
        :param path_energyplus_dir: Path to the directory containing EnergyPlus.
        :param list_building_id: List of building IDs.
        :param list_path_idf_file: List of paths to the IDF (EnergyPlus) files.
        :param list_of_list_outdoor_surface_name: List of lists containing outdoor surface names per building.
        :param path_vf_mtx_crs_npz: Path to the view factor matrix in compressed sparse format.
        :param path_eps_mtx_crs_npz: Path to the emissivity matrix in compressed sparse format.
        :param path_rho_mtx_crs_npz: Path to the reflectivity matrix in compressed sparse format.
        :param path_tau_mtx_crs_npz: Path to the transmissivity matrix in compressed sparse format.
        :param kwargs: Optional parameters for the GMRES-based matrix inversion method.
            - **tol** (float, optional): Overall inverse tolerance (default: 1e-5, valid range: 1e-10 to 1e-2).
            - **maxiter** (int, optional): Maximum number of iterations (default: 150, valid range: 1 to 1000).
            - **rtol** (float, optional): Relative tolerance within iterations on columns  (default: 5e-7, valid range: 1e-10 to 1e-5).
            - **precondition** (bool, optional): Whether to apply preconditioning (default: False).
            - **num_workers** (int, optional): Number of parallel workers (default: 0, valid range: 0 to 64).

        :return: Path to the generated configuration file.
        :raises FileNotFoundError: If required directories or files do not exist.
        :raises ValueError: If invalid parameters are provided in `kwargs`.

        Example usage:
        ```
        EpLwrSimulationManager.make_config_file(
            path_dir_config="config/",
            path_dir_outputs="outputs/",
            path_epw_file="weather.epw",
            path_energyplus_dir="/usr/local/EnergyPlus/",
            list_building_id=["B1", "B2"],
            list_path_idf_file=["building1.idf", "building2.idf"],
            list_of_list_outdoor_surface_name=[["surf1", "surf2"], ["surf3"]],
            path_vf_mtx_crs_npz="vf_matrix.npz",
            path_eps_mtx_crs_npz="eps_matrix.npz",
            path_rho_mtx_crs_npz="rho_matrix.npz",
            path_tau_mtx_crs_npz="tau_matrix.npz",
            tol=1e-6, maxiter=200  # Optional kwargs
        )
        ```
        """

        # Ensure the configuration directory exists
        if not os.path.exists(path_dir_config):
            raise FileNotFoundError(f"The configuration directory '{path_dir_config}' does not exist.")

        # Validate that all IDF files exist
        for path_idf in list_path_idf_file:
            if not os.path.exists(path_idf):
                raise FileNotFoundError(f"The IDF file '{path_idf}' does not exist.")

        # Construct the configuration dictionary
        config_dict = {
            building_id: {
                cls.CONFIG_BUILDING_ID: building_id,
                cls.CONFIG_PATH_IDF: path_idf,
                cls.CONFIG_LIST_KEY_OUTDOOR_SURFACE_ID: list_outdoor_surface_name,
                cls.CONFIG_NUM_OUTDOOR_SURFACES_PER_BUILDING: len(list_outdoor_surface_name),
            }
            for building_id, path_idf, list_outdoor_surface_name in
            zip(list_building_id, list_path_idf_file, list_of_list_outdoor_surface_name)
        }

        # Add global configuration settings
        config_dict[cls.CONFIG_BUILDING_ID_LIST] = list_building_id
        config_dict[cls.CONFIG_NUM_OUTDOOR_SURFACES] = sum(
            len(lst) for lst in list_of_list_outdoor_surface_name)
        config_dict[cls.CONFIG_KEY_PATH_OUT_DIR] = path_dir_outputs
        # Matrices
        config_dict[cls.CONFIG_KEY_PATH_VF_MTX] = path_vf_mtx_crs_npz
        config_dict[cls.CONFIG_KEY_PATH_EPS_MTX] = path_eps_mtx_crs_npz
        config_dict[cls.CONFIG_KEY_PATH_RHO_MTX] = path_rho_mtx_crs_npz
        config_dict[cls.CONFIG_KEY_PATH_TAU_MTX] = path_tau_mtx_crs_npz
        config_dict[cls.CONFIG_KEY_INV_PARA] = check_inversion_parameters(**kwargs)

        config_dict[cls.CONFIG_KEY_PATH_EP] = path_energyplus_dir  # Added EnergyPlus directory
        config_dict[cls.CONFIG_KEY_PATH_EPW] = path_epw_file  # Added EPW file path

        # Define the path to the output configuration file
        path_config_file = os.path.join(path_dir_config, cls.CONFIG_FILE_NAME)

        # Save the configuration dictionary as a JSON file
        with open(path_config_file, 'w') as f:
            json.dump(config_dict, f, indent=4)

        return path_config_file

    @staticmethod
    def _load_config_file(path_config_file: str) -> dict:
        """
        Loads a configuration file in JSON format.

        :param path_config_file: The path to the configuration file.
        :return: A dictionary containing the configuration data.
        :raises FileNotFoundError: If the file does not exist.
        """
        # Check if the file exists
        if not os.path.exists(path_config_file):
            raise FileNotFoundError(f"The config file {path_config_file} does not exist")

        # Load the config file as a dictionary
        with open(path_config_file, 'r') as f:
            config_dict = json.load(f)

        return config_dict

    @classmethod
    def set_up_coupled_lwr_simulation_from_config_file(cls, path_config_file:str, to_pkl:bool = False):
        """

        :param path_config_file:
        :param to_pkl
        :return:
        """
        config_dict = cls._load_config_file(path_config_file=path_config_file)

        sim_manager = cls(path_output_dir=config_dict[cls.CONFIG_KEY_PATH_OUT_DIR], path_epw=config_dict[cls.CONFIG_KEY_PATH_EPW],
                          path_energyplus_dir=config_dict[cls.CONFIG_KEY_PATH_EP])
        # Load and check matrices
        vf_mtx, eps_mtx, rho_mtx, tau_mtx = read_csr_matrices_from_npz(
            config_dict[cls.CONFIG_KEY_PATH_VF_MTX],
            config_dict[cls.CONFIG_KEY_PATH_EPS_MTX],
            config_dict[cls.CONFIG_KEY_PATH_RHO_MTX],
            config_dict[cls.CONFIG_KEY_PATH_TAU_MTX],
        )
        check_matrices(vf_mtx, eps_mtx, rho_mtx, tau_mtx)
        # Ensure the number of surface is consistent accross
        if vf_mtx.shape[0] != config_dict[cls.CONFIG_NUM_OUTDOOR_SURFACES]:
            raise ValueError("The size of the matrix must fit the number of outdoor surfaces.")
        sim_manager._num_outdoor_surfaces = config_dict[cls.CONFIG_NUM_OUTDOOR_SURFACES]
        # Generate the resolution matrices
        resolution_mtx, total_srd_vf_list = compute_resolution_matrices(vf_mtx, eps_mtx, rho_mtx, tau_mtx)
        # Generate EpSimulationInstance for each building
        min_surface_index = 0
        for index, building_id in enumerate(config_dict[sim_manager.CONFIG_BUILDING_ID_LIST]):
            # Make one folder per building
            path_sim_dir_building = os.path.join(sim_manager.path_output_dir,building_id)
            create_dir(path_dir=path_sim_dir_building,overwrite=True)
            # Initialize the EpSimulationInstance and save it to pkl
            max_surface_index = min_surface_index + config_dict[building_id[cls.CONFIG_NUM_OUTDOOR_SURFACES_PER_BUILDING]]-1
            path_pkl_file = EpSimulationInstance.init_and_preprocess_to_pkl(
                identifier= building_id,
                path_output_dir=path_sim_dir_building,
                outdoor_surface_id_list= config_dict[building_id][cls.CONFIG_LIST_KEY_OUTDOOR_SURFACE_ID],
                min_surface_index=min_surface_index,
                max_surface_index=max_surface_index,
                resolution_mtx=resolution_mtx,
                srd_vf_list=total_srd_vf_list
            )
            sim_manager._add_building_to_dict(building_id=building_id,path_to_pkl=path_pkl_file)

            min_surface_index += max_surface_index + 1



        # Save the manager to pkl if needed
        if to_pkl:
            sim_manager.to_pkl(path_folder=sim_manager.path_output_dir)

        return sim_manager


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
        # check that the output dir

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
