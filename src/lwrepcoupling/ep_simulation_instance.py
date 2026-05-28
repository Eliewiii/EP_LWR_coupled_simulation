"""
Class to represent an instance of a simulation of the EP model, generating the idf additional strings,
managing the handlers etc.
"""

import logging
import pickle
import shutil
import sys
from multiprocessing import shared_memory
from multiprocessing.synchronize import Barrier as MpBarrier
from pathlib import Path
from threading import BrokenBarrierError
from typing import List, Optional

import numpy as np

from .lwr_idf_additionnal_strings import (
    SurfaceAddStringConfig,
    generate_surface_lwr_idf_additional_string,
    name_surrounding_surface_temperature_schedule,
)
from .pyenergyplus.api import EnergyPlusAPI
from .schemas import BuildingInput, CompiledBuildingState

logger = logging.getLogger(__name__)


class EpSimulationBlueprint:
    """The lightweight, picklable data blueprint that lives in the parent process.

    Tracks all the static configuration, geometry mapping metadata, and arrays
    needed by the physics solvers. It manages its own serialization during
    the workspace preprocessing pipeline phase.
    """

    def __init__(
        self,
        identifier: str,
        simulation_index: int,
        surface_index_min: int,
        surface_index_max: int,
        outdoor_surface_names: List[str],
        resolution_mtx: np.ndarray,
    ):
        self.identifier = identifier

        # Synchronization properties
        self.simulation_index = simulation_index
        self.surface_index_min = surface_index_min
        self.surface_index_max = surface_index_max

        # Geometry arrays (populated during your preprocessing stage)
        self.outdoor_surface_names = outdoor_surface_names
        self.resolution_mtx = resolution_mtx

    # -----------------------------------------------------------------
    # SERIALIZATION LAYER
    # -----------------------------------------------------------------
    def to_pkl(self, destination_file_path: Path) -> str:
        """Serializes this blueprint configuration securely to a binary pickle file.

        Args:
            destination_file_path: The absolute target path for the file
                (e.g., /workspace/runs_dir/building_0.pkl).

        Returns:
            str: The string representation of the saved file path.
        """
        logger.info("Serializing preprocessing blueprint token to disk for: %s", self.identifier)

        # Ensure parent directories physically exist on disk before writing binary payloads
        destination_file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(destination_file_path, "wb") as f:
            # HIGHEST_PROTOCOL is essential here; it compresses multi-dimensional
            # NumPy arrays (like resolution_mtx) into incredibly efficient binary structures.
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

        return str(destination_file_path)

    @classmethod
    def from_pkl(cls, source_file_path: Path) -> "EpSimulationBlueprint":
        """Deserializes a saved blueprint binary file back into a live Python instance.

        This method will be called down the line within your child processes
        to instantly recover the preprocessed state.
        """
        if not source_file_path.is_file():
            raise FileNotFoundError(f"Target simulation blueprint file missing: {source_file_path}")

        with open(source_file_path, "rb") as f:
            instance = pickle.load(f)

        if not isinstance(instance, cls):
            raise TypeError(
                f"Loaded object type mismatch. Expected {cls.__name__}, got {type(instance).__name__}"
            )

        return instance

    # -----------------------------------------------------------------
    # PREPROCESSING LIFE CYCLE PIPELINE
    # -----------------------------------------------------------------
    @classmethod
    def init_and_preprocess_to_pkl(
        cls,
        building_input: BuildingInput,
        simulation_index: int,
        min_surface_index: int,
        max_surface_index: int,
        resolution_mtx: np.ndarray,
        srd_vf_list: List[float],
        runs_dir: Path,
        pkl_file_path: Path,
        sky_vf_list: Optional[List[float]] = None,
        ground_vf_list: Optional[List[float]] = None,
    ) -> None:
        """Initializes a simulation blueprint, handles text preprocessing, and serializes to a file.

        This factory method coordinates the complete pipeline lifecycle for offline
        workspace generation. It instantiates the tracking data container, duplicates
        and adjusts the physical EnergyPlus IDF text configuration with custom LWR strings,
        and dumps the frozen state to a highly optimized binary pickle token.

        Args:
            building_input: A data object containing target building definitions,
                including the primitive IDF path and list of outdoor surface names.
            simulation_index: The globally tracked unique sequence integer of
                the building within the simulation execution manager.
            min_surface_index: The lower bound index constraint indicating where this
                building's outdoor surfaces begin writing within the shared memory vector.
            max_surface_index: The upper bound index constraint indicating where this
                building's outdoor surfaces finish writing within the shared memory vector.
            resolution_mtx: A multi-dimensional NumPy array containing the precomputed
                radiosity/geometry resolution matrices for the physics solvers.
            srd_vf_list: A list of float view factors mapped directly to surrounding
                built surfaces for each outdoor boundary.
            runs_dir: The target absolute directory path where the simulation of all buildings will
            be executed (e.g., /path/to/workspace/runs/).
            pkl_file_path: The target absolute destination path where the binary blueprint
                token should be saved (e.g., /path/to/workspace/runs/building_0.pkl).
            sky_vf_list: Optional list of float view factors to the open sky vault.
                Defaults to None.
            ground_vf_list: Optional list of float view factors to the localized terrain ground.
                Defaults to None.

        Raises:
            ValidationError: If the view factor configurations fail internal
                summation limits or geometry constraint integrity guards.
            OSError: If copying the source IDF or writing data payloads to disk
                fails due to OS permissions or missing directories.
        """
        # 1. Instantiate the object container
        ep_simulation_blueprint_obj = cls(
            identifier=building_input.building_id,
            simulation_index=simulation_index,
            surface_index_min=min_surface_index,
            surface_index_max=max_surface_index,
            outdoor_surface_names=building_input.outdoor_surface_names,
            resolution_mtx=resolution_mtx,
        )
        # 2. Complete the physical file duplication and text adjustments on disk
        ep_simulation_blueprint_obj._adjust_idf(
            runs_dir=runs_dir,
            idf_path=building_input.idf_path,
            srd_vf_list=srd_vf_list,
            sky_vf_list=sky_vf_list,
            ground_vf_list=ground_vf_list,
        )
        pkl_file_path = CompiledBuildingState.derive_instance_pkl_path(
            runs_dir=runs_dir, building_index=simulation_index
        )

        # 3. Serialize and save the blueprint configuration asset to disk
        ep_simulation_blueprint_obj.to_pkl(pkl_file_path)

    def _adjust_idf(
        self,
        runs_dir: Path,
        idf_path: Path,
        srd_vf_list: List[float],
        sky_vf_list: Optional[List[float]] = None,
        ground_vf_list: Optional[List[float]] = None,
    ) -> None:
        """
        Handles the physical duplication of the original IDF file and appends the generated LWR additional strings.
        This method performs the necessary text manipulations to inject the LWR coupling definitions into the EnergyPlus configuration.

        Args:
            runs_dir: The target absolute directory path where the simulation of all buildings will
            be executed (e.g., /path/to/workspace/runs/).
            idf_path: The absolute path to the original IDF file that serves as the template for this building's simulation.
            srd_vf_list: A list of float view factors mapped directly to surrounding
                built surfaces for each outdoor boundary.
            sky_vf_list: Optional list of float view factors to the open sky vault.
                Defaults to None.
            ground_vf_list: Optional list of float view factors to the localized terrain ground.
                Defaults to None.
        Raises:
            ValidationError: If the view factor configurations fail internal
                summation limits or geometry constraint integrity guards.
            OSError: If copying the source IDF or writing data payloads to disk
                fails due to OS permissions or missing directories.
        """
        additional_strings = self._generate_idfs_additional_strings(
            srd_vf_list=srd_vf_list, sky_vf_list=sky_vf_list, ground_vf_list=ground_vf_list
        )
        adjusted_idf_path = CompiledBuildingState.derive_idf_path(
            runs_dir=runs_dir, building_index=self.simulation_index
        )

        shutil.copy(idf_path, adjusted_idf_path)
        # Add the additional strings to the IDF file
        with open(adjusted_idf_path, "a", encoding="utf-8") as file:
            file.write(additional_strings)

    def _generate_idfs_additional_strings(
        self,
        srd_vf_list: List[float],
        sky_vf_list: Optional[List[float]] = None,
        ground_vf_list: Optional[List[float]] = None,
    ) -> str:
        """
        Generate the additional strings to add to the IDF file for the LWR coupling.
        Consist of the generation of a SurfaceProperty:LocalEnvironment and a SurfaceProperty:SurroundingSurfaces for each
        outdoor surface, with temperature schedule for the surrounding surfaces to be actuated to account for the LWR.
        :return:
        """
        additional_strings = ""
        for i, surface_name in enumerate(self.outdoor_surface_names):
            add_string_config = SurfaceAddStringConfig(
                surface_name=surface_name,
                cumulated_ext_surf_view_factor=srd_vf_list[i],
                sky_view_factor=sky_vf_list[i] if sky_vf_list else None,
                ground_view_factor=ground_vf_list[i] if ground_vf_list else None,
            )
            additional_strings += generate_surface_lwr_idf_additional_string(add_string_config)

        return additional_strings


class EpSimulationRuntimeWorker:
    """The real-time execution engine born exclusively inside the spawned child process.

    All properties required by the runtime callbacks are either fully initialized
    here or delegated smoothly to the blueprint via properties.
    """

    def __init__(
        self,
        blueprint: EpSimulationBlueprint,
        energyplus_dir: Path,
        time_step: int,
        num_buildings: int,
        num_total_surfaces: int,
        shared_memory_temperatures_name: str,
        shared_memory_timesteps_name: str,
        synch_point_barrier: MpBarrier,
    ):
        # 1. Store the underlying data blueprint configuration
        self._blueprint = blueprint

        # 2. IMMEDIATELY initialize volatile runtime attributes (No more '= None')
        self._api = EnergyPlusAPI(
            running_as_python_plugin=True, path_to_ep_folder=str(energyplus_dir)
        )
        self._state = self._api.state_manager.new_state()

        # 3. IMMEDIATELY bind to operating system shared memory
        self._shm_temp = shared_memory.SharedMemory(name=shared_memory_temperatures_name)
        self._shared_array_temperature = np.ndarray(
            num_total_surfaces, dtype=np.float64, buffer=self._shm_temp.buf
        )

        self._shm_time = shared_memory.SharedMemory(name=shared_memory_timesteps_name)
        self._shared_array_timestep = np.ndarray(
            num_buildings, dtype=np.float64, buffer=self._shm_time.buf
        )

        self._synch_point_barrier = synch_point_barrier

        # 4. Runtime state lists that grow during execution
        self._schedule_actuator_handle_list: list = []
        self._surface_temp_handler_list: list = []
        self._surrounding_surface_temperature_schedule_temperature_handler_list: list = []

        # Runtime Tracking Flags
        self._warmup_started = False
        self._warmup_done = False

        self._time_step = time_step

    # -----------------------------------------------------------------
    # PROPERTY DELEGATION ZONE
    # Exposes blueprint data so your callback loop remains unchanged
    # -----------------------------------------------------------------
    @property
    def identifier(self) -> str:
        return self._blueprint.identifier

    @property
    def simulation_index(self) -> int:
        return self._blueprint.simulation_index

    @property
    def surface_index_min(self) -> int:
        return self._blueprint.surface_index_min

    @property
    def surface_index_max(self) -> int:
        return self._blueprint.surface_index_max

    @property
    def outdoor_surface_names(self) -> list[str]:
        return self._blueprint.outdoor_surface_names

    @property
    def resolution_mtx(self) -> np.ndarray | None:
        return self._blueprint.resolution_mtx

    # -----------------------------------------------------------------
    # EXECUTION ENGINE & CALLBACKS
    # -----------------------------------------------------------------
    @classmethod
    def run_coupled_simulation_from_ep_instance(
        cls,
        ep_simulation_blueprint_pkl_path: Path,
        *,
        epw_path: Path,
        energyplus_dir: Path,
        time_step: int,
        output_dir: Path,
        idf_path: Path,
        num_buildings: int,
        num_total_surfaces: int,
        shared_memory_temperatures_name: str,
        shared_memory_timesteps_name: str,
        synch_point_barrier: MpBarrier,
    ) -> int:
        """The entrypoint executed inside the spawned process boundary."""
        try:
            # 1. Load the un-modified blueprint template from disk
            blueprint = EpSimulationBlueprint.from_pkl(ep_simulation_blueprint_pkl_path)

            # 2. TODO modify     Perform your mandatory text/IDF modifications on disk

            # 3. Construct the runtime worker (Instantiates ALL attributes cleanly)
            worker = cls(
                blueprint=blueprint,
                energyplus_dir=energyplus_dir,
                time_step=time_step,
                num_buildings=num_buildings,
                num_total_surfaces=num_total_surfaces,
                shared_memory_temperatures_name=shared_memory_temperatures_name,
                shared_memory_timesteps_name=shared_memory_timesteps_name,
                synch_point_barrier=synch_point_barrier,
            )

            # 4. Fire the simulation execution context
            exit_code = worker.execute(
                epw_path=epw_path,
                output_dir=output_dir,
                idf_path=idf_path,
            )

        except Exception as e:
            logger.exception("Catastrophic worker process failure: %s", e)
            sys.exit(1)

        return exit_code

    def execute(
        self,
        epw_path: Path,
        output_dir: Path,
        idf_path: Path,
    ) -> int:
        """Hooks up closures and runs the compiled C++ engine loop."""

        logging.info("Starting EnergyPlus simulation for building [%s]", self.identifier)

        self._request_variables_before_running_simulation()

        self._api.runtime.callback_begin_new_environment(
            self._state, self._initialize_actuator_handler_callback_function
        )
        self._api.runtime.callback_begin_new_environment(
            self._state, self._init_surface_temperature_handlers_call_back_function
        )
        self._api.runtime.callback_end_zone_timestep_after_zone_reporting(
            self._state, self._coupled_simulation_callback_function
        )

        try:
            self._synch_point_barrier.wait()
        except BrokenBarrierError:
            sys.exit(1)

        try:
            logger.info("Launching EnergyPlus engine for building: %s", self.identifier)
            # Run native C++ engine loop
            exit_code = self._api.runtime.run_energyplus(
                self._state,
                [
                    "-r",
                    "-w",
                    str(epw_path),
                    "-d",
                    str(output_dir),
                    str(idf_path),
                ],
            )
        except Exception as e:
            logger.critical("Catastrophic error caught during EnergyPlus runtime loop: %s", e)
            sys.exit(1)
        finally:
            # Cleanup open handles
            self._shm_temp.close()
            self._shm_time.close()

        return exit_code

    def _request_variables_before_running_simulation(self):
        """
        Request the variables to access the surface temperature of the outdoor surfaces during the simulation.
        """
        for surface_name in self.outdoor_surface_names:
            self._api.exchange.request_variable(
                self._state, "SURFACE OUTSIDE FACE TEMPERATURE", surface_name
            )

    # ---------------------------------------------------------------#
    # --- Initialize Variable Request and Init Callback Functions ---#
    # ---------------------------------------------------------------#

    def _initialize_actuator_handler_callback_function(self) -> None:
        """Initialize the actuator handlers for the surrounding surface temperature schedules.

        Should be run at the end of the warmup period or beginning of environment.
        """
        missing_schedules_surfaces: list[str] = []

        for surface_name in self.outdoor_surface_names:
            target_schedule_name = name_surrounding_surface_temperature_schedule(surface_name)

            schedule_actuator_handle = self._api.exchange.get_actuator_handle(
                self._state,  # Use the state pointer passed directly by the C++ callback engine
                "Schedule:Constant",
                "Schedule Value",
                target_schedule_name,
            )

            if schedule_actuator_handle == -1:
                missing_schedules_surfaces.append(surface_name)
                continue  # Keep searching to collect ALL failures in one run!

            self._schedule_actuator_handle_list.append(schedule_actuator_handle)

        # Circuit breaker: If any handles failed to initialize, raise a comprehensive error
        if missing_schedules_surfaces:
            raise RuntimeError(
                f"Component Mapping Failure: Failed to retrieve API actuator handles for "
                f"{len(missing_schedules_surfaces)} schedules in building '{self.identifier}'. "
                f"Missing targets: {missing_schedules_surfaces}. Verify that the IDF macro injector "
                f"appended the correct strings during the preprocessing phase."
            )

    def _init_surface_temperature_handlers_call_back_function(self):
        """
        Initialize the handlers to access the surface temperatures of the outdoor surfaces.
        Should be run at the end of the warmup period.
        """
        for surface_name in self.outdoor_surface_names:
            self._surface_temp_handler_list.append(
                self._api.exchange.get_variable_handle(
                    self._state, "SURFACE OUTSIDE FACE TEMPERATURE", surface_name
                )
            )

    def _init_surrounding_surface_schedule_handlers_call_back_function_for_testing(self):
        """
        Initialize the handlers to access the schedule values of the surrounding surface temperatures.
        Should be run at the end of the warmup period.
        For testing purposes only as it is not needed for the LWR computation.
        """
        for surface_name in self.outdoor_surface_names:
            self._surrounding_surface_temperature_schedule_temperature_handler_list.append(
                self._api.exchange.get_variable_handle(
                    self._state,
                    "Schedule Value",
                    name_surrounding_surface_temperature_schedule(surface_name),
                )
            )

    # ------------------------------#
    # --- Main Callback Function ---#
    # ------------------------------#
    def _coupled_simulation_callback_function(
        self,
        state,
        shared_array,
        shared_array_timestep,
        shared_memory_lock,
        synch_point_barrier,
        time_step: float,
    ):
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
            if np.isclose(current_time, time_step, rtol=1e-01, atol=1e-02) or (
                2 * time_step > current_time > time_step
            ):
                self._warmup_done = True
            else:
                return
        # Get the surface temperatures of all the surfaces
        surface_temperatures_list = self.get_surface_temperature_of_all_outdoor_surfaces_in_kelvin()

        try:
            # Wait for all other buildings to catch up to this timestep
            synch_point_barrier.wait()
        except BrokenBarrierError:
            logger.warning(
                "Simulation synchronization grid collapsed due to a sister process crash. "
                "Aborting execution loop cleanly."
            )
            # Perform any local cleanup if needed, then exit the process orderly
            sys.exit(1)

        # write down the surface temperatures the shared memory
        with shared_memory_lock:
            # todo: need to indicated properly the start and end index of the shared memory
            # Here we are writing the list as a slice of the shared memory
            np.copyto(
                shared_array[self._surface_index_min : self._surface_index_max + 1],
                np.array(surface_temperatures_list) ** 4,
            )  # directly give the temperatures power 4

            np.copyto(
                shared_array_timestep[self._simulation_index : self._simulation_index + 1],
                np.array([current_time]),
            )

        try:
            """
            Wait for all other buildings to catch up to this timestep.
            If another process has crashed, the barrier will be broken and a BrokenBarrierError will 
            be raised, which is caught to exit the process cleanly instead of hanging.
            """
            synch_point_barrier.wait()
        except BrokenBarrierError:
            logger.warning(
                "Simulation synchronization grid collapsed due to a sister process crash. "
                "Aborting execution loop cleanly."
            )
            # Perform any local cleanup if needed, then exit the process orderly
            sys.exit(1)

        if self._simulation_index == 0:
            # Compute timestep indices by rounding to nearest integer
            floor_timestep_indices = [
                math.floor(round(ts, 4) / time_step) for ts in shared_array_timestep
            ]
            ceil_timestep_indices = [
                math.ceil(round(ts, 4) / time_step) for ts in shared_array_timestep
            ]
            # Check that all indices are the same
            if (
                max(floor_timestep_indices) - min(floor_timestep_indices) > 1
                and max(ceil_timestep_indices) - min(ceil_timestep_indices) > 1
            ):
                raise ValueError(
                    f"Timestep mismatch between simulations:\n"
                    f"timesteps: {shared_array_timestep}\n"
                )
            if shared_array_timestep[0] % int(24) == 0:
                print(f"Current day: {int(shared_array_timestep[0] // 24)}")

        # Compute a somewhat mean surface temperature, to test the synchronization, only one surface per building will be used
        list_srd_mean_radiant_temperature_in_c = self.compute_srd_mean_radiant_temperatures_in_c(
            temperature_p4_vector=shared_array
        )  # convert back to Celsius
        # Set the surrounding surface temperature to the average of the surface temperatures
        for i, srd_mrt in enumerate(list_srd_mean_radiant_temperature_in_c):
            self._api.exchange.set_actuator_value(
                state, self._schedule_actuator_handle_list[i], srd_mrt
            )
