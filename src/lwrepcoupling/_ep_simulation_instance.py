"""
Class to represent an instance of a simulation of the EP model, generating the idf additional strings,
managing the handlers etc.
"""

import logging
import sys
from math import ceil, floor
from multiprocessing import shared_memory
from pathlib import Path
from threading import BrokenBarrierError
from typing import Any, List

import numpy as np
from pyenergyplus.api import EnergyPlusAPI

from ._schemas import (
    CompiledBuildingState,
    EpSimulationRuntimeConfig,
    SynchronizerBarrier,
)
from ._utils.utils_idf_additional_strings import (
    name_surrounding_surface_temperature_schedule,
)

logger = logging.getLogger(__name__)


class EpSimulationRuntimeWorker:
    """The real-time execution engine born exclusively inside the spawned child process.

    All properties required by the runtime callbacks are either fully initialized
    here or delegated smoothly to the blueprint via properties.
    """

    # 1. Structural Dependency Types
    _building_state: CompiledBuildingState
    _synch_point_barrier: SynchronizerBarrier

    # 2. Foreign C-API State Management Types
    _api: EnergyPlusAPI
    _state: Any

    # 3. Shared Memory Buffer Views (Crucial for Matrix Safety!)
    _shm_temp: shared_memory.SharedMemory
    _shared_array_temperature: np.ndarray

    _shm_time: shared_memory.SharedMemory
    _shared_array_timestep: np.ndarray

    _local_temp_buffer: (
        np.ndarray
    )  # Local buffer to hold surface temperatures before writing to shared memory

    # 4. Runtime C++ Handle Tracking Containers (Explicitly typed lists)
    _schedule_actuator_handle_list: list[int]
    _surface_temp_handler_list: list[int]
    _surrounding_surface_temperature_schedule_temperature_handler_list: list[int]

    # 5. Core Numerical Simulation Metrics & Inverses
    _num_ts_per_h: int
    _time_step: float

    # 6. Stateful Workflow Flags
    _warmup_started: bool
    _warmup_done: bool

    def __init__(
        self,
        building_state: CompiledBuildingState,
        runs_dir: Path,
        num_ts_per_h: int,
        num_buildings: int,
        num_total_surfaces: int,
        shared_memory_temperatures_name: str,
        shared_memory_timesteps_name: str,
        synch_point_barrier: SynchronizerBarrier,
    ) -> None:  # __init__ always returns None
        # 1. Store the underlying data blueprint configuration
        self._building_state = building_state

        self._resolution_mtx = np.load(self._building_state.get_sub_mtx_path(runs_dir=runs_dir))

        # 2. IMMEDIATELY initialize volatile runtime attributes (No more '= None')
        self._api = EnergyPlusAPI(running_as_python_plugin=True)
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

        self._local_temp_buffer = np.zeros(self.surface_index_max - self.surface_index_min + 1)

        self._synch_point_barrier = synch_point_barrier

        # 4. Runtime state lists that grow during execution (Using specific type lists)
        self._schedule_actuator_handle_list = []
        self._surface_temp_handler_list = []
        self._surrounding_surface_temperature_schedule_temperature_handler_list = []

        # Runtime Tracking Flags
        self._warmup_started = False
        self._warmup_done = False

        self._num_ts_per_h = num_ts_per_h
        self._time_step = 1.0 / num_ts_per_h

    # -----------------------------------------------------------------
    # PROPERTY DELEGATION ZONE
    # Exposes blueprint data so your callback loop remains unchanged
    # -----------------------------------------------------------------
    @property
    def identifier(self) -> str:
        return self._building_state.building_id

    @property
    def simulation_index(self) -> int:
        return self._building_state.building_index

    @property
    def surface_index_min(self) -> int:
        return self._building_state.surface_index_min

    @property
    def surface_index_max(self) -> int:
        return self._building_state.surface_index_max

    @property
    def outdoor_surface_names(self) -> list[str]:
        return self._building_state.outdoor_surface_names

    # -----------------------------------------------------------------
    # EXECUTION ENGINE & CALLBACKS
    # -----------------------------------------------------------------
    @classmethod
    def run_coupled_simulation(cls, config: EpSimulationRuntimeConfig) -> int:
        """The entrypoint executed inside the spawned process boundary."""
        try:
            # 1. Construct the runtime worker (Instantiates ALL attributes cleanly)
            worker = cls(
                building_state=config.building_state,
                runs_dir=config.runs_dir,
                num_ts_per_h=config.num_ts_per_h,
                num_buildings=config.num_buildings,
                num_total_surfaces=config.num_total_surfaces,
                shared_memory_temperatures_name=config.shared_memory_temperatures_name,
                shared_memory_timesteps_name=config.shared_memory_timesteps_name,
                synch_point_barrier=config.synch_point_barrier,
            )

            output_dir = worker._building_state.get_output_dir(runs_dir=config.runs_dir)
            idf_path = worker._building_state.get_idf_path(runs_dir=config.runs_dir)

            # 2. Runs simulation execution context
            exit_code = worker.execute(
                epw_path=config.epw_path,
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

        # 1- Set up the callback functions to be triggered at the right moments during the simulation loop
        logging.info("Starting EnergyPlus simulation for building [%s]", self.identifier)

        self._request_variables_before_running_simulation()

        self._api.runtime.callback_begin_new_environment(
            self._state,
            self._initialize_actuator_handler_callback_function,  # type: ignore
        )
        self._api.runtime.callback_begin_new_environment(
            self._state,
            self._init_surface_temperature_handlers_call_back_function,  # type: ignore
        )
        self._api.runtime.callback_end_zone_timestep_after_zone_reporting(
            self._state,
            self._coupled_simulation_callback_function,  # type: ignore
        )

        try:
            self._synch_point_barrier.wait()
        except BrokenBarrierError:
            sys.exit(1)

        # 2- Run the simulation in EnergyPlus native C++ runtime loop, with the callbacks now hooked up and ready to go
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

    # ---------------------------------------------------#
    # --- Main Callback Function and helper functions ---#
    # ---------------------------------------------------#
    def _coupled_simulation_callback_function(self):
        """
        Function to run at the end (or beginning) of each time step, to update the schedule values and surrounding surface temperatures.
        This function is a test version that will not perform the LWR computation but will write the surface temperatures and update the schedules
        to test the synchronization of the shared memory and the barrier.
        :return:
        """

        # prevent from runnning the function if the actuator handlers are not initialized (at warmup)
        if not self._schedule_actuator_handle_list:
            return
        current_time = self._api.exchange.current_sim_time(self._state)

        if not self._warmup_started:
            self._warmup_started = True
            return
        if not self._warmup_done:
            if np.isclose(current_time, self._time_step, rtol=1e-01, atol=1e-02) or (
                2 * self._time_step > current_time > self._time_step
            ):
                self._warmup_done = True
            else:
                return
        # Get the surface temperatures of all the surfaces
        surface_temperatures_list = (
            self._get_surface_temperature_of_all_outdoor_surfaces_in_kelvin()
        )

        try:
            # Wait for all other buildings to catch up to this timestep
            self._synch_point_barrier.wait()
        except BrokenBarrierError:
            logger.warning(
                "Simulation synchronization grid collapsed due to a sister process crash. "
                "Aborting execution loop cleanly."
            )
            # Perform any local cleanup if needed, then exit the process orderly
            sys.exit(1)

        # write down the surface temperatures the shared memory
        np.copyto(
            self._shared_array_temperature[self.surface_index_min : self.surface_index_max + 1],
            np.array(surface_temperatures_list) ** 4,
        )  # directly give the temperatures power 4

        np.copyto(
            self._shared_array_timestep[self.simulation_index],
            np.array([current_time]),
        )

        try:
            """
            Wait for all other buildings to catch up to this timestep.
            If another process has crashed, the barrier will be broken and a BrokenBarrierError will 
            be raised, which is caught to exit the process cleanly instead of hanging.
            """
            self._synch_point_barrier.wait()
        except BrokenBarrierError:
            logger.warning(
                "Simulation synchronization grid collapsed due to a sister process crash. "
                "Aborting execution loop cleanly."
            )
            # Perform any local cleanup if needed, then exit the process orderly
            sys.exit(1)

        if self.simulation_index == 0:
            # Compute timestep indices by rounding to nearest integer
            floor_timestep_indices = [
                floor(round(ts, 4) * self._num_ts_per_h) for ts in self._shared_array_timestep
            ]
            ceil_timestep_indices = [
                ceil(round(ts, 4) * self._num_ts_per_h) for ts in self._shared_array_timestep
            ]
            # Check that all indices are the same
            if (
                max(floor_timestep_indices) - min(floor_timestep_indices) > 1
                and max(ceil_timestep_indices) - min(ceil_timestep_indices) > 1
            ):
                raise ValueError(
                    f"Timestep mismatch between simulations:\n"
                    f"timesteps: {self._shared_array_timestep}\n"
                )
            if self._shared_array_timestep[0] % int(24) == 0:
                logger.info("Current day: %s", int(self._shared_array_timestep[0] // 24))

        # Compute the equivalent surrounding surface temperature in Celsius for each outdoor surface
        self._compute_srd_sur_eq_temp_in_c(
            temp_k_p4_vector=self._shared_array_temperature
        )  # convert back to Celsius

        # Set the equivalent surrounding surface temperature
        for i, srd_mrt in enumerate(self._local_temp_buffer):
            self._api.exchange.set_actuator_value(
                self._state, self._schedule_actuator_handle_list[i], srd_mrt
            )

    def _get_surface_temperature_of_all_outdoor_surfaces_in_kelvin(self) -> List[float]:
        """
        Reads the surface temperature of all the outdoor surfaces and store them in a list.
        :return: list,  List of surface temperatures
        """
        surface_temperatures_list = []
        for i, _ in enumerate(self.outdoor_surface_names):
            surface_temperatures_list.append(
                self._api.exchange.get_variable_value(
                    self._state, self._surface_temp_handler_list[i]
                )
                + 273.15
            )  # convert to Kelvin
        return surface_temperatures_list

    def _compute_srd_sur_eq_temp_in_c(self, temp_k_p4_vector: np.ndarray):
        """
        Compute the equivalent surrounding surface temperature for each outdoor surface, using the view factors and the resolution matrix.

        Args:
            temp_k_p4_vector: np.ndarray, vector of the surface temperatures in Kelvin to the power 4

        """

        # TODO: factorize the expression to avaoid unnecessary computations, need to adjust the resolution matrix expression slightly to do so

        self._local_temp_buffer = (
            np.power(
                temp_k_p4_vector[self.surface_index_min : self.surface_index_max + 1]
                - self._resolution_mtx @ temp_k_p4_vector,
                1 / 4,
            )
            - 273.15
        )
