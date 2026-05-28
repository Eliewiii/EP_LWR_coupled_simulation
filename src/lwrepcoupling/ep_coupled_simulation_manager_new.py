"""
Class to manage the couped long-wave radiation (LWR) simulation with EnergyPlus among multiple buildings.
"""

import logging
import shutil
import sys
import time
from multiprocessing import Manager, Process, get_context, shared_memory
from pathlib import Path
from typing import NamedTuple

import numpy as np
from scipy.sparse import save_npz

from .ep_simulation_instance_with_shared_memory import EpSimulationInstance
from .schemas import CompiledBuildingState, SimulationInputs, SimulationManifest
from .utils import (
    check_matrices,
    compute_resolution_matrices,
    read_csr_matrices_from_npz,
)
from .utils.utils_io import WorkspaceConflictError, assert_path_is_safe_for_purging

logger = logging.getLogger(__name__)


class WorkerTracker(NamedTuple):
    """Local tracking container to monitor active simulation child processes."""

    building_id: str
    building_index: int
    process: Process


class SimulationCrashError(RuntimeError):
    """Raised when the isolated long-wave radiation simulation crashes."""


class EpLwrSimulationManager:
    """Manages the execution lifecycle of the coupled long-wave radiation (LWR) simulation
    with EnergyPlus.

    This coordinator acts as the central execution manager, encapsulating the
    underlying configuration manifest, orchestrating parallel solver processes,
    and enforcing boundary validation rules during numerical iterations.



    Attributes:
    """

    def __init__(self, manifest: SimulationManifest):
        self._manifest = manifest

    @property
    def workspace_dir(self) -> Path:
        """The absolute path to the active simulation environment."""
        return self._manifest.workspace_dir

    @property
    def energyplus_dir(self) -> Path:
        """The absolute path to the EnergyPlus installation directory."""
        return self._manifest.energyplus_dir

    @property
    def epw_path(self) -> Path:
        """The path to the weather file used in the simulation."""
        return self._manifest.epw_path

    @property
    def time_step(self) -> int:
        """The time step used in the simulation."""
        return self._manifest.time_step

    @property
    def num_buildings(self) -> int:
        """The total number of buildings assigned to this simulation run."""
        return self._manifest.num_buildings

    @property
    def num_total_surfaces(self) -> int:
        """The total global size of the shared-memory array allocation."""
        return self._manifest.num_total_surfaces

    @property
    def buildings(self) -> list[CompiledBuildingState]:
        """The list of building identifiers included in this simulation."""
        return self._manifest.compiled_buildings

    @classmethod
    def compile_and_initialize_workspace(
        cls, inputs: SimulationInputs, overwrite: bool = False
    ) -> "EpLwrSimulationManager":
        """Consumes raw simulation inputs and structurally generates the disk workspace.

        Args:
            inputs: Structured configuration container holding path anchors.
            overwrite: If True, explicitly allows clearing out an existing,
                non-empty workspace folder if it passes fingerprint validation.

        Returns:
            An initialized instance of the simulation manager engine.

        Raises:
            SecurityViolationError: If the target path threatens OS or root safety.
            WorkspaceConflictError: If the folder is occupied and overwrite is disabled
                or if the folder fails structural fingerprint verification.
        """
        # 1. Validate and prepare the target workspace directory, enforcing safety and exclusivity guardrails
        cls._make_target_workspace_dir(
            target_workspace_dir=inputs.workspace_dir, overwrite=overwrite
        )
        working_dir = inputs.workspace_dir.resolve()  # For easier reference in the next steps

        # 2. Check EnergyPlus directory and EPW file existence and move the EPW file into the workspace
        epw_file_name = inputs.epw_path.name
        path_epw_in_workspace = SimulationManifest.derive_epw_path(
            workspace_dir=working_dir, epw_file_name=epw_file_name
        )
        shutil.copy(inputs.epw_path, path_epw_in_workspace)

        # 3. Make the runs directory to hold the building subfolders
        runs_dir = SimulationManifest.derive_runs_dir(working_dir)
        runs_dir.mkdir(exist_ok=False)

        # 4. Use the temporary raw path inputs to load the heavy arrays into memory
        vf_mtx, eps_mtx, rho_mtx, tau_mtx = read_csr_matrices_from_npz(
            inputs.vf_matrix_path,
            inputs.eps_matrix_path,
            inputs.rho_matrix_path,
            inputs.tau_matrix_path,
        )
        check_matrices(vf_mtx, eps_mtx, rho_mtx, tau_mtx)

        num_total_surfaces = inputs.num_total_surfaces

        if vf_mtx.shape[0] != num_total_surfaces:  # Validate matrix size consistency
            raise ValueError("The matrix dimensions must match the number of outdoor surfaces.")

        # 5. Solve the physics system (The heavy linear algebra computation)
        resolution_mtx, total_srd_vf_list = compute_resolution_matrices(
            vf_matrix=vf_mtx,
            eps_matrix=eps_mtx,
            rho_matrix=rho_mtx,
            tau_matrix=tau_mtx,
            inversion_config=inputs.inversion_parameters,
        )

        # 6. Save the compiled resolution matrix directly into its long-term home
        res_matrix_path = SimulationManifest.derive_resolution_matrix_path(working_dir)
        if inputs.save_resolution_matrix:
            save_npz(res_matrix_path, resolution_mtx)

        # 7. Process buildings, slice the resolution matrix, and dump the individual worker pkls

        compiled_buildings_list: list[CompiledBuildingState] = []
        min_surface_index = 0

        for i, b_input in enumerate(inputs.buildings):
            # Create a dedicated subfolder for this building's simulation instance
            building_output_dir = CompiledBuildingState.derive_output_dir(
                runs_dir=runs_dir, building_index=i
            )
            building_output_dir.mkdir(exist_ok=False)

            # Slice matrix segments for this building core
            num_surfaces = len(b_input.outdoor_surface_names)
            max_surface_index = min_surface_index + num_surfaces - 1

            building_matrix_slice = resolution_mtx[min_surface_index : max_surface_index + 1, :]
            building_vf_srd_slice = total_srd_vf_list[min_surface_index : max_surface_index + 1]

            # Initialize the building instance and dump it to disk in one atomic operation
            building_instance_pkl_path = CompiledBuildingState.derive_instance_pkl_path(
                runs_dir=runs_dir, building_index=i
            )
            # TODO adjust the method bellow
            EpSimulationInstance.init_and_preprocess_to_pkl(
                building_input=b_input,
                simulation_index=i,
                min_surface_index=min_surface_index,
                max_surface_index=max_surface_index,
                resolution_mtx=building_matrix_slice,
                path_output_dir=building_output_dir,
                srd_vf_list=building_vf_srd_slice,
                pkl_file_path=building_instance_pkl_path,
            )

            compiled_buildings_list.append(
                CompiledBuildingState(
                    building_id=b_input.building_id,
                    building_index=i,
                    num_surfaces=num_surfaces,
                )
            )
            min_surface_index = max_surface_index + 1

        # 6. Build the lean runtime manifest.
        manifest = SimulationManifest(
            workspace_dir=working_dir,
            energyplus_dir=inputs.energyplus_dir,
            epw_file_name=epw_file_name,
            time_step=inputs.time_step,
            num_total_surfaces=num_total_surfaces,
            save_resolution_matrix=inputs.save_resolution_matrix,  # Only care about the output matrix
            compiled_buildings=compiled_buildings_list,
        )

        # Write out the clean, unpolluted manifest to JSON
        manifest.write_to_disk()

        return cls(manifest)

    @staticmethod
    def _make_target_workspace_dir(target_workspace_dir: Path, overwrite: bool = False) -> None:
        """Generate a clean, structurally sound workspace directory is ready for the simulation to run.

        Args:
            target_workspace_dir (Path): The absolute Path to the directory intended for the simulation workspace.
            overwrite (bool, optional): If True, allows overwriting existing data, if the data is safe to overwrite.
              Defaults to False.

        Raises:
            SecurityViolationError: If the target path threatens critical system or
                user root boundaries (via assert_path_is_safe_for_purging).
            WorkspaceConflictError: Blended exception cases:
                - If the directory contains data and overwrite is False.
                - If the folder is occupied but missing a valid 'simulation_manifest.json'.
                - If unexpected foreign assets breach the exclusivity perimeter.
                - If multiple climate profiles (.epw) are detected inside the root.
        """
        resolved_workspace = target_workspace_dir.resolve()
        if resolved_workspace.exists() and any(resolved_workspace.iterdir()):
            # Enforce system safety boundaries via our utility function
            assert_path_is_safe_for_purging(resolved_workspace)
            # Enforce data protection
            if not overwrite:
                raise WorkspaceConflictError(
                    target_path=resolved_workspace,
                    message=(
                        f"Target workspace directory '{resolved_workspace}' already exists and contains data. "
                        f"Set 'overwrite=True' to allow clearing this folder."
                    ),
                )

            SimulationManifest.verify_workspace_exclusivity(resolved_workspace)

            # =====================================================================
            # ATOMIC PURGE OF SANCTIONED ENVIRONMENT
            # =====================================================================
            # The folder passed strict exclusivity matching, meaning every single file inside
            # it is guaranteed to belong entirely to our simulation framework.
            # We can safely delete the tree natively without risk of data corruption.
            shutil.rmtree(resolved_workspace)

        # Ensure a clean directory root exists to begin compiling
        resolved_workspace.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------#
    # -----------------  Run Simulation     ---------------#
    # -----------------------------------------------------#

    @classmethod
    def run_from_workspace_dir(cls, workspace_dir: Path) -> int:
        """The clean pipeline entrypoint executing within the clean process boundary.

        Args:
            workspace_dir: Absolute path to the workspace directory containing the simulation manifest.
        Returns:
            The execution return code from the coupled simulation run.
        """
        manifest_path = SimulationManifest.derive_json_path(workspace_dir)
        adapted_manifest = SimulationManifest.load_and_adapt(manifest_path)
        manager = cls(adapted_manifest)

        # 1. Capture the 0 or 1 returned by the building simulation batch loop
        result_code = manager.run_lwr_coupled_simulation()

        # 2. If it's a 1 (failure), we explicitly tell the OS to exit with code 1
        if result_code != 0:
            logger.error("Simulation engine returned a non-zero status. Forcing exit.")
            sys.exit(result_code)  # This directly sets process.exitcode in the parent!

        return result_code

    @classmethod
    def run_in_new_process_from_workspace_dir(
        cls, workspace_dir: Path, debug_mode: bool = False
    ) -> int:
        """Executes the coupled long-wave radiation simulation in an isolated process.

        Spawns a clean, pristine Python runtime environment to execute the
        urban-scale physics solvers. This pattern acts as a memory firewall,
        preventing the parent process's geometric data footprint from duplicating
        across parallel building workers.

        Args:
            workspace_dir: Absolute path to the workspace directory containing the simulation manifest.
            debug_mode: If True, runs the simulation in the main process for debugging purposes.

        Returns:
            The execution return code from the isolated core engine process.

        Raises:
            FileNotFoundError: If the manifest JSON path does not exist on disk.
        """
        manifest_path = SimulationManifest.derive_json_path(workspace_dir)

        # Load data, automatically fixing any moved path references safely and ensure integrity
        adapted_manifest = SimulationManifest.load_and_adapt(manifest_path)

        if debug_mode:
            print("Debug mode enabled: Running simulation in the main process.")
            manager = cls(adapted_manifest)
            exit_code = manager.run_lwr_coupled_simulation()
        else:
            ctx = get_context("spawn")

            process = ctx.Process(target=cls.run_from_workspace_dir, args=(workspace_dir,))

            process.start()
            process.join()  # Wait for it to finish and free all memory cleanly
            exit_code = process.exitcode

        if exit_code is None:
            raise SimulationCrashError(
                "The simulation process vanished without returning an exit code."
            )

        if exit_code != 0:
            if exit_code < 0:
                signal_number = -exit_code
                error_msg = (
                    f"Simulation process killed by operating system signal: {signal_number}."
                )
                if signal_number == 9:
                    error_msg += (
                        " (Heuristic: This strongly indicates an Out-Of-Memory / OOM termination)."
                    )

                logger.critical(error_msg)
                raise SimulationCrashError(error_msg)

            error_msg = f"Simulation process encountered an unhandled exception and crashed with exit code: {exit_code}."
            logger.error(error_msg)
            raise SimulationCrashError(error_msg)

        logger.info("Simulation process finished successfully with exit code: %d", exit_code)
        return exit_code

    def run_lwr_coupled_simulation(self) -> int:
        """Runs the coupled long-wave radiation (LWR) simulation for all buildings.

        Spawns and monitors individual building solver processes. Implements a
        defensive polling execution loop to detect child process failures early,
        aborting synchronization barriers to prevent permanent simulation deadlocks.

        Raises:
            BrokenBarrierError: If a child worker process crashes and forces
                the synchronization grid to collapse.

        Returns:
            An integer status code: 0 for full success across all buildings,
            1 if one or more building simulations encountered a runtime crash.
        """

        with Manager() as manager:
            # Initialize Barrier, one process per building
            synch_point_barrier = manager.Barrier(self.num_buildings)

            # Shared memory vectors
            shm_temperatures = shared_memory.SharedMemory(
                create=True, size=self.num_total_surfaces * np.float64().itemsize
            )
            shm_timesteps = shared_memory.SharedMemory(
                create=True, size=self.num_buildings * np.float64().itemsize
            )

            processes: list[WorkerTracker] = []
            failed_processes: list[tuple[str, int, int]] = []

            try:
                # Note: Swapped to placeholder notation for logging best practices
                logger.info("Launching %d processes...", self.num_buildings)

                # --- 1. Create and Start Processes Manually ---
                for building_index, building_state in enumerate(self.buildings):
                    p = Process(
                        target=EpSimulationInstance.run_coupled_simulation_from_ep_instance,
                        kwargs={
                            "ep_instance_pkl_path": self._manifest.get_building_instance_pkl_path(
                                building_state
                            ),
                            "epw_path": self.epw_path,
                            "energyplus_dir": self.energyplus_dir,
                            "time_step": self.time_step,
                            "output_dir": self._manifest.get_building_output_dir(building_state),
                            "idf_path": self._manifest.get_building_idf_path(building_state),
                            "num_buildings": self.num_buildings,
                            "num_total_surfaces": self.num_total_surfaces,
                            "shared_memory_temperatures_name": shm_temperatures.name,
                            "shared_memory_timesteps_name": shm_timesteps.name,
                            "synch_point_barrier": synch_point_barrier,
                        },
                    )
                    processes.append(WorkerTracker(building_state.building_id, building_index, p))
                    p.start()

                # --- 2. Active Polling Supervision Loop ---
                logger.info("Entering active process supervision loop.")

                # Keep looping as long as there are active background processes running
                while any(worker.process.is_alive() for worker in processes):
                    for worker in processes:
                        # Check if a process has finished or died
                        if not worker.process.is_alive():
                            exit_code = worker.process.exitcode

                            # If a process exited with a code other than 0, it crashed!
                            if exit_code is not None and exit_code != 0:
                                logger.error(
                                    "CRITICAL: Building process [%d] with ID %s died with exit code %s!",
                                    worker.building_index,
                                    worker.building_id,
                                    exit_code,
                                )

                                # Add to tracking
                                if (
                                    worker.building_id,
                                    worker.building_index,
                                    exit_code,
                                ) not in failed_processes:
                                    failed_processes.append(
                                        (worker.building_id, worker.building_index, exit_code)
                                    )

                                # THE SELF-HEAL TRIGGER: Break the barrier immediately!
                                # This raises a BrokenBarrierError inside all other 119 buildings,
                                # causing them to stop waiting and gracefully terminate.
                                logger.warning(
                                    "Aborting synch_point_barrier to release surviving processes."
                                )
                                synch_point_barrier.abort()
                                break  # Drop out of the inner loop to accelerate shutdown

                    # Small sleep window prevents the main thread from maxing out a CPU core while polling
                    time.sleep(0.5)

                # --- 3. Final Orderly Cleanup Join ---
                # Now that the loop finished (either naturally or via abort), clean up the handles.
                # This is non-blocking now because they are all guaranteed to be dead or finishing.
                for worker in processes:
                    worker.process.join()

            finally:
                # Secure cleanup of shared memory resources to prevent OS memory leaks
                shm_temperatures.close()
                shm_temperatures.unlink()
                shm_timesteps.close()
                shm_timesteps.unlink()

            # --- 4. Evaluate Aggregated Results ---
            if failed_processes:
                logger.critical(
                    "Simulation batch failed. %d out of %d buildings encountered errors.",
                    len(failed_processes),
                    self.num_buildings,
                )
                return 1

        logger.info("All %d building simulations completed successfully.", self.num_buildings)
        return 0
