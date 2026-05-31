"""
Class to manage the couped long-wave radiation (LWR) simulation with EnergyPlus among
multiple buildings.
"""

import logging
import shutil
import sys
import time
from multiprocessing import Manager, get_context, shared_memory
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import NamedTuple, Self

import numpy as np
from scipy.sparse import csr_matrix, save_npz

from ._ep_simulation_instance import EpSimulationRuntimeWorker
from ._schemas import (
    BuildingInput,
    CompiledBuildingState,
    EpSimulationRuntimeConfig,
    SimulationInputs,
    SimulationManifest,
)
from ._utils import (
    check_matrices,
    compute_resolution_matrices,
    generate_idfs_additional_strings,
    read_csr_matrices_from_npz,
)
from ._utils.utils_io import assert_path_is_safe_for_purging
from .exceptions import SimulationCrashError, WorkspaceConflictError

logger = logging.getLogger(__name__)


class WorkerTracker(NamedTuple):
    """Local tracking container to monitor active simulation child processes."""

    building_id: str
    building_index: int
    process: BaseProcess


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
    def epw_path(self) -> Path:
        """The path to the weather file used in the simulation."""
        return self._manifest.epw_path

    @property
    def num_ts_per_h(self) -> int:
        """The time step used in the simulation."""
        return self._manifest.num_ts_per_h

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
    ) -> Self:
        """Consumes raw simulation inputs and structurally generates the disk workspace.

        Acts as the central orchestration workflow, validating environment safety,
        staging global dependencies, executing structural matrix calculations, and
        compiling individual subfolders for parallel worker nodes.

        Args:
            inputs: Structured configuration container holding data shape boundaries
                and raw file system path anchors.
            overwrite: If True, explicitly allows clearing out an occupied,
                non-empty workspace folder if it passes fingerprint validation.

        Returns:
            An initialized and ready-to-run instance of the EpLwrSimulationManager engine.

        Raises:
            SecurityViolationError: If the target workspace path threatens OS or root boundaries.
            WorkspaceConflictError: If the folder is occupied and overwrite is disabled,
                or if the folder fails structural fingerprint verification.
            ValueError: If the input view factor configurations fail internal matrix sizing,
                or if sequential building sequence indexes are misaligned.
            FileNotFoundError: If a generated asset or critical software dependency fails
                the post-compilation integrity check.
            OSError: If copying source assets or writing data payloads to disk fails
                due to OS permissions or missing directories.
        """
        # 1. Environment Guardrails & Directory Creation
        cls._make_target_workspace_dir(
            target_workspace_dir=inputs.workspace_dir, overwrite=overwrite
        )
        working_dir = inputs.workspace_dir.resolve()
        runs_dir = SimulationManifest.derive_runs_dir(working_dir)
        runs_dir.mkdir(exist_ok=False)

        # 2. Stage Climate Asset
        shutil.copy(
            inputs.epw_path, SimulationManifest.derive_epw_path(working_dir, inputs.epw_path.name)
        )

        # 3. Load & Process Numerical Infrastructure
        resolution_mtx, total_srd_vf_list = cls._load_and_validate_matrices(inputs)
        if inputs.save_resolution_matrix:
            save_npz(SimulationManifest.derive_resolution_matrix_path(working_dir), resolution_mtx)

        # 4. Compile Individual Building Sandbox Environments
        compiled_buildings_list: list[CompiledBuildingState] = []
        min_surface_index = 0

        for i, b_input in enumerate(inputs.buildings):
            building_state = cls._process_single_building_workspace(
                building_index=i,
                building_input=b_input,
                runs_dir=runs_dir,
                min_surface_index=min_surface_index,
                resolution_mtx=resolution_mtx,
                total_srd_vf_list=total_srd_vf_list,
            )
            compiled_buildings_list.append(building_state)
            min_surface_index = building_state.surface_index_max + 1

        # 5. Lock and Commit the Manifest Data Contract
        manifest = SimulationManifest(
            workspace_dir=working_dir,
            epw_file_name=inputs.epw_path.name,
            num_ts_per_h=inputs.num_ts_per_h,
            num_total_surfaces=inputs.num_total_surfaces,
            save_resolution_matrix=inputs.save_resolution_matrix,
            compiled_buildings=compiled_buildings_list,
        )
        cls._verify_workspace_integrity(manifest)
        manifest.write_to_disk()

        return cls(manifest)

    @staticmethod
    def _load_and_validate_matrices(inputs: SimulationInputs) -> tuple[csr_matrix, list[float]]:
        """Handles matrix extraction, structural size matching, and physical system inversion.

        Parses incoming CSR matrices from raw disk archives, cross-checks total surface boundaries,
        and computes the long-wave radiation matrix inversion problem.

        Args:
            inputs: Structured configuration container holding the incoming matrix paths
                and custom inversion parameters.

        Returns:
            A tuple containing:
                - resolution_mtx (np.ndarray): The fully solved dense system resolution matrix.
                - total_srd_vf_list (list[float]): Vector of calculated total surrounding surfaces
                    view factors.

        Raises:
            ValueError: If the shapes of the read sparse matrix inputs do not perfectly match
                the aggregate surface numbers declared in the simulation input.
        """
        vf_mtx, eps_mtx, rho_mtx, tau_mtx = read_csr_matrices_from_npz(
            inputs.vf_matrix_path,
            inputs.eps_matrix_path,
            inputs.rho_matrix_path,
            inputs.tau_matrix_path,
        )
        check_matrices(vf_mtx, eps_mtx, rho_mtx, tau_mtx)

        if vf_mtx.shape[0] != inputs.num_total_surfaces:
            raise ValueError("The matrix dimensions must match the number of outdoor surfaces.")

        return compute_resolution_matrices(
            vf_matrix=vf_mtx,
            eps_matrix=eps_mtx,
            rho_matrix=rho_mtx,
            tau_matrix=tau_mtx,
            inversion_config=inputs.inversion_parameters,
        )

    @staticmethod
    def _process_single_building_workspace(
        building_index: int,
        building_input: BuildingInput,
        runs_dir: Path,
        min_surface_index: int,
        resolution_mtx: csr_matrix,
        total_srd_vf_list: list[float],
    ) -> CompiledBuildingState:
        """Isolates the filesystem modifications and mathematical slicing for a unique building.

        Creates individual run folders on disk, slices out the specific building segment from the
        global resolution matrix, saves the sub-matrix as a native NumPy `.npy` binary, and triggers
        the IDF macro string injection.

        Args:
            building_index: Zero-based sequence positioning tracker for the targeted building.
            building_input: Raw incoming layout data contract containing surface collections
                and source paths.
            runs_dir: Path to the active workspace subfolder hosting all execution directories.
            min_surface_index: Offset marker defining where this building's surfaces start in
                the global array.
            resolution_mtx: The fully solved dense system matrix representing all combined
                buildings.
            total_srd_vf_list: Global array containing computed surrounding view factors for
                all surfaces.

        Returns:
            CompiledBuildingState: A validated Pydantic runtime data asset tracking the newly
                compiled building sandbox.

        Raises:
            OSError: If copying the source geometry file, appending strings, or writing the
                sliced NumPy arrays fails due to storage write exceptions.
        """
        building_output_dir = CompiledBuildingState.derive_output_dir(runs_dir, building_index)
        building_output_dir.mkdir(exist_ok=False)

        num_surfaces = len(building_input.outdoor_surface_names)
        max_surface_index = min_surface_index + num_surfaces - 1

        # Slice and commit the optimized dense array segment to disk
        building_matrix_slice = resolution_mtx[min_surface_index : max_surface_index + 1, :]
        building_res_matrix_path = CompiledBuildingState.derive_sub_mtx_path(
            runs_dir, building_index
        )
        np.save(building_res_matrix_path, building_matrix_slice)

        # Inject additional physics strings to target geometry description layout
        building_vf_srd_slice = total_srd_vf_list[min_surface_index : max_surface_index + 1]
        additional_strings = generate_idfs_additional_strings(
            outdoor_surface_names=building_input.outdoor_surface_names,
            srd_vf_list=building_vf_srd_slice,
            sky_vf_list=None,
            ground_vf_list=None,
        )

        adjusted_idf_path = CompiledBuildingState.derive_idf_path(runs_dir, building_index)
        shutil.copy(building_input.idf_path.resolve(), adjusted_idf_path)
        with open(adjusted_idf_path, "a", encoding="utf-8") as file:
            file.write(additional_strings)

        return CompiledBuildingState(
            building_id=building_input.building_id,
            building_index=building_index,
            num_surfaces=num_surfaces,
            outdoor_surface_names=building_input.outdoor_surface_names,
            surface_index_min=min_surface_index,
            surface_index_max=max_surface_index,
        )

    @classmethod
    def _make_target_workspace_dir(
        cls, target_workspace_dir: Path, overwrite: bool = False
    ) -> None:
        """
        Generate a clean, structurally sound workspace directory is ready for the simulation to run.

        Args:
            target_workspace_dir (Path): The absolute Path to the directory intended for
                the simulation workspace.
            overwrite (bool, optional): If True, allows overwriting existing data, if the data is
                safe to overwrite.
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
                        f"Target workspace directory '{resolved_workspace}' already exists and "
                        f"contains data."
                        f"Set 'overwrite=True' to allow clearing this folder."
                    ),
                )

            cls._verify_workspace_exclusivity(resolved_workspace)

            # =====================================================================
            # ATOMIC PURGE OF SANCTIONED ENVIRONMENT
            # =====================================================================
            # The folder passed strict exclusivity matching, meaning every single file inside
            # it is guaranteed to belong entirely to our simulation framework.
            # We can safely delete the tree natively without risk of data corruption.
            shutil.rmtree(resolved_workspace)

        # Ensure a clean directory root exists to begin compiling
        resolved_workspace.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _verify_workspace_integrity(manifest: SimulationManifest) -> None:
        """Strictly verifies that all core engines, layout folders, and assets exist on disk.

        Raises:
            FileNotFoundError: If any critical compilation dependency is missing.
        """
        if not manifest.epw_path.is_file():
            raise FileNotFoundError(
                f"Missing simulation weather file (EPW) at: {manifest.epw_path}"
            )

        # If you saved a master resolution matrix file during workspace compilation
        res_matrix_path = SimulationManifest.derive_resolution_matrix_path(manifest.workspace_dir)
        if manifest.save_resolution_matrix and not res_matrix_path.is_file():
            raise FileNotFoundError(
                f"Missing compiled master resolution matrix at: {res_matrix_path}"
            )

        # Iterate through compiled states, verifying file structures
        runs_dir = SimulationManifest.derive_runs_dir(manifest.workspace_dir)
        for b_state in manifest.compiled_buildings:
            # We check the native .npy file rather than the dead binary .pkl!
            expected_npy = b_state.get_sub_mtx_path(runs_dir)
            if not expected_npy.is_file():
                raise FileNotFoundError(
                    f"Corrupted workspace detected! Missing critical matrix binary "
                    f"for building '{b_state.building_id}' at: {expected_npy}"
                )

            expected_idf = b_state.get_idf_path(runs_dir)
            if not expected_idf.is_file():
                raise FileNotFoundError(
                    f"Corrupted workspace detected! Missing critical IDF file "
                    f"for building '{b_state.building_id}' at: {expected_idf}"
                )

    @staticmethod
    def _verify_workspace_exclusivity(target_workspace_dir: Path) -> None:
        """Strictly verifies that the workspace contains *only* authorized assets.

        Prevents foreign or unknown data corruption by serving as a fingerprint guardrail.
        """
        resolved_workspace = target_workspace_dir.resolve()

        # Define our rigid, sanctioned execution perimeter names
        authorized_paths: set[Path] = {
            resolved_workspace / SimulationManifest.MANIFEST_FILE_NAME,
            (resolved_workspace / SimulationManifest.RUNS_DIR_NAME).resolve(),
            (resolved_workspace / SimulationManifest.RESOLUTION_MTX_FILE_NAME).resolve(),
        }

        epw_file_count = 0

        for physical_item in resolved_workspace.iterdir():
            resolved_item = physical_item.resolve()

            if resolved_item in authorized_paths:
                continue

            if resolved_item.is_file() and resolved_item.suffix.lower() == ".epw":
                epw_file_count += 1
                if epw_file_count <= 1:
                    continue  # Authorized standard environment profile

                raise WorkspaceConflictError(
                    target_path=resolved_workspace,
                    message=(
                        f"Security Block: Multiple climate profiles discovered inside "
                        f"'{resolved_workspace.name}'. "
                        f"Workspace must contain at most one weather data file to guarantee "
                        f"deterministic execution."
                    ),
                )

            raise WorkspaceConflictError(
                target_path=resolved_workspace,
                message=(
                    f"Security Block: Workspace exclusivity breach! Found unexpected foreign asset "
                    f"'{resolved_item.name}' inside the workspace perimeter. Operations blocked to "
                    f"preserve data."
                ),
            )

    # -----------------------------------------------------#
    # -----------------  Run Simulation     ---------------#
    # -----------------------------------------------------#

    @classmethod
    def _run_from_workspace_dir(cls, workspace_dir: Path) -> int:
        """The clean pipeline entrypoint executing within the clean process boundary.

        Args:
            workspace_dir: Absolute path to the workspace directory containing the
                simulation manifest.
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
            workspace_dir: Absolute path to the workspace directory containing
                the simulation manifest.
            debug_mode: If True, runs the simulation in the main process for debugging purposes.

        Returns:
            The execution return code from the isolated core engine process.

        Raises:
            FileNotFoundError: If the manifest JSON path does not exist on disk.
        """
        manifest_path = SimulationManifest.derive_json_path(workspace_dir)

        # Load data, automatically fixing any moved path references safely and ensure integrity
        adapted_manifest = SimulationManifest.load_and_adapt(manifest_path)
        cls._verify_workspace_integrity(manifest=adapted_manifest)

        if debug_mode:
            print("Debug mode enabled: Running simulation in the main process.")
            manager = cls(adapted_manifest)
            exit_code = manager.run_lwr_coupled_simulation()
        else:
            ctx = get_context("spawn")

            process = ctx.Process(target=cls._run_from_workspace_dir, args=(workspace_dir,))

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

            error_msg = "Simulation process encountered an unhandled exception and crashed with "
            f"exit code: {exit_code}."
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
        # Using an explicit 'spawn' context here instead of the default Linux 'fork'
        # to prevent deadlocks when duplicating the underlying  pyenergyplus C++ runtime bindings.
        ctx = get_context("spawn")

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
                    runtime_config = EpSimulationRuntimeConfig(
                        building_state=building_state,
                        epw_path=self.epw_path,
                        num_ts_per_h=self.num_ts_per_h,
                        runs_dir=self._manifest.runs_dir,
                        num_buildings=self.num_buildings,
                        num_total_surfaces=self.num_total_surfaces,
                        shared_memory_temperatures_name=shm_temperatures.name,
                        shared_memory_timesteps_name=shm_timesteps.name,
                        synch_point_barrier=synch_point_barrier,
                    )
                    p = ctx.Process(
                        target=EpSimulationRuntimeWorker.run_coupled_simulation,
                        args=(runtime_config,),
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
                                    "CRITICAL: Building process [%d] with ID %s died with"
                                    "exit code %s!",
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

                    time.sleep(0.01)

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
