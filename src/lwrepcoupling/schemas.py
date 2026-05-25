from pathlib import Path
from typing import Self

from pydantic import (
    BaseModel,
    ConfigDict,
    DirectoryPath,
    Field,
    FilePath,
    field_validator,
    model_validator,
)

from .utils.utils_inverse_matrices import InversionConfig
from .utils.utils_io import WorkspaceConflictError

# =====================================================================
# THE PRE-PROCESSING MODELS
# =====================================================================


class BuildingInput(BaseModel):
    """Raw incoming building layout definition from the upstream application.

    Attributes:
        building_id: Unique alphanumeric string identifying the building.
        path_idf: Absolute filesystem path to the building's EnergyPlus IDF file.
        outdoor_surface_names: List of outdoor surfaces in the radiation ring.
    """

    building_id: str
    path_idf: FilePath
    outdoor_surface_names: list[str]

    @field_validator("path_idf", mode="after")
    @classmethod
    def enforce_idf_extension(cls, v: FilePath) -> FilePath:
        """Enforces that the incoming layout file uses a valid .idf extension."""
        # .suffix returns the extension with the leading dot included (e.g., '.idf')
        if v.suffix.lower() != ".idf":
            raise ValueError(
                f"Invalid file type for building layout. Expected a '.idf' file, "
                f"but received '{v.name}' instead."
            )
        return v


class SimulationInputs(BaseModel):
    """The temporary memory container used to compile the workspace.

    Attributes:
        workspace_dir: Target root directory path for the simulation workspace.
        energyplus_dir: Path to the local EnergyPlus installation directory.
        epw_path: Path to the raw source weather (EPW) file.
        time_step: Core simulation time step fraction (e.g., 0.25).
        vf_matrix_path: Path to the computed View Factor sparse matrix file.
        eps_matrix_path: Path to the computed Emissivity surface vector file.
        rho_matrix_path: Path to the computed Reflectivity surface matrix file.
        tau_matrix_path: Path to the computed Transmissivity surface matrix file.
        inversion_parameters: Custom tracking options for the matrix solvers.
        buildings: Collection of raw input layout definitions.
    """

    workspace_dir: DirectoryPath
    energyplus_dir: DirectoryPath
    epw_path: FilePath
    time_step: int

    vf_matrix_path: FilePath
    eps_matrix_path: FilePath
    rho_matrix_path: FilePath
    tau_matrix_path: FilePath

    inversion_parameters: InversionConfig = Field(default_factory=InversionConfig)
    buildings: list[BuildingInput]

    save_resolution_matrix: bool = False

    # Guarantees the weather data format is pristine
    @field_validator("epw_path", mode="after")
    @classmethod
    def enforce_epw_extension(cls, v: FilePath) -> FilePath:
        """Enforces that the incoming climate asset uses a valid .epw extension."""
        if v.suffix.lower() != ".epw":
            raise ValueError(
                f"Invalid file type for weather data. Expected a '.epw' file, "
                f"but received '{v.name}' instead."
            )
        return v

    # 3. Batch-validate all data matrix assets in one clean block
    @field_validator(
        "vf_matrix_path", "eps_matrix_path", "rho_matrix_path", "tau_matrix_path", mode="after"
    )
    @classmethod
    def enforce_matrix_extensions(cls, v: FilePath) -> FilePath:
        """Enforces that all structural sparse matrices use verified computational formats."""
        # Example: Allowing both numpy binaries (.npy) or compressed archives (.npz)
        allowed_extensions = {".npy", ".npz"}

        if v.suffix.lower() not in allowed_extensions:
            raise ValueError(
                f"Invalid matrix file format. Expected a numpy storage file ({allowed_extensions}), "
                f"but received '{v.name}' instead."
            )
        return v

    @property
    def num_total_surfaces(self) -> int:
        """Calculates the aggregate outdoor surface count across all inputs."""
        return sum(len(b.outdoor_surface_names) for b in self.buildings)


# =====================================================================
# THE RUNTIME MODELS (Saved to disk, drives parallel execution)
# =====================================================================


class CompiledBuildingState(BaseModel):
    """Execution mapping metadata for an isolated building worker.

    Attributes:
        building_id: Unique alphanumeric string identifying the building.
        building_index: Zero-indexed matrix positioning tracker for execution loops.
    """

    building_id: str
    building_index: int

    @classmethod
    def short_id(cls, building_index: int) -> str:
        """Generates a standardized short string identifier (e.g., 'b_0')."""
        return f"b_{building_index}"

    # =====================================================================
    # EXTRACTED CORE ARCHITECTURE FORMULAS (STATIC / CLASS)
    # =====================================================================
    @classmethod
    def derive_output_dir(cls, runs_dir: Path, building_index: int) -> Path:
        """Pure formula determining a worker's target run folder destination."""
        return runs_dir / cls.short_id(building_index)

    @classmethod
    def derive_instance_pkl_path(cls, runs_dir: Path, building_index: int) -> Path:
        """Pure formula determining a building's serialized worker binary destination."""
        return (
            cls.derive_output_dir(runs_dir, building_index) / f"{cls.short_id(building_index)}.pkl"
        )

    # =====================================================================
    # LIVE PROPERTY INSTANCE WRAPPERS
    # =====================================================================
    def get_output_dir(self, runs_dir: Path) -> Path:
        """Instance method wrapper calling the underlying classmethod formula."""
        return self.derive_output_dir(runs_dir, self.building_index)

    def get_instance_pkl_path(self, runs_dir: Path) -> Path:
        """Instance method wrapper calling the underlying classmethod formula."""
        return self.derive_instance_pkl_path(runs_dir, self.building_index)


class SimulationManifest(BaseModel):
    """The streamlined structural blueprint used to drive the execution engine.

    Attributes:
        workspace_dir: The active absolute path hosting the workspace root.
        energyplus_dir: System path to the targeted EnergyPlus installation.
        epw_file_name: Local filename string of the workspace weather asset.
        time_step: EnergyPlus simulation number of time steps per minute.
        num_outdoor_surfaces: Overall global surface boundary array dimension.
        compiled_buildings: Catalog mapping sequence indexes to worker entities.
    """

    model_config = ConfigDict(frozen=True)

    RUNS_DIR_NAME: str = "runs"
    MANIFEST_FILE_NAME: str = "simulation_manifest.json"
    RESOLUTION_MTX_FILE_NAME: str = "resolution_matrix.npz"

    workspace_dir: Path
    energyplus_dir: Path
    epw_file_name: str
    time_step: int
    num_total_surfaces: int

    compiled_buildings: list[CompiledBuildingState]

    save_resolution_matrix: bool

    @classmethod
    def derive_runs_dir(cls, workspace_dir: Path) -> Path:
        """Pure formula to determine the runs folder location from a workspace root."""
        return workspace_dir / cls.RUNS_DIR_NAME

    @classmethod
    def derive_epw_path(cls, workspace_dir: Path, epw_file_name: str) -> Path:
        """Pure formula to determine the EPW file location from a workspace root."""
        return workspace_dir / epw_file_name

    @classmethod
    def derive_resolution_matrix_path(cls, workspace_dir: Path) -> Path:
        """Pure formula to determine the resolution matrix file location from a workspace root."""
        return workspace_dir / cls.RESOLUTION_MTX_FILE_NAME

    @classmethod
    def derive_building_instance_pkl_path(cls, workspace_dir: Path, building_index: int) -> Path:
        """Pure formula to determine a building worker's instance PKL file location."""
        return CompiledBuildingState.derive_instance_pkl_path(
            runs_dir=cls.derive_runs_dir(workspace_dir), building_index=building_index
        )

    @classmethod
    def derive_json_path(cls, workspace_dir: Path) -> Path:
        """Pure formula to determine the manifest JSON file location from a workspace root."""
        return workspace_dir / cls.MANIFEST_FILE_NAME

    # =====================================================================
    # LIVE DYNAMIC PATH PROPERTIES (Instance Wrappers)
    # =====================================================================
    @property
    def num_buildings(self) -> int:
        """Calculates total quantity of tracked compiled building worker states."""
        return len(self.compiled_buildings)

    @property
    def runs_dir(self) -> Path:
        """Resolves the absolute path to the active runs subfolder pool."""
        return self.derive_runs_dir(self.workspace_dir)

    @property
    def epw_path(self) -> Path:
        """Computes the absolute path to the local weather file asset."""
        return self.derive_epw_path(self.workspace_dir, self.epw_file_name)

    @property
    def resolution_matrix_path(self) -> Path:
        """Computes the absolute path to the compiled solver matrix file."""
        return self.derive_resolution_matrix_path(self.workspace_dir)

    def get_building_instance_pkl_path(self, building_state: CompiledBuildingState) -> Path:
        """Resolves a target building binary file location on disk via delegation."""
        return self.derive_building_instance_pkl_path(
            workspace_dir=self.workspace_dir, building_index=building_state.building_index
        )

    @property
    def json_path(self) -> Path:
        """Resolves the absolute path to the manifest JSON file."""
        return self.derive_json_path(self.workspace_dir)

    # =====================================================================
    # INTEGRITY VALIDATION GATES & LOADERS
    # =====================================================================
    @model_validator(mode="after")
    def verify_list_sequence_integrity(self) -> Self:
        """Guarantees the structural index properties perfectly align with list ordering.

        Raises:
            ValueError: If an array tracking offset or sequence gap is detected.
        """
        for expected_idx, b_state in enumerate(self.compiled_buildings):
            if b_state.building_index != expected_idx:
                raise ValueError(
                    f"Critical manifest alignment error! Building '{b_state.building_id}' "
                    f"is at list position {expected_idx} but holds index {b_state.building_index}."
                )
        return self

    @model_validator(mode="after")
    def verify_num_total_surfaces(self) -> Self:
        """Guarantees that the total number of outdoor surfaces tracked across all buildings perfectly
        matches the manifest declaration, to avoid connflicts between the resolution matrix size and
        the actual surface count.
        Raises:
            ValueError: If a mismatch is detected between the total number of outdoor surfaces
            tracked across all buildings and the manifest declaration.
        """
        num_total_surfaces = sum(len(b_state.building_id) for b_state in self.compiled_buildings)
        if num_total_surfaces != self.num_total_surfaces:
            raise ValueError(
                f"Critical manifest alignment error! The total number of outdoor surfaces "
                f"tracked across all buildings is {num_total_surfaces}, but the manifest "
                f"declares a different total of {self.num_total_surfaces}."
            )
        return self

    def verify_workspace_integrity(self) -> None:
        """Strictly verifies that all core engines, layout folders, and assets exist on disk.

        Raises:
            FileNotFoundError: If any critical compilation dependency is missing.
        """
        if not self.energyplus_dir.is_dir():
            raise FileNotFoundError(
                f"Missing core EnergyPlus engine directory at: {self.energyplus_dir}"
            )

        if not self.epw_path.is_file():
            raise FileNotFoundError(f"Missing simulation weather file (EPW) at: {self.epw_path}")

        if self.save_resolution_matrix and not self.resolution_matrix_path.is_file():
            raise FileNotFoundError(
                f"Missing compiled master resolution matrix at: {self.resolution_matrix_path}"
            )

        for b_state in self.compiled_buildings:
            expected_pkl = self.get_building_instance_pkl_path(b_state)
            if not expected_pkl.is_file():
                raise FileNotFoundError(
                    f"Corrupted workspace detected! Missing critical execution binary "
                    f"for building '{b_state.building_id}' at: {expected_pkl}"
                )

    @classmethod
    def verify_workspace_exclusivity(cls, target_workspace_dir: Path) -> None:
        """Strictly verifies that the workspace contains *only* authorized assets.

        This method acts as a double-sided fingerprint guardrail. It checks that
        every item resting at the first level of the physical workspace folder
        is explicitly tracked by this manifest, preventing foreign data corruption.

        Raises:
            WorkspaceConflictError: If unauthorized files or folders are discovered.
            WorkspaceConflictError: If multiple EPW files are found, breaching the "one weather file" rule.
        """

        resolved_workspace = target_workspace_dir.resolve()

        authorized_paths: set[Path] = {
            resolved_workspace / cls.MANIFEST_FILE_NAME,
            (resolved_workspace / cls.RUNS_DIR_NAME).resolve(),
            (resolved_workspace / cls.RESOLUTION_MTX_FILE_NAME).resolve(),
        }

        # Tracker to enforce the "maximum of one weather file" rule
        epw_file_count = 0

        # Scan the actual physical directory (outward pass)
        for physical_item in resolved_workspace.iterdir():
            resolved_item = physical_item.resolve()

            # Pass A: If it's a hardcoded authorized structural asset, skip to next item
            if resolved_item in authorized_paths:
                continue

            # Pass B: Pragmatic Best Practice check using path.suffix
            # Note: .suffix returns lowercase string with the dot included (e.g., '.epw')
            if resolved_item.is_file() and resolved_item.suffix.lower() == ".epw":
                epw_file_count += 1

                if epw_file_count <= 1:
                    continue  # Safely allow the first epw file encountered

                # If we get here, it means epw_file_count reached 2
                raise WorkspaceConflictError(
                    target_path=resolved_workspace,
                    message=(
                        f"Security Block: Multiple climate profiles discovered! Found more than one "
                        f"'.epw' file inside '{resolved_workspace.name}'. Workspace must contain "
                        f"at most one weather data file to ensure deterministic runtime execution."
                    ),
                )

            # Pass C: If it's not an authorized asset AND not the first valid .epw file, breach!
            raise WorkspaceConflictError(
                target_path=resolved_workspace,
                message=(
                    f"Security Block: Workspace exclusivity breach! Found unexpected "
                    f"foreign asset '{resolved_item.name}' inside the workspace. "
                    f"Clearing or modifying this folder has been blocked to protect external data."
                ),
            )

    # =====================================================================
    #  IO METHODS
    # =====================================================================
    def write_to_disk(self) -> None:
        """Serializes the data contract and writes it out as an organized JSON file."""
        target_json_path = self.json_path
        json_payload = self.model_dump_json(indent=4)
        target_json_path.write_text(json_payload, encoding="utf-8")

    @classmethod
    def load_and_adapt(cls, manifest_json_path: Path) -> "SimulationManifest":
        """Factory gateway that reads the raw JSON from disk and self-heals root paths.

        Args:
            manifest_json_path: Path to the JSON file or parent workspace directory root.

        Returns:
            A fully initialized, self-healed, path-verified manifest model instance.

        Raises:
            FileNotFoundError: If physical assets break workspace integrity checks.
        """
        if manifest_json_path.is_dir():
            manifest_json_path = cls.derive_json_path(manifest_json_path)

        if not manifest_json_path.is_file():
            raise FileNotFoundError(f"Manifest JSON file not found at: {manifest_json_path}")

        manifest_obj = cls.model_validate_json(manifest_json_path.read_text())

        actual_workspace_dir = manifest_json_path.parent.resolve()
        stored_workspace_dir = manifest_obj.workspace_dir.resolve()

        if actual_workspace_dir != stored_workspace_dir:
            manifest_obj = manifest_obj.model_copy(update={"workspace_dir": actual_workspace_dir})

        manifest_obj.verify_workspace_integrity()
        return manifest_obj
