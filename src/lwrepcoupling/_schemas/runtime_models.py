from pathlib import Path
from typing import Protocol, Self, runtime_checkable

from pydantic import (
    BaseModel,
    ConfigDict,
    model_validator,
)


@runtime_checkable
class SynchronizerBarrier(Protocol):
    """Structural type interface representing any process or thread barrier."""

    def wait(self, timeout: float | None = None) -> int: ...
    def abort(self) -> None: ...


class CompiledBuildingState(BaseModel):
    """Execution mapping metadata for an isolated building worker.

    Attributes:
        building_id: Unique alphanumeric string identifying the building.
        building_index: Zero-indexed matrix positioning tracker for execution loops.
    """

    building_id: str
    building_index: int
    num_surfaces: int
    surface_index_min: int
    surface_index_max: int
    outdoor_surface_names: list[str]

    @classmethod
    def short_id(cls, building_index: int) -> str:
        """Generates a standardized short string identifier (e.g., 'b_0')."""
        return f"b_{building_index}"

    @classmethod
    def mtx_file_name(cls, building_index: int) -> str:
        """Generates a standardized short string identifier (e.g., 'b_0')."""
        return f"{cls.short_id(building_index)}_mtx.npy"

    # =====================================================================
    # EXTRACTED CORE ARCHITECTURE FORMULAS (STATIC / CLASS)
    # =====================================================================
    @classmethod
    def derive_output_dir(cls, runs_dir: Path, building_index: int) -> Path:
        """Pure formula determining a worker's target run folder destination."""
        return runs_dir / cls.short_id(building_index)

    @classmethod
    def derive_idf_path(cls, runs_dir: Path, building_index: int) -> Path:
        """Pure formula determining a building's serialized worker binary destination."""
        return (
            cls.derive_output_dir(runs_dir, building_index) / f"{cls.short_id(building_index)}.idf"
        )

    @classmethod
    def derive_sub_mtx_path(cls, runs_dir: Path, building_index: int) -> Path:
        """Pure formula determining a building's serialized worker binary destination."""
        return cls.derive_output_dir(runs_dir, building_index) / cls.mtx_file_name(building_index)

    # =====================================================================
    # LIVE PROPERTY INSTANCE WRAPPERS
    # =====================================================================
    def get_output_dir(self, runs_dir: Path) -> Path:
        """Instance method wrapper calling the underlying classmethod formula."""
        return self.derive_output_dir(runs_dir, self.building_index)

    def get_idf_path(self, runs_dir: Path) -> Path:
        """Instance method wrapper calling the underlying classmethod formula."""
        return self.derive_idf_path(runs_dir, self.building_index)

    def get_sub_mtx_path(self, runs_dir: Path) -> Path:
        """Instance method wrapper calling the underlying classmethod formula."""
        return self.derive_sub_mtx_path(runs_dir, self.building_index)


class SimulationManifest(BaseModel):
    """The streamlined structural blueprint used to drive the execution engine.

    Attributes:
        workspace_dir: The active absolute path hosting the workspace root.
        energyplus_dir: System path to the targeted EnergyPlus installation.
        epw_file_name: Local filename string of the workspace weather asset.
        num_ts_per_h: EnergyPlus simulation number of time steps per minute.
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
    num_ts_per_h: int
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
    def derive_building_idf_path(cls, workspace_dir: Path, building_index: int) -> Path:
        """Pure formula to determine a building worker's IDF file location."""
        return CompiledBuildingState.derive_idf_path(
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

    @property
    def json_path(self) -> Path:
        """Resolves the absolute path to the manifest JSON file."""
        return self.derive_json_path(self.workspace_dir)

    # =====================================================================
    # Getters for building-specific paths that delegate to the underlying classmethod formulas
    # =====================================================================
    def get_building_idf_path(self, building_state: CompiledBuildingState) -> Path:
        """Resolves a target building IDF file location on disk via delegation."""
        return self.derive_building_idf_path(
            workspace_dir=self.workspace_dir,
            building_index=building_state.building_index,
        )

    def get_building_output_dir(self, building_state: CompiledBuildingState) -> Path:
        """Resolves a target building output directory location on disk via delegation."""
        return building_state.get_output_dir(runs_dir=self.runs_dir)

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
        matches the manifest declaration, to avoid conflicts between the resolution matrix size and
        the actual surface count.
        Raises:
            ValueError: If a mismatch is detected between the total number of outdoor surfaces
            tracked across all buildings and the manifest declaration.
        """
        num_total_surfaces = sum(b_state.num_surfaces for b_state in self.compiled_buildings)
        if num_total_surfaces != self.num_total_surfaces:
            raise ValueError(
                f"Critical manifest alignment error! The total number of outdoor surfaces "
                f"tracked across all buildings is {num_total_surfaces}, but the manifest "
                f"declares a different total of {self.num_total_surfaces}."
            )
        return self

    # =====================================================================
    #  IO METHODS
    # =====================================================================
    def write_to_disk(self) -> None:
        """Serializes the data contract and writes it out as an organized JSON file."""
        target_json_path = self.json_path
        json_payload = self.model_dump_json(indent=4)
        target_json_path.write_text(json_payload, encoding="utf-8")

    @classmethod
    def load_and_adapt(cls, manifest_json_path: Path) -> Self:
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

        return manifest_obj


class EpSimulationRuntimeConfig(BaseModel):
    """The strongly-typed configuration payload passed to spawned child processes."""

    # Allow arbitrary types so the multiprocessing Barrier passes validation gates
    model_config = ConfigDict(frozen=True)

    building_state: CompiledBuildingState
    epw_path: Path
    num_ts_per_h: int
    runs_dir: Path
    num_buildings: int
    num_total_surfaces: int
    shared_memory_temperatures_name: str
    shared_memory_timesteps_name: str
    synch_point_barrier: SynchronizerBarrier
