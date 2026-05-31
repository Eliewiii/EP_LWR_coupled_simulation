from typing import Annotated

from pydantic import (
    BaseModel,
    DirectoryPath,
    Field,
    FilePath,
    field_validator,
)


class InversionConfig(BaseModel):
    # Boundaries/Constraints
    tol: Annotated[float, Field(ge=1e-8, le=1e-4)] = 1e-5
    maxiter: Annotated[int, Field(ge=2, le=1000)] = 150
    rtol: Annotated[float, Field(ge=1e-10, le=1e-5)] = 5e-7
    precondition: bool = (
        False  # Create preconditioner if needed (Jacobi Preconditioner: inverse of diagonal)
    )
    strict_tol: bool = False  # If True, will raise an error if the final error norm is above tol
    num_workers: Annotated[int, Field(ge=0, le=60)] = (
        1  # Parallel workers (0 means use all available CPUs)
    )

    # Custom validation
    @field_validator("num_workers")
    @classmethod
    def check_workers(cls, v: int) -> int:
        # Example: logic that Field can't do alone
        if v == 0:
            import os

            return os.cpu_count() or 1
        return v


class BuildingInput(BaseModel):
    """Raw incoming building layout definition from the upstream application.

    Attributes:
        building_id: Unique alphanumeric string identifying the building.
        idf_path: Absolute filesystem path to the building's EnergyPlus IDF file.
        outdoor_surface_names: List of outdoor surfaces in the radiation ring.
    """

    building_id: str
    idf_path: FilePath
    outdoor_surface_names: list[str]

    @field_validator("idf_path", mode="after")
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
        epw_path: Path to the raw source weather (EPW) file.
        num_ts_per_h: Number of simulation time steps per hour, as defined in the EnergyPlus IDF.
        vf_matrix_path: Path to the computed View Factor sparse matrix file.
        eps_matrix_path: Path to the computed Emissivity surface vector file.
        rho_matrix_path: Path to the computed Reflectivity surface matrix file.
        tau_matrix_path: Path to the computed Transmissivity surface matrix file.
        inversion_parameters: Custom tracking options for the matrix solvers.
        buildings: Collection of raw input layout definitions.
    """

    workspace_dir: DirectoryPath
    epw_path: FilePath
    num_ts_per_h: int

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
