from typing import Annotated
import os

from pydantic import (
    BaseModel,
    DirectoryPath,
    Field,
    FilePath,
    field_validator,
    ConfigDict
)

class InversionConfig(BaseModel):
    """Configuration schema defining convergence metrics and performance settings

    for the sparse matrix linear inversion numerical solver.
    """

    model_config = ConfigDict(frozen=True)

    # Boundaries, Constraints, and Runtime Descriptions
    tol: Annotated[float, Field(ge=1e-8, le=1e-4, description=(
        "Absolute error tolerance threshold for mathematical convergence tracking."
    ))] = 1e-5

    maxiter: Annotated[int, Field(ge=2, le=1000, description=(
        "The upper boundary limit for linear system solver iteration loops before aborting."
    ))] = 150

    rtol: Annotated[float, Field(ge=1e-10, le=1e-5, description=(
        "Relative tolerance stopping criterion representing residual reduction scaling limits."
    ))] = 5e-7

    precondition: bool = Field(default=False, description=(
        "Toggle to dynamically construct a Jacobi Preconditioner to accelerate convergence speeds."
    ))

    strict_tol: bool = Field(default=False, description=(
        "If True, raises an error if the final error norm remains above the absolute tolerance."
    ))

    num_workers: Annotated[int, Field(ge=0, le=60, description=(
        "Parallel CPU allocation target. Setting to 0 automatically provisions all available cores."
    ))] = 1

    # Custom validation
    @field_validator("num_workers")
    @classmethod
    def check_workers(cls, v: int) -> int:
        """Dynamically evaluate CPU topology when 0 is requested."""
        if v == 0:
            return os.cpu_count() or 1
        return v


class BuildingInput(BaseModel):
    """Raw incoming building layout definition from the upstream application."""

    building_id: str = Field(description=(
        "Unique alphanumeric string identifying the specific building within the simulation environment."
    ))
    
    idf_path: FilePath = Field(description=(
        "Absolute filesystem path to the building's EnergyPlus input data dictionary (.idf) file."
    ))
    
    outdoor_surface_names: list[str] = Field(description=(
        "Collection of outdoor surface identifiers participating in the localized long-wave radiation exchange loop."
    ))

    @field_validator("idf_path", mode="after")
    @classmethod
    def enforce_idf_extension(cls, v: FilePath) -> FilePath:
        """Enforces that the incoming layout file uses a valid .idf extension."""
        if v.suffix.lower() != ".idf":
            raise ValueError(
                f"Invalid file type for building layout. Expected a '.idf' file, "
                f"but received '{v.name}' instead."
            )
        return v


class SimulationInputs(BaseModel):
    """The memory container used to initialize and compile the simulation workspace."""

    workspace_dir: DirectoryPath = Field(description=(
        "Target root directory path where the runtime folders, matrices, and simulation sandboxes will be built."
    ))
    
    epw_path: FilePath = Field(description=(
        "Absolute filesystem path to the target EnergyPlus weather (.epw) climate asset file."
    ))
    
    num_ts_per_h: int = Field(description=(
        "Number of simulation time steps per hour. This must match the internal frequency defined in the building IDFs."
    ))

    vf_matrix_path: FilePath = Field(description=(
        "Absolute filesystem path to the computed geometric View Factor (.npy/.npz) sparse matrix file."
    ))
    
    eps_matrix_path: FilePath = Field(description=(
        "Absolute filesystem path to the computed material Emissivity surface properties vector data file."
    ))
    
    rho_matrix_path: FilePath = Field(description=(
        "Absolute filesystem path to the computed diffuse Reflectivity structural surface matrix data file."
    ))
    
    tau_matrix_path: FilePath = Field(description=(
        "Absolute filesystem path to the computed material Transmissivity shortwave matrix data file."
    ))

    inversion_parameters: InversionConfig = Field(default_factory=InversionConfig, description=(
        "Custom parameter convergence adjustments, bounds, and thresholds configured for the matrix solvers."
    ))
    
    buildings: list[BuildingInput] = Field(description=(
        "A structured assembly containing the raw input layouts and configuration states for all buildings."
    ))

    save_resolution_matrix: bool = Field(default=False, description=(
        "Toggle to explicitly commit the full assembled master resolution mapping matrix (.npz) to the disk space."
    ))
    enable_lwr_coupling: bool = Field(default=True, description=(
        "If True, activates the shared-memory synchronization loop to dynamically couple "
        "inter-building long-wave radiation exchange. If False, buildings run completely uncoupled."
    ))

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

    # Batch-validate all data matrix assets in one clean block
    @field_validator(
        "vf_matrix_path", "eps_matrix_path", "rho_matrix_path", "tau_matrix_path", mode="after"
    )
    @classmethod
    def enforce_matrix_extensions(cls, v: FilePath) -> FilePath:
        """Enforces that all structural sparse matrices use verified computational formats."""
        allowed_extensions = {".npy", ".npz"}

        if v.suffix.lower() not in allowed_extensions:
            raise ValueError(
                f"Invalid matrix file format. Expected a numpy storage file ({allowed_extensions}),"
                f" but received '{v.name}' instead."
            )
        return v

    @property
    def num_total_surfaces(self) -> int:
        """Calculates the aggregate outdoor surface count across all inputs."""
        return sum(len(b.outdoor_surface_names) for b in self.buildings)
