"""
Module handling file I/O operations, matrix serialization, and legacy native IDF generation
pipelines.
"""

from pathlib import Path

import numpy as np
from pyenergyplus.model import EnergyPlusModel

from .idf_exporter import write_native_idf


def save_and_convert_to_idf(
    ep_model: EnergyPlusModel, target_dir: Path, base_filename: str, lwr_coupling_text: str = ""
) -> Path:
    """
    Natively serializes an EnergyPlusModel directly down to a pristine, augmented legacy IDF
    text file.

    Bypasses external command-line compilers by utilizing a native tree-walking text stream.

    Args:
        ep_model: Validated Pydantic model tracking all system components.
        target_dir: Destination path folder for output deployment.
        base_filename: String name for the exported file asset (without extension).
        lwr_coupling_text: Optional custom macro string blocks to append to the footer.

    Returns:
        Path: The absolute path to the generated .idf file asset.
    """
    target_dir.mkdir(exist_ok=True, parents=True)
    idf_path = target_dir / f"{base_filename}.idf"

    return write_native_idf(ep_model, idf_path, additional_text=lwr_coupling_text)


def serialize_numpy_matrices(
    target_dir: Path,
    prefix: str,
    vf_matrix: np.ndarray,
    eps_vector: np.ndarray,
    rho_vector: np.ndarray,
    tau_vector: np.ndarray,
) -> dict[str, Path]:
    """Commits compiled scientific matrices and matching radiative property vectors to disk.

    Automatically expands flat 1D parameter vectors into square diagonal matrices to comply
    with structural sizing constraints required by the coupled solver engine validation hooks.
    """
    target_dir.mkdir(exist_ok=True, parents=True)

    paths = {
        "vf": target_dir / f"{prefix}_vf_matrix.npy",
        "eps": target_dir / f"{prefix}_eps_vector.npy",
        "rho": target_dir / f"{prefix}_rho_vector.npy",
        "tau": target_dir / f"{prefix}_tau_vector.npy",
    }

    # 1. Save the primary geometric view factor cross-matrix as-is
    np.save(str(paths["vf"]), vf_matrix)

    # 2. Convert flat 1D vectors to square NxN diagonal matrices before serialization
    np.save(str(paths["eps"]), np.diag(eps_vector))
    np.save(str(paths["rho"]), np.diag(rho_vector))
    np.save(str(paths["tau"]), np.diag(tau_vector))

    return paths
