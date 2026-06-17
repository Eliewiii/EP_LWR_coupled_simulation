"""Functions to read/write csr sparse matrices."""

from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix, issparse, load_npz


def read_csr_matrices_from_npz(*path_csr_matrix_npz_files: Path) -> list[Any]:
    """Load multiple matrices from file paths, supporting both SciPy sparse .npz and dense .npy format.

    Args:
        path_csr_matrix_npz_files: Variadic list of Paths to matrix tracking files.

    Raises:
        FileNotFoundError: If any of the requested files do not exist on disk.

    Returns:
        list[csr_matrix]: A list of validated matrices cast safely to csr format.
    """
    for path_file in path_csr_matrix_npz_files:
        if not path_file.exists():
            raise FileNotFoundError(f"The matrix file {path_file} does not exist")

    mtx_list: list[Any] = []
    for path_file in path_csr_matrix_npz_files:
        # --- DYNAMIC FORMAT SNIFFING LAYER ---
        # SciPy sparse matrices saved with save_npz typically use a .npz extension,
        # while dense matrices in our benchmarks use .npy via np.save
        if path_file.suffix.lower() == ".npy":
            # Load standard dense numpy array natively
            raw_data = np.load(path_file)
            mtx = csr_matrix(raw_data)
        else:
            try:
                # Attempt standard SciPy sparse extraction
                mtx = load_npz(path_file)
                if not issparse(mtx):
                    mtx = csr_matrix(mtx)
            except TypeError:
                # Defensive fallback: If load_npz fails due to an underlying dense array format,
                # catch it safely and pass it directly to the numpy recovery tracker
                raw_data = np.load(path_file)
                mtx = csr_matrix(raw_data)

        mtx_list.append(mtx)

    return mtx_list
