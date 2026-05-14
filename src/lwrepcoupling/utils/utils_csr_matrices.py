"""
Functions to read/write crs sparse matrices
"""

from pathlib import Path

from scipy.sparse import csr_matrix, load_npz


def read_csr_matrices_from_npz(*path_csr_matrix_npz_files: str) -> list[csr_matrix]:
    """Load multiple sparse matrices from a single .npz file.

    Args:
            path_csr_matrix_npz_files (str): Path to the csr matrix file.

    Raises:
        FileNotFoundError: If the file does not exist.

    Returns:
        list[csr_matrix]: A list of sparse matrices in csr format.
    """

    for path_file in path_csr_matrix_npz_files:
        path_file = Path(path_file)
        if not path_file.exists():
            raise FileNotFoundError(f"The matrix file {path_file} does not exist")
    # Load the matrices
    mtx_list: list[csr_matrix] = []
    for path_file in path_csr_matrix_npz_files:
        mtx_list.append(load_npz(path_file))

    return mtx_list
