from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
from lwrepcoupling._utils import read_csr_matrices_from_npz


def test_read_csr_matrices_throws_on_missing_file() -> None:
    """Ensures a FileNotFoundError is raised if a matrix asset path is missing."""
    non_existent_file = Path("ghost_matrix_perimeter.npz")
    with pytest.raises(FileNotFoundError, match="does not exist"):
        read_csr_matrices_from_npz(non_existent_file)


def test_read_csr_matrices_loads_and_promotes_to_csr(tmp_path: Path) -> None:
    """Verifies that sparse structures are correctly streamed and verified as CSR format."""
    matrix_file = tmp_path / "test_matrix.npz"
    test_matrix = sp.csr_matrix(np.eye(5))
    sp.save_npz(matrix_file, test_matrix)

    loaded_matrices = read_csr_matrices_from_npz(matrix_file)

    assert len(loaded_matrices) == 1
    assert sp.issparse(loaded_matrices[0])
    assert loaded_matrices[0].format == "csr"
    assert loaded_matrices[0].shape == (5, 5)
