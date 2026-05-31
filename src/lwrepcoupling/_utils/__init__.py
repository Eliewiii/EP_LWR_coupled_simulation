"""Public API gateway for the Long-Wave Radiation (LWR) coupling utility layer."""

from .utils_csr_matrices import read_csr_matrices_from_npz
from .utils_idf_additional_strings import generate_idfs_additional_strings
from .utils_inverse_matrices import compute_full_inverse_via_gmres_parallel
from .utils_io import assert_path_is_safe_for_purging
from .utils_resolution_matrices import check_matrices, compute_resolution_matrices, compute_total_vf

# The explicit public surface contract scrutinized by static type analyzers (Pyright/Pylance)
__all__ = [
    "read_csr_matrices_from_npz",
    "generate_idfs_additional_strings",
    "compute_full_inverse_via_gmres_parallel",
    "assert_path_is_safe_for_purging",
    "check_matrices",
    "compute_resolution_matrices",
    "compute_total_vf",
]
