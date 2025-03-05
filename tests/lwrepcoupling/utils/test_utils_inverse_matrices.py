"""

"""

import os

from lwrepcoupling.utils import read_csr_matrices_from_npz, compute_full_inverse_via_gmres,compute_full_inverse_via_gmres_parallel
from lwrepcoupling.utils.utils_resolution_matrices import compute_f_star_rho

# Get file location
tests_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
path_dir_mtx = os.path.join(tests_dir, "data", "matrix_samples")
name_file_vf_matrix = "vf_mtx"
name_file_rho_matrix = "reflectance_mtx"


def test_inverse():
    """

    :return:
    """
    mtx_dict = read_csr_matrices_from_npz(path_dir_mtx, *[name_file_rho_matrix, name_file_vf_matrix])

    f_star_rho = compute_f_star_rho(vf_matrix=mtx_dict[name_file_vf_matrix],
                                    rho_matrix=mtx_dict[name_file_rho_matrix])

    compute_full_inverse_via_gmres_parallel(mtx=f_star_rho,rtol=1e-6)
