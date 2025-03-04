"""

"""

from typing import List

import scipy.sparse as sp


def check_matrices(*matrix):
    """
    Check if the given matrices are sparse matrices.
    :param matrix: scipy.sparse.csr_matrix, matrices to be checked.
    :return: bool, True if all matrices are sparse matrices, False otherwise.
    """
    matrix_size = matrix[0].shape[0]
    for m in matrix:
        if not sp.issparse(m):
            raise ValueError(f"Matrix '{m}' is not sparse! Please provide a sparse matrix.")
        if m.shape[0] != matrix_size or m.shape[1] != matrix_size:
            raise ValueError(f"Matrices should have the same size and be square!")


def compute_total_vf(vf_matrix: sp.csr_matrix) -> List[float]:
    """
    Compute the total view factors for each surface.
    :param vf_matrix: scipy.sparse.csr_matrix, view factor matrix.
    :return: numpy.ndarray, total view factors for each surface.
    """
    vf_tot_list = list(vf_matrix.sum(axis=1).A1)
    # Check if the total view factors is in [0, 1]
    for vf in vf_tot_list:
        if not 0 <= vf <= 1:
            raise ValueError("Total view factors should be in [0, 1]")

    return vf_tot_list


def compute_f_star_rho(vf_matrix: sp.csr_matrix,
                       rho_matrix: sp.csr_matrix):
    # Size of the matrix
    n = vf_matrix.shape[0]
    # Identity matrix
    id_mtx = sp.eye(n, format='csr')
    # F^{*rho} matrix to inverse
    f_star_rho = id_mtx - rho_matrix.dot(vf_matrix)

    return f_star_rho


def compute_resolution_matrix(vf_matrix: sp.csr_matrix,
                              eps_matrix: sp.csr_matrix, rho_matrix: sp.csr_matrix,
                              tau_matrix: sp.csr_matrix):
    """

    :param vf_matrix:
    :param eps_matrix:
    :param rho_matrix:
    :param tau_matrix:
    :return:
    """
    # Size of the matrix
    n = vf_matrix.shape[0]
    # Identity matrix
    id_mtx = sp.eye(n, format='csr')
    # F^{*rho} matrix to inverse
    f_star_rho = compute_f_star_rho(vf_matrix=vf_matrix, rho_matrix=rho_matrix)
