""" """

import logging
from typing import Any

import scipy.sparse as sp
from scipy.sparse import csr_matrix

from .utils_inverse_matrices import InversionConfig, compute_full_inverse_via_gmres_parallel

logger = logging.getLogger(__name__)


def check_matrices(*mtx: Any):
    """
    Verify that the input matrices are sparse, square and have the same size.
    Args:
        mtx (csr_matrix): Variable number of matrices to check.

    Raises:
        ValueError: _description_
        ValueError: _description_
    """
    matrix_size = mtx[0].shape[0]
    for m in mtx:
        if not sp.issparse(m):
            raise ValueError(f"Matrix '{m}' is not sparse! Please provide a sparse matrix.")
        if m.shape[0] != matrix_size or m.shape[1] != matrix_size:
            raise ValueError("Matrices should have the same size and be square!")


def compute_total_vf(vf_matrix: csr_matrix) -> list[float]:
    """
    Compute the total view factors for each surface (sum over rows).
    Args:
        vf_matrix (scipy.sparse.csr_matrix): View factor matrix.
    Returns:
        list[float]: Total view factors for each surface.
    """
    vf_tot_array = vf_matrix.sum(axis=1).A1
    # Check if the total view factors is in [0, 1]
    invalid_indices = [i for i, vf in enumerate(vf_tot_array) if not (0.0 <= vf <= 1.0)]
    if invalid_indices:
        for idx in invalid_indices:
            logger.error(
                f"CRITICAL ENERGY CONSERVATION VIOLATION: Surface {idx} has a total "
                f"view factor of {vf_tot_array[idx]:.8f}. Must be strictly between 0 and 1."
            )

        raise ValueError(
            f"Matrix Inversion Aborted: {len(invalid_indices)} surfaces violated energy "
            f"conservation. Non-invertible system detected."
        )
    # Convert the clean NumPy array to a list only at the return boundary
    return vf_tot_array.tolist()


def compute_f_star_rho(vf_matrix: Any, rho_matrix: sp.csr_matrix) -> sp.csr_matrix:
    """
    Compute the F^{*rho} matrix, defined in the documentation as:
      F^{*rho} = I - rho_matrix @ vf_matrix, where I is the identity matrix.

    Args:
        vf_matrix (Any): _description_
        rho_matrix (sp.csr_matrix): _description_

    Returns:
        sp.csr_matrix: F^{*rho} matrix in sparse format.
    """
    n = vf_matrix.shape[0]
    id_mtx = sp.eye(n, format="csr")
    f_star_rho = id_mtx - rho_matrix @ vf_matrix

    return f_star_rho


def compute_resolution_matrices(
    vf_matrix: Any,
    eps_matrix: sp.csr_matrix,
    rho_matrix: sp.csr_matrix,
    tau_matrix: sp.csr_matrix,
    inversion_config: InversionConfig,
) -> tuple[sp.csr_matrix, list[float]]:
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
    id_mtx = sp.eye(n, format="csr")
    # F^{*rho} matrix to inverse
    f_star_rho = compute_f_star_rho(vf_matrix=vf_matrix, rho_matrix=rho_matrix)
    inv_f_star_rho, _ = compute_full_inverse_via_gmres_parallel(
        mtx=f_star_rho, config=inversion_config
    )

    # Get total VF from surrounding surfaces
    total_srd_vf_list = compute_total_vf(vf_matrix=vf_matrix)
    # Get (F^{srd,\epsilon})^{-1}
    diag_f_srd_epsilon_inv = [
        1 / (total_srd_vf_list[i] * eps_matrix[i, i])
        if total_srd_vf_list[i] * eps_matrix[i, i] != 0
        else 0
        for i in range(n)
    ]
    inv_f_srd_epsilon = sp.diags(diag_f_srd_epsilon_inv, offsets=0, format="csr")
    # Get F^{srd,*}
    f_srd_star = sp.diags([1.0 - vf_srd for vf_srd in total_srd_vf_list], offsets=0, format="csr")

    # Final resolution matrix
    # resolution_mtx = (  # Note: the original formula included a tau_matrix @ vf_matrix term, but it is not present in the final implementation.
    #     inv_f_srd_epsilon
    #     @ ((id_mtx - vf_matrix + tau_matrix @ vf_matrix) @ inv_f_star_rho - f_srd_star)
    #     @ eps_matrix
    # )
    resolution_mtx = (
        inv_f_srd_epsilon @ ((id_mtx - vf_matrix) @ inv_f_star_rho - f_srd_star) @ eps_matrix
    )

    # Check the result is csr
    if not sp.issparse(resolution_mtx):
        resolution_mtx = sp.csr_matrix(resolution_mtx)

    return resolution_mtx, total_srd_vf_list
