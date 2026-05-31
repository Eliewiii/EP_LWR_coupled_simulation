import numpy as np
import pytest
import scipy.sparse as sp
from lwrepcoupling._utils.utils_inverse_matrices import InversionConfig
from lwrepcoupling._utils.utils_resolution_matrices import (
    check_matrices,
    compute_f_star_rho,
    compute_resolution_matrices,
)

# =====================================================================
# FIXTURES
# =====================================================================


@pytest.fixture
def clean_inversion_config() -> InversionConfig:
    """Provides a basic, fast configuration for testing execution logic."""
    return InversionConfig(num_workers=1, maxiter=10, tol=1e-4, strict_tol=False)


@pytest.fixture
def valid_3x3_physics_setup() -> tuple[sp.spmatrix, sp.spmatrix, sp.spmatrix, sp.spmatrix]:
    """Creates a mathematically consistent 3x3 layout where rows sum to 1.0."""
    vf_data = np.array([[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]])
    vf_matrix = sp.csr_matrix(vf_data)

    eps_matrix = sp.diags([0.9, 0.8, 0.85], format="csr")
    rho_matrix = sp.diags([0.1, 0.2, 0.15], format="csr")
    tau_matrix = sp.diags([0.0, 0.0, 0.0], format="csr")

    return vf_matrix, eps_matrix, rho_matrix, tau_matrix


# =====================================================================
# TESTS
# =====================================================================


def test_compute_resolution_matrices_return_signature(
    valid_3x3_physics_setup, clean_inversion_config
) -> None:
    """Verify that the function matches its typed contract output (tuple of matrix and list)."""
    vf, eps, rho, tau = valid_3x3_physics_setup

    result = compute_resolution_matrices(
        vf_matrix=vf,
        eps_matrix=eps,
        rho_matrix=rho,
        tau_matrix=tau,
        inversion_config=clean_inversion_config,
    )

    assert isinstance(result, tuple)
    assert len(result) == 2

    resolution_mtx, total_srd_vf_list = result
    assert sp.issparse(resolution_mtx)
    assert resolution_mtx.format == "csr"
    assert isinstance(total_srd_vf_list, list)
    assert all(isinstance(val, float) for val in total_srd_vf_list)


def test_energy_conservation_violation_halts_execution(clean_inversion_config) -> None:
    """If row sums break physical boundary laws (> 1.0), it must abort before resolving."""
    broken_vf_data = np.array([[0.5, 0.5, 0.5], [0.2, 0.2, 0.2], [0.1, 0.1, 0.1]])
    broken_vf = sp.csr_matrix(broken_vf_data)
    identity_diag = sp.csr_matrix(sp.eye(3, format="csr"))

    with pytest.raises(ValueError, match="Matrix Inversion Aborted"):
        compute_resolution_matrices(
            vf_matrix=broken_vf,
            eps_matrix=identity_diag,
            rho_matrix=identity_diag,
            tau_matrix=identity_diag,
            inversion_config=clean_inversion_config,
        )


def test_division_by_zero_handling_for_isolated_surfaces(clean_inversion_config) -> None:
    """Ensure surfaces with a total view factor of 0 don't cause a zero-division crash."""
    isolated_vf_data = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    isolated_vf = sp.csr_matrix(isolated_vf_data)

    eps = sp.csr_matrix(sp.diags([0.9, 0.9, 0.9], format="csr"))
    rho = sp.csr_matrix(sp.diags([0.1, 0.1, 0.1], format="csr"))
    tau = sp.csr_matrix(sp.diags([0.0, 0.0, 0.0], format="csr"))

    resolution_mtx, total_srd_vf_list = compute_resolution_matrices(
        vf_matrix=isolated_vf,
        eps_matrix=eps,
        rho_matrix=rho,
        tau_matrix=tau,
        inversion_config=clean_inversion_config,
    )
    assert total_srd_vf_list[2] == 0.0


def test_check_matrices_validation_profiles() -> None:
    """Verify check_matrices handles type verification, sizing, and square checks correctly."""
    valid_m1 = sp.csr_matrix(sp.eye(4, format="csr"))
    valid_m2 = sp.csr_matrix(sp.diags([1.0, 2.0, 3.0, 4.0], format="csr"))

    check_matrices(valid_m1, valid_m2)

    dense_matrix = np.eye(4)
    with pytest.raises(ValueError, match="is not sparse"):
        check_matrices(valid_m1, dense_matrix)


def test_compute_f_star_rho_numerical_correctness() -> None:
    """Verify the arithmetic formula execution of F^{*rho} = I - rho @ VF."""
    vf_matrix = sp.csr_matrix(np.array([[0.0, 1.0], [1.0, 0.0]]))
    rho_matrix = sp.csr_matrix(sp.diags([0.5, 0.2], format="csr"))

    expected_f_star = np.array([[1.0, -0.5], [-0.2, 1.0]])
    res_f_star_rho = compute_f_star_rho(vf_matrix, rho_matrix)

    assert sp.issparse(res_f_star_rho)
    np.testing.assert_array_almost_equal(res_f_star_rho.toarray(), expected_f_star)
