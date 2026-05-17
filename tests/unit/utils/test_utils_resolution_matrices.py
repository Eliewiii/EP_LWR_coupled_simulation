import numpy as np
import pytest
import scipy.sparse as sp

# Replace 'your_module' with the actual file path names in your package
from your_module import InversionConfig, compute_resolution_matrices

# =====================================================================
# FIXTURES: Setting up reusable physical states
# =====================================================================


@pytest.fixture
def clean_inversion_config():
    """Provides a basic, fast configuration for testing execution logic."""
    return InversionConfig(num_workers=1, maxiter=10, tol=1e-4, strict_tol=False)


@pytest.fixture
def valid_3x3_physics_setup():
    """
    Creates a mathematically consistent 3x3 layout.
    The rows of the view factor matrix sum up to exactly 1.0 (Closed cavity).
    """
    # 3x3 Enclosed cavity view factors (Rows sum to 1.0)
    vf_data = np.array([[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]])
    vf_matrix = sp.csr_matrix(vf_data)

    # Standard environmental physics properties (diagonal parameters)
    eps_matrix = sp.diags([0.9, 0.8, 0.85], format="csr")  # Emissivity
    rho_matrix = sp.diags([0.1, 0.2, 0.15], format="csr")  # Reflectivity
    tau_matrix = sp.diags([0.0, 0.0, 0.0], format="csr")  # Transmissivity (Opaque)

    return vf_matrix, eps_matrix, rho_matrix, tau_matrix


# =====================================================================
# 1. STRUCTURAL & CONTRACT TYPE TESTS
# =====================================================================


def test_compute_resolution_matrices_return_signature(
    valid_3x3_physics_setup, clean_inversion_config
):
    """Verify that the function matches its typed contract output (tuple of matrix and list)."""
    vf, eps, rho, tau = valid_3x3_physics_setup

    result = compute_resolution_matrices(
        vf_matrix=vf,
        eps_matrix=eps,
        rho_matrix=rho,
        tau_matrix=tau,
        inversion_config=clean_inversion_config,
    )

    # 1. Verify it returns a structural tuple
    assert isinstance(result, tuple)
    assert len(result) == 2

    resolution_mtx, total_srd_vf_list = result

    # 2. Check types match the specific tuple annotations precisely
    assert sp.issparse(resolution_mtx)
    assert resolution_mtx.format == "csr"
    assert isinstance(total_srd_vf_list, list)
    assert all(isinstance(val, float) for val in total_srd_vf_list)


# =====================================================================
# 2. PHYSICAL SOUNDNESS & EXCEPTION GATEWAY TESTS
# =====================================================================


def test_energy_conservation_violation_halts_execution(clean_inversion_config):
    """If row sums break physical boundary laws (> 1.0), it must abort before resolving."""
    # A broken matrix where row 0 leaks energy (Sums to 1.5)
    broken_vf_data = np.array(
        [
            [0.5, 0.5, 0.5],  # Sum = 1.5 -> Violation!
            [0.2, 0.2, 0.2],
            [0.1, 0.1, 0.1],
        ]
    )
    broken_vf = sp.csr_matrix(broken_vf_data)

    identity_diag = sp.eye(3, format="csr")

    # The inner compute_total_vf step should catch this and trigger a ValueError
    with pytest.raises(ValueError, match="Matrix Inversion Aborted|Total view factors"):
        compute_resolution_matrices(
            vf_matrix=broken_vf,
            eps_matrix=identity_diag,
            rho_matrix=identity_diag,
            tau_matrix=identity_diag,
            inversion_config=clean_inversion_config,
        )


def test_division_by_zero_handling_for_isolated_surfaces(clean_inversion_config):
    """Ensure surfaces with a total view factor of 0 don't cause a zero-division crash."""
    # A matrix where row 2 is completely isolated (Sums to 0.0)
    isolated_vf_data = np.array(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],  # Isolated surface (e.g. error geometry state)
        ]
    )
    isolated_vf = sp.csr_matrix(isolated_vf_data)

    eps = sp.diags([0.9, 0.9, 0.9], format="csr")
    rho = sp.diags([0.1, 0.1, 0.1], format="csr")
    tau = sp.diags([0.0, 0.0, 0.0], format="csr")

    # Should execute safely because your list comprehension/vectorized check traps the zero
    resolution_mtx, total_srd_vf_list = compute_resolution_matrices(
        vf_matrix=isolated_vf,
        eps_matrix=eps,
        rho_matrix=rho,
        tau_matrix=tau,
        inversion_config=clean_inversion_config,
    )

    # Verify the total view factor list accurately caught the zero boundary row
    assert total_srd_vf_list[2] == 0.0


# =====================================================================
# 3. CONVERGENCE INTEGRATION
# =====================================================================


def test_resolution_execution_with_strict_solver_failure():
    """Verify that configuration settings flow downward to the core solver layer."""
    # Highly unstable singular layout
    bad_vf = sp.csr_matrix(np.array([[1.0, 0.0], [0.0, 0.0]]))
    eps = sp.eye(2, format="csr")
    rho = sp.eye(2, format="csr")
    tau = sp.eye(2, format="csr")

    # Enforce a highly aggressive requirement that will fail
    strict_config = InversionConfig(tol=1e-9, maxiter=2, strict_tol=True)

    # If config properties are mapped correctly down to gmres, this should raise an error
    with pytest.raises(ValueError, match="Accuracy check failed"):
        compute_resolution_matrices(
            vf_matrix=bad_vf,
            eps_matrix=eps,
            rho_matrix=rho,
            tau_matrix=tau,
            inversion_config=strict_config,
        )
