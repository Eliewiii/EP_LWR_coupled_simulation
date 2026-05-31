import logging

import numpy as np
import pytest
from lwrepcoupling._schemas import InversionConfig
from lwrepcoupling._utils import compute_full_inverse_via_gmres_parallel
from pydantic import ValidationError
from scipy.sparse import csr_matrix

# =====================================================================
# 1. Testing Input Boundaries (The Pydantic Contract)
# =====================================================================


def test_inversion_config_defaults() -> None:
    """Verify that default settings apply correctly."""
    config = InversionConfig()
    assert config.tol == 1e-5
    assert config.maxiter == 150
    assert config.num_workers == 1


def test_inversion_config_cpu_fallback() -> None:
    """Verify that setting num_workers=0 invokes the system CPU fallback."""
    config = InversionConfig(num_workers=0)
    # The custom validator should turn 0 into a positive integer (CPU cores)
    assert config.num_workers >= 1


def test_inversion_config_out_of_bounds_throws() -> None:
    """Verify constraints safely reject invalid physics or solver inputs."""
    with pytest.raises(ValidationError):
        # maxiter must be <= 1000
        InversionConfig(maxiter=5000)

    with pytest.raises(ValidationError):
        # tol must be >= 1e-8
        InversionConfig(tol=1e-10)


# =====================================================================
# 2. Testing Numerical Execution (The Math Gate)
# =====================================================================


def test_identity_matrix_inversion() -> None:
    """Inverting an identity matrix should yield an identical identity matrix."""
    id_mtx = csr_matrix(np.eye(3))
    config = InversionConfig(num_workers=1, tol=1e-5)

    inv_approx, error_norm = compute_full_inverse_via_gmres_parallel(id_mtx, config)

    assert error_norm < config.tol
    np.testing.assert_array_almost_equal(inv_approx.toarray(), np.eye(3))


def test_diagonal_matrix_inversion() -> None:
    """Inverting a simple diagonal matrix with preconditioning."""
    # Matrix: diag([2, 4, 5]) -> Inverse should be diag([0.5, 0.25, 0.2])
    A = csr_matrix(np.diag([2.0, 4.0, 5.0]))
    config = InversionConfig(precondition=True, tol=1e-6)

    inv_approx, error_norm = compute_full_inverse_via_gmres_parallel(A, config)

    expected_inverse = np.diag([0.5, 0.25, 0.2])
    assert error_norm < config.tol
    np.testing.assert_array_almost_equal(inv_approx.toarray(), expected_inverse)


# =====================================================================
# 3. Testing Failure Handling & Logging Control
# =====================================================================


def test_strict_tolerance_failure_behavior() -> None:
    """If error exceeds tol and strict_tol is True, it must raise a ValueError."""
    # An ill-conditioned matrix that will fail a high-precision convergence requirement
    A = csr_matrix(np.array([[1e-3, 1e-4], [1e-11, 1e-12]]))

    # Set a strict tolerance and small maxiter to guarantee failure
    config = InversionConfig(tol=1e-8, maxiter=2, strict_tol=True)

    with pytest.raises(ValueError, match="Accuracy check failed"):
        compute_full_inverse_via_gmres_parallel(A, config)


def test_gmres_failure_behavior(caplog: pytest.LogCaptureFixture) -> None:
    """Verify that worker convergence failures log explicit warnings if strict_tol is False."""
    # An ill-conditioned matrix that will fail a high-precision convergence requirement
    A = csr_matrix(np.array([[1e-3, 1e-4], [1e-11, 1e-12]]))

    # Set a strict tolerance and small maxiter to guarantee failure
    config = InversionConfig(tol=1e-8, maxiter=2, strict_tol=False)

    with caplog.at_level(logging.WARNING):
        compute_full_inverse_via_gmres_parallel(A, config)

        # Check that the lower-level column execution loop captured the convergence warning
        assert any("GMRES did not converge" in record.message for record in caplog.records)


def test_non_invertible_matrix_behavior(caplog: pytest.LogCaptureFixture) -> None:
    """Ensure a completely singular matrix is caught and flagged by precision checks."""
    # A 3x3 matrix of pure zeros is completely singular (non-invertible)
    singular_mtx = csr_matrix((3, 3), dtype=float)

    # Case A: With strict_tol=False, it should log warnings but return gracefully
    config_lax = InversionConfig(strict_tol=False, maxiter=5)

    with caplog.at_level(logging.WARNING):
        inv_approx, error_norm = compute_full_inverse_via_gmres_parallel(singular_mtx, config_lax)

    # Verify the code warned the user that the global system precision collapsed
    assert any("Accuracy check failed" in record.message for record in caplog.records)

    # Case B: With strict_tol=True, it must immediately raise a ValueError
    config_strict = InversionConfig(strict_tol=True, maxiter=5)

    with pytest.raises(ValueError, match="Accuracy check failed"):
        compute_full_inverse_via_gmres_parallel(singular_mtx, config_strict)
