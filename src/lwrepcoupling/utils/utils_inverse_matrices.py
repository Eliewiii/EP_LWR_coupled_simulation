"""
Functions for matrix inversion
"""

import numpy as np

from concurrent.futures import ProcessPoolExecutor

from scipy.sparse.linalg import gmres
from scipy.sparse import csr_matrix, diags


def compute_full_inverse_via_gmres(mtx, tol=1e-5, maxiter=100, rtol=1e-5, precondition=False):
    """
    Approximates the inverse of a sparse matrix mtx using the GMRES method for all columns,
    with optional preconditioning to improve convergence, while keeping the result sparse.

    Args:
        mtx (csr_matrix): Sparse matrix to be inverted.
        tol (float): Tolerance for checking accuracy of the approximate inverse.
        maxiter (int): Maximum number of iterations for GMRES.
        precondition (bool): Whether to use preconditioning (Jacobi preconditioner).

    Returns:
        csr_matrix: Approximate inverse of the matrix (in CSR format).
        float: Frobenius norm of the error between A * inverse_approx and identity matrix.
    """
    # Get the size of the matrix (assuming it's square)
    n = mtx.shape[0]

    # Create an empty sparse matrix to store the result (approximate inverse)
    inverse_approx = csr_matrix((n, n), dtype=np.float64)

    # Create preconditioner if needed (Jacobi Preconditioner: diagonal inverse)
    M_inv = None
    if precondition:
        # Create a diagonal preconditioner (inverse of the diagonal)
        M_inv = diags(1 / mtx.diagonal())  # Inverse of the diagonal of A

    # Solve for each column of the identity matrix (i.e., the inverse columns)
    for i in range(n):
        # Create the right-hand side as the i-th column of the identity matrix
        b = np.zeros(n)
        b[i] = 1  # Identity column

        # Solve mtx * x = b using GMRES (x will be the i-th column of the inverse)
        x, exitCode = gmres(mtx, b, M=M_inv, maxiter=maxiter, rtol=rtol)

        # If GMRES didn't converge, you may want to handle this case
        if exitCode != 0:
            print(f"Warning: GMRES did not converge for column {i}. Exit code: {exitCode}")

        x_sparse = csr_matrix(x.reshape(-1, 1))

        # Store the result (x is the approximation for the i-th column of the inverse)
        inverse_approx[:, i] = x_sparse  # Store as sparse matrix column

    # Calculate the product A * inverse_approx and compare it to the identity matrix
    identity_approx = mtx.dot(inverse_approx)

    # Compute the Frobenius norm of the error: ||A * inverse_approx - I||
    error_matrix = identity_approx - csr_matrix(np.eye(n))
    error_norm = np.linalg.norm(error_matrix.toarray(),
                                'fro')  # Convert sparse result to dense for error calculation

    # Check if the error is below the tolerance
    if error_norm < tol:
        print(f"Accuracy check passed: Error norm {error_norm:.2e} is below tolerance.")
    else:
        print(f"Accuracy check failed: Error norm {error_norm:.2e} is above tolerance.")

    return inverse_approx, error_norm



def solve_for_column_wrapper(args):
    """Wrapper function for solving a single column of the inverse matrix."""
    mtx, i, M_inv, maxiter, rtol = args
    n = mtx.shape[0]
    b = np.zeros(n)
    b[i] = 1  # Create the i-th column of the identity matrix
    x, exitCode = gmres(mtx, b, M=M_inv, maxiter=maxiter, rtol=rtol)

    if exitCode != 0:
        print(f"Warning: GMRES did not converge for column {i}. Exit code: {exitCode}")

    return i, x  # Return column index and result to maintain order


def compute_full_inverse_via_gmres_parallel(mtx, tol=1e-5, maxiter=100, rtol=1e-5, precondition=False,
                                            num_workers=None):
    """
    Approximates the inverse of a sparse matrix mtx using GMRES in parallel for all columns.

    Args:
        mtx (csr_matrix): Sparse matrix to be inverted.
        tol (float): Tolerance for checking accuracy.
        maxiter (int): Maximum GMRES iterations.
        rtol (float): GMRES relative tolerance.
        precondition (bool): Whether to use preconditioning (Jacobi preconditioner).
        num_workers (int, optional): Number of parallel processes (default: number of CPU cores).

    Returns:
        csr_matrix: Approximate inverse of the matrix (in CSR format).
        float: Frobenius norm of the error between A * inverse_approx and identity matrix.
    """
    # Get matrix size (assuming it's square)
    n = mtx.shape[0]

    # Create preconditioner if needed (Jacobi Preconditioner: inverse of diagonal)
    M_inv = None
    if precondition:
        M_inv = diags(1 / mtx.diagonal())

    # Use ProcessPoolExecutor for parallel execution
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(
            executor.map(solve_for_column_wrapper, [(mtx, i, M_inv, maxiter, rtol) for i in range(n)]))

    # Ensure results are sorted in column order (executor.map() preserves order, but just to be safe)
    results.sort(key=lambda x: x[0])
    inverse_columns = [x[1] for x in results]

    # Assemble the result as a sparse matrix
    inverse_approx = csr_matrix(np.column_stack(inverse_columns))

    # Compute accuracy check
    identity_approx = mtx.dot(inverse_approx)
    error_matrix = identity_approx - csr_matrix(np.eye(n))
    error_norm = np.linalg.norm(error_matrix.toarray(), 'fro')

    if error_norm < tol:
        print(f"✅ Accuracy check passed: Error norm {error_norm:.2e} is below tolerance.")
    else:
        print(f"⚠️ Accuracy check failed: Error norm {error_norm:.2e} is above tolerance.")

    return inverse_approx, error_norm


