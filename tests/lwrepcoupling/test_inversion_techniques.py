"""

"""

import numpy as np
from scipy import sparse

import time


from scipy.sparse.linalg import gmres
from scipy.sparse import csr_matrix,diags

def read_and_get_f_star_rho_mtx(path_npz_file):
    """

    :return:
    """
    # Read matrix from

    npz_data = np.load(path_npz_file, allow_pickle=True)
    vf_mtx = sparse.csr_matrix(npz_data["vf_mtx"])
    rho_mtx = sparse.csr_matrix(npz_data["reflectance_mtx"])
    # Size of the matrix
    n = vf_mtx.shape[0]
    # Identity matrix
    id_mtx = sparse.eye(n, format='csr')
    # F^{*rho} matrix to inverse
    f_star_rho = id_mtx - rho_mtx.dot(vf_mtx)

    return f_star_rho

def invert_moore_penrose(mtx):
    """

    :return:
    """
    from scipy.sparse.linalg import svds

    # Using Singular Value Decomposition (SVD) to compute the pseudoinverse
    U, s, Vt = svds(mtx, k=min(mtx.shape) - 1)  # Singular Value Decomposition
    S_inv = 1.0 / s
    A_pseudo_inv = Vt.T @ np.diag(S_inv) @ U.T






def compute_full_inverse_via_gmres(mtx, tol=1e-5, maxiter=100,rtol=1e-5, precondition=False):
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
        x, exitCode = gmres(mtx, b, M=M_inv, maxiter=maxiter,rtol=rtol)

        # If GMRES didn't converge, you may want to handle this case
        if exitCode != 0:
            print(f"Warning: GMRES did not converge for column {i}. Exit code: {exitCode}")

        # Store the result (x is the approximation for the i-th column of the inverse)
        inverse_approx[:, i] = csr_matrix(x)  # Store as sparse matrix column

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


if __name__ == "__main__":
    path_matrix_npz = ""


    mtx = read_and_get_f_star_rho_mtx(path_npz_file=path_matrix_npz)

    n = mtx.shape[0]

    dur=time.time()
    compute_full_inverse_via_gmres(mtx)
    print(f"inversion for {n} surface in {dur-time.time()}s")