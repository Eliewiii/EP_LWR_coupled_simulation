"""
Functions to read/write crs sparse matrices
"""

import os

from scipy.sparse import csr_matrix, save_npz, load_npz, issparse


def read_csr_matrices_from_npz(path_dir, *matrix_file_names):
    """
    Read multiple sparse matrices from a single .npz file.
    :param path_file:
    :param matrices_id:
    :return:
    """
    # Check if the files exist
    for name in matrix_file_names:
        if not os.path.isfile(os.path.join(path_dir, name + ".npz")):
            raise FileNotFoundError(f"The matrix file {os.path.join(path_dir, name + '.npz')} does not exist")
    #Load the matrices
    matrix_dict = {}

    for name in matrix_file_names:
        matrix_dict[name] = load_npz(os.path.join(path_dir, name + ".npz"))

    return matrix_dict