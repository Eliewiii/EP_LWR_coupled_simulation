"""
Functions to read/write crs sparse matrices
"""

import os

from scipy.sparse import csr_matrix, save_npz, load_npz, issparse


def read_csr_matrices_from_npz(*path_csr_matrix_npz_files):
    """
    Read multiple sparse matrices from a single .npz file.
    :param path_file:
    :param matrices_id:
    :return:
    """
    # Check if the files exist
    for pah_file in path_csr_matrix_npz_files:
        if not os.path.isfile(pah_file):
            raise FileNotFoundError(f"The matrix file {pah_file} does not exist")
    #Load the matrices
    mtx_list = []

    for path_file in path_csr_matrix_npz_files:
        mtx_list.append(load_npz(path_file))

    return mtx_list