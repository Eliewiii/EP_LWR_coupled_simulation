"""
Utility functions for folder manipulation.
"""

import os
import shutil

from pathlib import Path



def create_dir(path_dir: str, overwrite: bool = False):
    """
    Create a folder if it does not exist.
    :param path_dir: str, the path of the folder.
    :param overwrite: bool, overwrite the folder if it already
    """
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)
    elif overwrite:
        shutil.rmtree(path_dir)
        os.makedirs(path_dir)

def check_file_exist(file_path: str):
    """
    Check if a file exists and raise an error if not.
    :param file_path: str, the path of the file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")


def check_parent_folder_exist(file_path: str):
    """
    Check if the parent folder of a file path exists and raise an error if not.
    :param file_path: str, the path of the file.
    """
    file_path = Path(file_path)
    parent_folder_path = file_path.parent
    if not os.path.exists(parent_folder_path):
        raise FileNotFoundError(f"Folder not found: {parent_folder_path}")


if __name__ == "__main__":
    check_parent_folder_exist(r"../tests/test_generate_input_for_radiance.py")
