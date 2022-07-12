from __future__ import annotations

from collections import Sequence

import os
import pickle


def checkNgen_folder(folder_path: str) -> None:
    """
    Check if the folder or the subfolder exists
    to create a new directory if not
    
    Args:
    - folder_path: str, the folder path
    """

    if not os.path.exists(folder_path):
        print(f"Making {folder_path}...")
        os.mkdir(folder_path)

def replace_ext(input_path: str, ext: str) -> str:
    
    """
    Replace a file extention of a path to another

    Args:
    - input_path: str, the input path of the file
    - ext: str, the new file extension

    Returns:
    - str, the output path with the new file extension
    """
    
    if ext[0] == ".":
        return os.path.splitext(input_path)[0] + ext
    else:
        return os.path.splitext(input_path)[0] + "." + ext

def pickle_save(what2save, where2save: str) -> None:

    """
    Save variable to a pickle file

    Args:
    - what2save, the varible that needs to be saved
    - where2save: str, the .pkl path for saving
    """

    with open(where2save, "wb") as f:
        pickle.dump(what2save, f)

def pickle_load(path2load: str):

    """
    Load pickle file

    Args:
    - path2load: str, the .pkl path for loading
    """

    with open(path2load, "rb") as f:
        return pickle.load(f)