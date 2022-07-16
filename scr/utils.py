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

    split_list = folder_path.split("/")
    for p, _ in enumerate(split_list):
        subfolder_path = "/".join(split_list[:p+1])
        if not os.path.exists(subfolder_path):
            print(f"Making {subfolder_path}...")
            os.mkdir(subfolder_path)


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

def get_folder_file_names(
    parent_folder: str,
    dataset_path: str,
    encoder_name: str,
    embed_layer: int,
    flatten_emb: bool | str,
) -> list[str]:
    """
    A function for specify folder and file names for the output given input dataset

    Args:
    - parent_folder: str, the parent result folder, such as results/train_val_test
    - dataset_path: str, full path to the input dataset, in pkl or panda readable format
        columns include: sequence, target, set, validation, mut_name (optional), mut_numb (optional)
    - encoder_name: str, the name of the encoder
    - embed_layer: int, the layer number of the embedding
    - flatten_emb: bool or str, if and how (one of ["max", "mean"]) to flatten the embedding

    Returns:
    - dataset_subfolder: str, the full path for the dataset based subfolder
    - file_name: str, the name of the file with embedding details without file extension
    """
    # path for the subfolder
    dataset_subfolder = os.path.join(
        parent_folder, "/".join(os.path.splitext(dataset_path)[0].split("/")[1:])
    )

    # check and generate the folder
    checkNgen_folder(dataset_subfolder)

    file_name = f"{encoder_name}-layer_{embed_layer}-{flatten_emb}"

    return dataset_subfolder, file_name

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