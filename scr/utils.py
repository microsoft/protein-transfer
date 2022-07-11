from __future__ import annotations

from collections import Sequence

import os
from glob import glob

def get_subfolder_name(parent_folder: str|list) -> list(list):
    """
    Return a list of the name of the subfolder in a parent folder
    
    Args:
    - parent_folder: str|list, folder path or a list returned from glob
    """
    if isinstance(parent_folder, str):
        if parent_folder[-1] not in ["*", "/"]:
            parent_folder = parent_folder + "/*"
        elif parent_folder[-1] == "/":
            parent_folder = parent_folder + "*"
        else:
            assert parent_folder[-1] == "*" and parent_folder[-2] == "/"
        subfolder_list = glob(parent_folder)
    else:
        subfolder_list = parent_folder

    return [os.path.basename(subfolder) for subfolder in subfolder_list], subfolder_list