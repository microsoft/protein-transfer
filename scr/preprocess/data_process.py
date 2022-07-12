"""Pre-processing the dataset"""

from __future__ import annotations

from collections import Sequence

import os
from glob import glob
import pandas as pd

from Bio import SeqIO

from scr.utils import pickle_save, replace_ext


def load_parent_seq(parent_seq_path: str) -> str:
    """
    Load parent fasta sequence

    Args:
    - parent_seq_path: str, the fasta path for the parent sequence,
        with only one sequence per file

    Returns:
    - str, the full parent sequence in the fasta file
    """

    with open(parent_seq_path) as handle:
        recs = []
        for rec in SeqIO.parse(handle, "fasta"):
            recs.append(rec)

    # Confirm that there is only one seq file
    assert len(recs) == 1, "More than 1 sequence in the seq file"
    return str(recs[0].seq)


def get_mut_name(mut_seq: str, parent_seq: str) -> str:
    """
    A function for returning the mutant name

    Args:
    - mut_seq: str, the full mutant sequence
    - parent_seq: str, the full parent sequence

    Returns:
    - str, parent, indel, or mutant name in the format of
        ParentAAMutLocMutAA:ParentAAMutLocMutAA:..., ie. W39W:D40G:G41C:V54Q
    """

    mut_list = []
    if parent_seq == mut_seq:
        return "parent"
    elif len(parent_seq) == len(mut_seq):
        for i, (p, m) in enumerate(zip(list(parent_seq), list(mut_seq))):
            if p != m:
                mut_list.append(f"{p}{i+1}{m}")
        return ":".join(mut_list)
    else:
        return "indel"


class AddMutInfo:
    """A class for appending mutation info for mainly protein engineering tasks"""

    def __init__(self, parent_seq_path: str, csv_path: str):

        # Load the parent sequence from the fasta file
        self._parent_seq = load_parent_seq(parent_seq_path=parent_seq_path)

        # load the dataframe
        self._init_df = pd.read_csv(csv_path)

        self._df = self._init_df.copy()
        # add a column with the mutant names
        self._df["mut_name"] = self._init_df["sequence"].apply(
            get_mut_name, parent_seq=self._parent_seq
        )
        # add a column with the number of mutations
        self._df["mut_numb"] = (
            self._df["mut_name"].str.split(":").map(len, na_action="ignore")
        )

        # get the pickle file path
        self._pkl_path = replace_ext(input_path=csv_path, ext=".pkl")

        pickle_save(what2save=self._df, where2save=self._pkl_path)

    @property
    def parent_seq(self) -> str:
        """Return the parent sequence"""
        return self._parent_seq

    @property
    def pkl_path(self) -> str:
        """Return the pkl file path for the processed dataframe"""
        return self._pkl_path

    @property
    def df(self) -> pd.DataFrame:
        """Return the processed dataframe"""
        return self._df


class TaskProcess:
    """A class for handling different downstream tasks"""

    def __init__(self, data_folder: str = "data/"):
        """
        Args:
        - data_folder: str, a folder path with all the tasks as subfolders where
            all the subfolders have datasets as the subsubfolders
        """

        if data_folder[-1] == "/":
            self._data_folder = data_folder
        else:
            self._data_folder = data_folder + "/"

        # sumamarize all files i nthe data folder
        self._sum_file_df = self.sum_files()

    def sum_files(self) -> pd.DataFrame:
        """
        Summarize all files in the data folder

        Returns:
        - A dataframe with "task", "dataset", "split", "csv_path", "fasta_path", "pkl_path" as columns, ie.
            (proeng, gb1, low_vs_high, data/proeng/gb1/low_vs_high.csv, data/proeng/gb1/5LDE_1.fasta)
            note that csv_path is the list of lmdb files for the structure task
        """
        dataset_folders = glob(f"{self._data_folder}*/*")
        # need a list of tuples in the order of:
        # (task, dataset, split, csv_path, fasta_path)
        list_for_df = []
        for dataset_folder in dataset_folders:
            _, task, dataset = dataset_folder.split("/")
            if task == "structure":
                structure_file_list = [
                    file_path
                    for file_path in glob(f"{dataset_folder}/*.*")
                    if os.path.basename(os.path.splitext(file_path)[0]).split("_")[-1]
                    in ["train", "valid", "cb513"]
                ]
                list_for_df.append(
                    tuple([task, dataset, "cb513", structure_file_list, "", ""])
                )
            else:
                csv_paths = glob(f"{dataset_folder}/*.csv")
                fasta_paths = glob(f"{dataset_folder}/*.fasta")
                pkl_paths = glob(f"{dataset_folder}/*.pkl")

                assert len(csv_paths) >= 1, "Less than one csv"
                assert len(fasta_paths) <= 1, "More than one fasta"

                for csv_path in csv_paths:
                    # if parent seq fasta exists
                    if len(fasta_paths) == 1:
                        fasta_path = fasta_paths[0]

                        # if no existing pkl file, generate and save
                        if len(pkl_paths) == 0:
                            print(f"Adding mutation info to {csv_path}...")
                            pkl_path = AddMutInfo(
                                parent_seq_path=fasta_path, csv_path=csv_path
                            ).pkl_path
                        # pkl file exits
                        else:
                            pkl_path = replace_ext(input_path=csv_path, ext=".pkl")
                    # no parent fasta no pkl file
                    else:
                        fasta_path = ""
                        pkl_path = ""

                    list_for_df.append(
                        tuple(
                            [
                                task,
                                dataset,
                                os.path.basename(os.path.splitext(csv_path)[0]),
                                csv_path,
                                fasta_path,
                                pkl_path,
                            ]
                        )
                    )

        return pd.DataFrame(
            list_for_df,
            columns=["task", "dataset", "split", "csv_path", "fasta_path", "pkl_path"],
        )

    @property
    def sum_file_df(self) -> pd.DataFrame:
        """A summary table for all files in the data folder"""
        return self._sum_file_df