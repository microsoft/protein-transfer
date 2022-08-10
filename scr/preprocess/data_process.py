"""Pre-processing the dataset"""

from __future__ import annotations

from collections import Sequence, defaultdict

import os
from glob import glob
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

from scr.utils import pickle_save, pickle_load, replace_ext
from scr.params.sys import RAND_SEED
from scr.params.emb import TRANSFORMER_INFO, CARP_INFO, MAX_SEQ_LEN
from scr.preprocess.seq_loader import SeqLoader
from scr.encoding.encoding_classes import (
    AbstractEncoder,
    ESMEncoder,
    CARPEncoder,
    OnehotEncoder,
)


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

        """
        Args:
        - parent_seq_path: str, path for the parent sequence
        - csv_path: str, path for the fitness csv file
        """

        # Load the parent sequence from the fasta file
        self._parent_seq = SeqLoader(parent_seq_path=parent_seq_path)

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
            all the subfolders have datasets as the subsubfolders, ie

            {data_folder}/
                proeng/
                    aav/
                        one_vs_many.csv
                        two_vs_many.csv
                        P03135.fasta
                    thermo/
                        mixed.csv
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
        - A dataframe with "task", "dataset", "split",
            "csv_path", "fasta_path", "pkl_path" as columns, ie.
            (proeng, gb1, low_vs_high, data/proeng/gb1/low_vs_high.csv,
            data/proeng/gb1/5LDE_1.fasta)
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

class ProtranDataset(Dataset):

    """A dataset class for processing protein transfer data"""

    def __init__(
        self,
        dataset_path: str,
        subset: str,
        encoder_name: str,
        reset_param: bool = False,
        resample_param: bool = False,
        embed_batch_size: int = 0,
        flatten_emb: bool | str = False,
        embed_path: str = None,
        seq_start_idx: bool | int = False,
        seq_end_idx: bool | int = False,
        if_encode_all: bool = True,
        **encoder_params,
    ):

        """
        Args:
        - dataset_path: str, full path to the dataset, in pkl or panda readable format
            columns include: sequence, target, set, validation,
            mut_name (optional), mut_numb (optional)
        - subset: str, train, val, test
        - encoder_name: str, the name of the encoder
        - reset_param: bool = False, if update the full model to xavier_uniform_
        - resample_param: bool = False, if update the full model to xavier_normal_
        - embed_batch_size: int, set to 0 to encode all in a single batch
        - flatten_emb: bool or str, if and how (one of ["max", "mean"]) to flatten the embedding
        - embed_path: str = None, path to presaved embedding
        - seq_start_idx: bool | int = False, the index for the start of the sequence
        - seq_end_idx: bool | int = False, the index for the end of the sequence
        - encoder_params: kwarg, additional parameters for encoding
        """

        # with additional info mut_name, mut_numb
        if os.path.splitext(dataset_path)[-1] in [".pkl", ".PKL", ""]:
            self._df = pickle_load(dataset_path)
            self._add_mut_info = True
        # without such info
        else:
            self._df = pd.read_csv(dataset_path)
            self._add_mut_info = False

        assert "set" in self._df.columns, f"set is not a column in {dataset_path}"
        assert (
            "validation" in self._df.columns
        ), f"validation is not a column in {dataset_path}"

        self._df_train = self._df.loc[
            (self._df["set"] == "train") & (self._df["validation"] != True)
        ]
        self._df_val = self._df.loc[
            (self._df["set"] == "train") & (self._df["validation"] == True)
        ]
        self._df_test = self._df.loc[(self._df["set"] == "test")]

        self._df_dict = {
            "train": self._df_train,
            "val": self._df_val,
            "test": self._df_test,
        }

        assert subset in list(
            self._df_dict.keys()
        ), "split can only be 'train', 'val', or 'test'"
        self._subset = subset

        self._subdf_len = len(self._df_dict[self._subset])

        # not specified seq start will be from 0
        if seq_start_idx == False:
            self._seq_start_idx = 0
        else:
            self._seq_start_idx = int(seq_start_idx)
        # not specified seq end will be the full sequence length
        if seq_end_idx == False:
            self._seq_end_idx = -1
            self._max_seq_len = self._df.sequence.str.len().max()
        else:
            self._seq_end_idx = int(seq_end_idx)
            self._max_seq_len = self._seq_end_idx - self._seq_start_idx

        # get unencoded string of input sequence
        # will need to convert data type
        self.sequence = self._get_column_value("sequence")

        self.if_encode_all = if_encode_all
        if self.if_encode_all:
            # get the encoder class
            if encoder_name in TRANSFORMER_INFO.keys():
                encoder_class = ESMEncoder
            elif encoder_name in CARP_INFO.keys():
                encoder_class = CARPEncoder
            else:
                encoder_class = OnehotEncoder
                encoder_params["max_seq_len"] = self._max_seq_len

            # get the encoder
            self._encoder = encoder_class(
                encoder_name=encoder_name,
                reset_param=reset_param,
                resample_param=resample_param,
                **encoder_params,
            )
            self._total_emb_layer = self._encoder.total_emb_layer

            # check if pregenerated embedding
            if embed_path is not None:
                print(f"Loading pregenerated embeddings from {embed_path}")
                encoded_dict = pickle_load(embed_path)

            # encode the sequences without the mut_name
            else:
                # init an empty dict with empty list to append emb
                encoded_dict = defaultdict(list)

                # use the encoder generator for batch emb
                # assume no labels included
                for encoded_batch_dict in self._encoder.encode(
                    mut_seqs=self.sequence,
                    batch_size=embed_batch_size,
                    flatten_emb=flatten_emb,
                ):

                    for layer, emb in encoded_batch_dict.items():
                        encoded_dict[layer].append(emb)

            # assign each layer as its own variable
            for layer, emb in encoded_dict.items():
                setattr(
                    self,
                    "layer" + str(layer),
                    np.vstack(emb)
                    # torch.tensor(np.vstack(emb), dtype=torch.float32),
                )

        # get and format the fitness or secondary structure values
        # can be numbers or string
        # will need to convert data type
        # make 1D tensor 2D
        self.y = np.expand_dims(self._get_column_value("target"), 1)

        # add mut_name and mut_numb for relevant proeng datasets
        if self._add_mut_info:
            self.mut_name = self._get_column_value("mut_name")
            self.mut_numb = self._get_column_value("mut_numb")
        else:
            self.mut_name = [""] * self._subdf_len
            self.mut_numb = [np.nan] * self._subdf_len

    def __len__(self):
        """Return the length of the selected subset of the dataframe"""
        return self._subdf_len

    def __getitem__(self, idx: int):

        """
        Return the item in the order of
        target (y), sequence, mut_name (optional), mut_numb (optional),
        embedding per layer upto the max number of layer for the encoder

        Args:
        - idx: int
        """
        if self.if_encode_all:
            return (
                self.y[idx],
                self.sequence[idx],
                self.mut_name[idx],
                self.mut_numb[idx],
                *(
                    getattr(self, "layer" + str(layer))[idx]
                    for layer in range(self._total_emb_layer)
                ),
            )
        else:
            return (
                self.y[idx],
                self.sequence[idx],
                self.mut_name[idx],
                self.mut_numb[idx],
            )


    def _get_column_value(self, column_name: str) -> np.ndarray:
        """
        Check and return the column values of the selected dataframe subset

        Args:
        - column_name: str, the name of the dataframe column
        """
        if column_name in self._df.columns:
            if column_name == "sequence":
                return (
                    self._df_dict[self._subset]["sequence"]
                    .astype(str)
                    .str[self._seq_start_idx : self._seq_end_idx]
                    .apply(
                        lambda x: x[: int(MAX_SEQ_LEN // 2)]
                        + x[-int(MAX_SEQ_LEN // 2) :]
                        if len(x) > MAX_SEQ_LEN
                        else x
                    )
                    .values
                )
            else:
                return self._df_dict[self._subset][column_name].values

    @property
    def df_full(self) -> pd.DataFrame:
        """Return the full loaded dataset"""
        return self._df

    @property
    def df_train(self) -> pd.DataFrame:
        """Return the dataset for training only"""
        return self._df_train

    @property
    def df_val(self) -> pd.DataFrame:
        """Return the dataset for validation only"""
        return self._df_val

    @property
    def df_test(self) -> pd.DataFrame:
        """Return the dataset for training only"""
        return self._df_test

    @property
    def max_seq_len(self) -> int:
        """Longest sequence length"""
        return self._max_seq_len


def split_protrain_loader(
    dataset_path: str,
    encoder_name: str,
    reset_param: bool = False,
    resample_param: bool = False,
    embed_batch_size: int = 128,
    flatten_emb: bool | str = False,
    embed_path: str | None = None,
    seq_start_idx: bool | int = False,
    seq_end_idx: bool | int = False,
    subset_list: list[str] = ["train", "val", "test"],
    loader_batch_size: int = 64,
    worker_seed: int = RAND_SEED,
    if_encode_all: bool = True,
    **encoder_params,
):

    """
    A function encode and load the data from a path

    Args:
    - dataset_path: str, full path to the dataset, in pkl or panda readable format
        columns include: sequence, target, set, validation,
        mut_name (optional), mut_numb (optional)
    - encoder_name: str, the name of the encoder
    - reset_param: bool = False, if update the full model to xavier_uniform_
    - resample_param: bool = False, if update the full model to xavier_normal_
    - embed_batch_size: int, set to 0 to encode all in a single batch
    - flatten_emb: bool or str, if and how (one of ["max", "mean"]) to flatten the embedding
    - embed_path: str = None, path to presaved embedding
    - seq_start_idx: bool | int = False, the index for the start of the sequence
    - seq_end_idx: bool | int = False, the index for the end of the sequence
    - subset_list: list of str, train, val, test
    - loader_batch_size: int, the batch size for train, val, and test dataloader
    - worker_seed: int, the seed for dataloader
    - encoder_params: kwarg, additional parameters for encoding
    """

    assert set(subset_list) <= set(
        ["train", "val", "test"]
    ), "subset_list can only contain terms with in be 'train', 'val', or 'test'"

    # specify no shuffling for validation and test
    if_shuffle_list = [True if subset == "train" else False for subset in subset_list]

    return (
        DataLoader(
            dataset=ProtranDataset(
                dataset_path=dataset_path,
                subset=subset,
                encoder_name=encoder_name,
                reset_param=reset_param,
                resample_param=resample_param,
                embed_batch_size=embed_batch_size,
                flatten_emb=flatten_emb,
                embed_path=embed_path,
                seq_start_idx=seq_start_idx,
                seq_end_idx=seq_end_idx,
                if_encode_all=if_encode_all,
                **encoder_params,
            ),
            batch_size=loader_batch_size,
            shuffle=if_shuffle,
            worker_init_fn=worker_seed,
        )
        for subset, if_shuffle in zip(subset_list, if_shuffle_list)
    )