"""Analyzing per layer output"""

from __future__ import annotations

import os
from glob import glob
import numpy as np

import matplotlib.pyplot as plt

from scr.params.emb import TRANSFORMER_INFO
from scr.utils import pickle_load, get_filename, checkNgen_folder


class LayerLoss:
    """A class for handling layer analysis"""

    def __init__(
        self,
        input_path: str = "results/train_val_test",
        output_path: str = "results/analysis_layer",
        metric_list: list[str] = ["train_mse", "test_ndcg", "test_rho"],
    ):
        """
        Args:
        - input_path: str = "results/train_val_test",
        - output_path: str = "results/analysis_layer"
        - metric_list: list[str] = ["train_mse", "test_ndcg", "test_rho"]
        """
        # get rid of the last "/" if any
        self._input_path = os.path.normpath(input_path)
        # get the list of subfolders for each dataset
        self._dataset_folders = glob(f"{self._input_path}/*/*/*")

        # get rid of the last "/" if any
        self._output_path = os.path.normpath(output_path)
        self._metric_list = metric_list

        # init a dictionary for recording outputs
        self._layer_analysis_dict = {}

        for dataset_folder in self._dataset_folders:
            self._layer_analysis_dict[dataset_folder] = self.parse_result_dicts(
                dataset_folder
            )

    def parse_result_dicts(self, folder_path: str):
        """
        Parse the output result dictionaries for plotting

        Args:
        - folder_path: str, the folder path for the datasets

        Returns:
        - output_numb_dict: dict, metric name as keys and the array of losses as values
        - output_numb_details: dict, details as folder_path, encoder_name, flatten_emb
        """

        # get the list of output pickle files
        pkl_list = glob(f"{folder_path}/*.pkl")

        # get the max layer number for the array
        encoder_name, _, flatten_emb = get_filename(pkl_list[0]).split("-")
        max_layer_numb = TRANSFORMER_INFO[encoder_name][1] + 1

        # init the ouput dict
        output_numb_dict = {
            metric: np.zeros([max_layer_numb]) for metric in self._metric_list
        }

        # loop through the list of the pickle files
        for pkl_file in pkl_list:
            # get the layer number
            layer_numb = int(get_filename(pkl_file).split("-")[1].split("_")[-1])
            # load the result dictionary
            result_dict = pickle_load(pkl_file)

            # populate the processed dictionary
            for metric in self._metric_list:
                subset, kind = metric.split("_")
                if kind == "rho":
                    output_numb_dict[metric][layer_numb] = result_dict[subset][kind][0]
                else:
                    output_numb_dict[metric][layer_numb] = result_dict[subset][kind]

            # get the details for the dataset such as proeng/gb1/two_vs_rest
            task_subflder = os.path.dirname(
                pkl_list[0].split(self._input_path + "/")[-1]
            )
            task, dataset, split = task_subflder.split("/")

            # get some details for plotting and saving
            output_subfolder = checkNgen_folder(
                os.path.join(
                    self._output_path,
                    task_subflder,
                )
            )

        for metric in output_numb_dict.keys():

            plot_name = f"{encoder_name}_{flatten_emb}_{metric}"
            plot_prefix = f"{task}_{dataset}_{split}"

            plt.figure()
            plt.plot(output_numb_dict[metric])
            plt.title(f"{plot_prefix} \n {plot_name}")
            plt.xlabel("layers")
            plt.ylabel("loss")

            for plot_ext in [".svg", ".png"]:
                plt.savefig(
                    os.path.join(output_subfolder, plot_name + plot_ext),
                    bbox_inches="tight",
                )
            plt.close()

        output_numb_details = {
            "folder_path": output_subfolder,
            "encoder_name": encoder_name,
            "flatten_emb": flatten_emb,
        }

        return output_numb_dict, output_numb_details

    @property
    def layer_analysis_dict(self) -> dict:
        """Return a dict with dataset name as the key"""
        return self._layer_analysis_dict