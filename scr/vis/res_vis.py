# a script for vis results from summary data df
from __future__ import annotations

import ast

import os

import numpy as np
import pandas as pd

import seaborn as sns
import holoviews as hv
from holoviews import dim

hv.extension("bokeh")

from scr.vis.vis_utils import BokehSave
from scr.params.emb import MODEL_SIZE
from scr.params.vis import ORDERED_TASK_LIST, TASK_LEGEND_MAP

class PlotLayerDelta:
    """
    A class for plotting performance at layer cut x-0 vs f-x
    """

    def __init__(
        self,
        sum_folder: str = "results/summary",
        sum_df_name: str = "all_results",
    ) -> None:

        self._sum_folder = os.path.normpath(sum_folder)
        self._sum_df_name = sum_df_name

    def plot_sub_df(
        self,
        layer_cut: int,
        metric: str = "test_performance_1",
        ablation: str = "emb",
        arch: str = "esm",
    ) -> pd.DataFrame:

        """
        A method for getting the sliced dataframe

        Add per-train degree as a col
        """

        slice_df = self.result_df[
            (self.result_df["metric"] == metric)
            & (self.result_df["ablation"] == ablation)
            & (self.result_df["arch"] == arch)
        ].copy()

        # Apply the function and generate two new columns
        slice_df["x-0"], slice_df["f-x"] = zip(
            *slice_df.apply(
                lambda row: delta_layer(layer_cut=layer_cut, value_array=row["value"]),
                axis=1,
            )
        )
        slice_df["task_type"] = slice_df["task"].str.split("_").str[0]
        slice_df["model_size"] = slice_df["model"].map(MODEL_SIZE)

        # sort based on given task order for plot legend
        slice_df["task"] = pd.Categorical(
            slice_df["task"], categories=ORDERED_TASK_LIST, ordered=True
        ).map(TASK_LEGEND_MAP)
        slice_df = slice_df.sort_values(["task", "ptp"], ascending=[True, False])

        # add pre-train degree to that

        # now plot and save
        delta_plot = plot_layer_delta(
            df=slice_df,
            layer_cut=layer_cut,
            arch=arch,
            metric=metric,
            path2folder=self._sum_folder,
        )

        return slice_df, delta_plot

    @property
    def result_df_path(self) -> str:
        """Return full summary result csv path"""
        df_path = os.path.join(
            os.path.normpath(self._sum_folder), self._sum_df_name + ".csv"
        )

        assert os.path.exists(df_path), f"{df_path} does not exist"

        return df_path

    @property
    def result_df(self) -> pd.DataFrame:
        """Return full result df with value cleaned up"""

        result_df = pd.read_csv(self.result_df_path)

        # check column name existance
        for c in ["metric", "ablation", "arch", "value", "task", "model", "ptp"]:
            assert c in result_df.columns, f"{c} not in df from {self.result_df_path}"

        # Convert the string of lists to NumPy arrays
        result_df["value"] = result_df["value"].apply(ast.literal_eval).apply(np.array)

        # make ptp float
        result_df["ptp"] = result_df["ptp"].astype(float)

        return result_df


def delta_layer(layer_cut: int, value_array: np.array) -> np.array:
    """
    A function return the difference between a given layer performance
    to 0th and the last layer

    Args:
    - layer_cut: int, the layer whose performance will be compared
    - value_array: np.array, the array of all layer performances

    Returns:
    - np.arrary, the performance difference between
        [layer_cut - layer0, final_layer - layer_cut]
    """

    last_layer_numb = len(value_array)

    assert (
        0 < layer_cut < last_layer_numb
    ), f"{layer_cut} not in between 0 and {last_layer_numb}"

    layer_perf = value_array[layer_cut]

    return np.array([layer_perf - value_array[0], value_array[-1] - layer_perf])


def plot_layer_delta(
    df: pd.DataFrame,
    layer_cut: int,
    arch: str,
    metric: str,
    path2folder: str = "results/summary",
):
    """A function for plotting and saving layer delta"""

    plot_title = f"{arch.upper()} layer {metric} at x = {layer_cut}"

    if arch == "esm":
        alpha = 0.8
    else:
        alpha = "ptp"
    
    delta_scatter = hv.render(
        hv.Scatter(
            df,
            kdims=["x-0"],
            vdims=["f-x", "task", "model_size", "ptp"],
        ).opts(
            color="task",
            cmap={
                l: c
                for l, c in zip(
                    list(TASK_LEGEND_MAP.values()),
                    list(
                        sns.color_palette(
                            "blend:#EDA,#7AB", len(ORDERED_TASK_LIST)
                        ).as_hex()
                    ),
                )
            },
            alpha=alpha,
            line_width=2,
            width=800,
            height=400,
            legend_position="right",
            legend_offset=(1, 0),
            size=np.log(dim("model_size")+1) * 1.5,
            title=plot_title,
        )
    )  # * hv.Curve([[0, 0], [0.15, 0.15]]).opts(line_dash="dotted", color="gray")

    # turn off legend box line
    delta_scatter.legend.border_line_alpha = 0

    BokehSave(
        bokeh_plot=delta_scatter,
        path2folder=path2folder,
        plot_name=plot_title,
        # plot_exts=PLOT_EXTS,
        # plot_height = 400,
        plot_width = 800,
        # axis_font_size = "10pt",
        # title_font_size = "10pt",
        # x_name = "x-0",
        # y_name = "f-x",
        # gridoff = True,
        # showplot = True
    )