# a script for vis results from summary data df
from __future__ import annotations

import ast

import os

import math

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import seaborn as sns
import holoviews as hv
from holoviews import dim

hv.extension("bokeh")

from scr.vis.vis_utils import BokehSave
from scr.params.emb import MODEL_SIZE, MODEL_LAYER, ARCH_TYPE
from scr.params.vis import (
    ORDERED_TASK_LIST,
    TASK_LEGEND_MAP,
    TASK_COLORS,
    TASK_SIMPLE_COLOR_MAP,
    PLOT_EXTS,
)
from scr.utils import checkNgen_folder


class PlotResultScatter:
    """
    A class handling plotting results in scatter plots, including:
    - PlotLayerDelta
    - PlotEmbvsOnehot
    """

    def __init__(
        self,
        sum_folder: str = "results/summary",
        sum_df_name: str = "all_results",
    ) -> None:

        self._sum_folder = checkNgen_folder(os.path.normpath(sum_folder))
        self._sum_df_name = sum_df_name

    def plot_emb_onehot(self, metric: str = "test_performance_1") -> pd.DataFrame:

        """
        A method for getting df for emb vs onehot

        Args:
        - ifclean_metric: bool = True, simplify test performance
        """

        df = self.prep_df.copy()

        slice_df = get_best_metric_df(df=df, metric=metric)

        # now plot and save
        vs_plot = plot_emb_onehot(
            df=slice_df,
            metric=metric.replace("test_performance_1", "test performance").replace(
                "test_performance_2", "test performance"
            ),
            path2folder=checkNgen_folder(
                os.path.join(self._sum_folder, "embvsonetho", metric)
            ),
        )

        # get a bar plot with percent performance achieved
        emb_df = slice_df[slice_df["ablation"] == "emb"].copy()
        emb_df["model_layer_percent"] = (
            emb_df["best_layer"] / emb_df["model_layer"]
        )

        bar_plot = plot_best_layer_bar(
            df=emb_df,
            metric=metric.replace("test_performance_1", "test performance").replace(
                "test_performance_2", "test performance"
            ),
            path2folder=checkNgen_folder(
                os.path.join(self._sum_folder, "bestemblayer", metric)
            ),
        )

        return vs_plot, bar_plot

    def plot_layer_delta(
        self,
        layer_cut: int,
        metric: str = "test_performance_1",
        arch: str = "esm",
        ifsimple: bool = True,
    ) -> pd.DataFrame:

        """
        A method for getting the sliced dataframe

        Add per-train degree as a col

        Args:
        - ifsimple: bool = True, if selecting the best
        """

        assert arch in ARCH_TYPE, f"{arch} not in {ARCH_TYPE}"

        df = self.prep_df.copy()

        # get rid of onehot
        df = df[df["ablation"] == "emb"].copy()

        if ifsimple or arch is None:

            slice_df = get_best_metric_df(df=df, metric=metric)

            # Apply the function and generate two new columns
            slice_df["x-0"], slice_df["f-x"] = zip(
                *slice_df.apply(
                    lambda row: delta_layer(
                        layer_cut=layer_cut, value_array=row["value"]
                    ),
                    axis=1,
                )
            )

            # now plot and save
            delta_plot = plot_layer_delta_simple(
                df=slice_df,
                layer_cut=layer_cut,
                metric=metric.replace("test_performance_1", "test performance").replace(
                    "test_performance_2", "test performance"
                ),
                path2folder=checkNgen_folder(
                    os.path.join(self._sum_folder, "layerdelta_simple", metric)
                ),
            )

            return slice_df, delta_plot

        else:

            slice_df = df[(df["metric"] == metric) & (df["arch"] == arch)].copy()

            # Apply the function and generate two new columns
            slice_df["x-0"], slice_df["f-x"] = zip(
                *slice_df.apply(
                    lambda row: delta_layer(
                        layer_cut=layer_cut, value_array=row["value"]
                    ),
                    axis=1,
                )
            )

            # now plot and save
            delta_plot = plot_layer_delta_plt(
                df=slice_df,
                layer_cut=layer_cut,
                arch=arch,
                metric=metric.replace("test_performance_1", "test performance").replace(
                    "test_performance_2", "test performance"
                ),
                path2folder=checkNgen_folder(
                    os.path.join(self._sum_folder, "layerdelta", metric)
                ),
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

    @property
    def prep_df(self) -> pd.DataFrame:
        """Return a more plotting compatible df"""

        prepped_df = self.result_df.copy()

        # add task type and model size details for plotting legends
        prepped_df["task_type"] = prepped_df["task"].str.split("_").str[0]
        prepped_df["model_size"] = prepped_df["model"].map(MODEL_SIZE)
        prepped_df["model_layer"] = prepped_df["model"].map(MODEL_LAYER)

        # get rid of pooling details
        prepped_df["task"] = prepped_df["task"].str.replace("_mean", "")
        prepped_df["task"] = prepped_df["task"].str.replace("_noflatten", "")

        # sort based on given task order for plot legend
        prepped_df["task"] = pd.Categorical(
            prepped_df["task"], categories=ORDERED_TASK_LIST, ordered=True
        ).map(TASK_LEGEND_MAP)
        prepped_df = prepped_df.sort_values(["task", "ptp"], ascending=[True, False])

        return prepped_df


def get_best_metric_df(
    df: pd.DataFrame, metric: str = "test_performance_1"
) -> pd.DataFrame:

    """
    A function for cleaning up the df to get best layer based on chosen metric
    """

    slice_df = df[(df["metric"] == metric)].copy()

    slice_df = slice_df[
        (slice_df["ablation"] == "onehot") | (slice_df["ablation"] == "emb")
    ]

    # get the max perform layer and the layer number
    slice_df["best_value"] = slice_df["value"].apply(np.max)
    slice_df["best_layer"] = slice_df["value"].apply(np.argmax)

    # Find the index of the maximum value in 'value_column' for each group
    max_indices = slice_df.groupby(["task", "ablation"])["best_value"].idxmax().dropna()

    # Use loc to select the rows corresponding to the max indices
    slice_df = slice_df.loc[max_indices]

    return slice_df.copy()


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


def plot_best_layer_bar(df: pd.DataFrame, metric: str, path2folder: str):

    """
    A function for plotting a bar plot for
    """

    plot_title = f"Best {metric} achieved at percent depth of pretrain model"

    print(f"Plotting {plot_title}...")

    fig, ax = plt.subplots()

    df.plot(
        kind="bar",
        x="task",
        y="model_layer_percent",
        color=[TASK_SIMPLE_COLOR_MAP.get(task, "gray") for task in df["task"]],
        ax=ax,
        legend=None,
    )

    ax.set_ylim(0, 1)

    # set labels and title
    plt.xlabel("Task")
    plt.ylabel("Percent")
    plt.title(plot_title, pad=10)

    path2folder = os.path.normpath(path2folder)

    print(f"Saving to {path2folder}...")

    for ext in PLOT_EXTS:
        plot_title_no_space = plot_title.replace(" ", "_")
        plt.savefig(
            os.path.join(path2folder, f"{plot_title_no_space}{ext}"),
            bbox_inches="tight",
        )

    return fig


def plot_emb_onehot(
    df: pd.DataFrame,
    metric: str,
    path2folder: str = "results/summary/embonehot",
):
    """A function for plotting best emb vs onehot"""

    plot_title = f"Best {metric} embedding against onehot baseline"

    print(f"Plotting {plot_title}...")

    fig, ax = plt.subplots()
    fig.set_size_inches(6, 6)

    # get the min x or y for the diagnol line
    diag_min = 1

    for (task, c) in TASK_SIMPLE_COLOR_MAP.items():
        sliced_df = df[df["task"] == task]
        emb_df = sliced_df[sliced_df["ablation"] == "emb"]
        onehot_df = sliced_df[sliced_df["ablation"] == "onehot"]

        y = emb_df["best_value"].values
        # note their is only one onehot for all embeddings for each tast
        x = onehot_df["best_value"].values
        x = np.repeat(x, len(y))

        min_xy = min(min(y), min(x))
        if min_xy < diag_min:
            diag_min = min_xy

        c = c

        scatter = ax.scatter(x, y, c=c, s=200, alpha=0.8, label=task, edgecolors="none")

    # diag min to smallest one decimal
    diag_min = math.floor(diag_min * 10) / 10

    # Add a diagonal line
    plt.plot(
        [diag_min, 1],
        [diag_min, 1],
        linestyle=":",
        color="grey",
    )

    legend1 = ax.legend(title="Tasks", bbox_to_anchor=(1, 1.012), loc="upper left")
    ax.add_artist(legend1)

    plt.ylabel("Best embedding test performance")
    plt.xlabel("Onehot")
    plt.title(plot_title)

    path2folder = os.path.normpath(path2folder)

    print(f"Saving to {path2folder}...")

    for ext in PLOT_EXTS:
        plot_title_no_space = plot_title.replace(" ", "_")
        plt.savefig(
            os.path.join(path2folder, f"{plot_title_no_space}{ext}"),
            bbox_inches="tight",
        )

    return fig


def plot_layer_delta_simple(
    df: pd.DataFrame,
    layer_cut: int,
    metric: str,
    path2folder: str = "results/summary/layerdelta_simple",
):
    """
    A function for plotting and saving layer delta
    after selecting the best performance based on given metric
    """

    plot_title = f"Best {metric} at x = {layer_cut}"

    print(f"Plotting {plot_title}...")

    fig, ax = plt.subplots()
    fig.set_size_inches(6, 6)

    # get the min x or y for the diagnol line
    diag_min = 1
    diag_max = 0

    for (task, c) in TASK_SIMPLE_COLOR_MAP.items():
        sliced_df = df[df["task"] == task]
        x = sliced_df["x-0"].values
        y = sliced_df["f-x"].values

        min_xy = min(min(y), min(x))
        max_xy = max(max(x), max(y))

        if min_xy < diag_min:
            diag_min = min_xy

        if max_xy > diag_max:
            diag_max = max_xy

        scatter = ax.scatter(x, y, c=c, label=task, s=200, alpha=0.8, edgecolors="none")

    # diag min to smallest one decimal
    diag_min = math.floor(diag_min * 10) / 10
    diag_max = math.ceil(diag_max * 10) / 10

    # Add a diagonal line
    plt.plot(
        [diag_min, diag_max],
        [diag_min, diag_max],
        linestyle=":",
        color="grey",
    )

    # add colored task legend
    ax.add_artist(ax.legend(title="Tasks", bbox_to_anchor=(1, 1.012), loc="upper left"))

    plt.xlabel("x-0")
    plt.ylabel("f-x")
    plt.title(plot_title)

    path2folder = os.path.normpath(path2folder)

    print(f"Saving to {path2folder}...")

    for ext in PLOT_EXTS:
        plot_title_no_space = plot_title.replace(" ", "_")
        plt.savefig(
            os.path.join(path2folder, f"{plot_title_no_space}{ext}"),
            bbox_inches="tight",
        )

    return fig


def plot_layer_delta_plt(
    df: pd.DataFrame,
    layer_cut: int,
    arch: str,
    metric: str,
    path2folder: str = "results/summary/layerdelta",
):
    """A function for plotting and saving layer delta"""

    plot_title = f"{arch.upper()} layer {metric} at x = {layer_cut}"

    print(f"Plotting {plot_title}...")

    if arch == "esm":
        alaph_values = [0.8]
        alpha_unique = alaph_values
        alpha_label = ["1"]
        size_legend_label = list(MODEL_SIZE.keys())[:4]
    else:
        alaph_values = df["ptp"].values
        alpha_unique = list(df["ptp"].unique())
        alpha_label = [str(a) for a in alpha_unique]
        size_legend_label = list(MODEL_SIZE.keys())[-4:]

    fig, ax = plt.subplots()
    fig.set_size_inches(10, 7.5)

    for (task, c) in TASK_SIMPLE_COLOR_MAP.items():
        sliced_df = df[df["task"] == task]
        x = sliced_df["x-0"].values
        y = sliced_df["f-x"].values
        s = np.log(sliced_df["model_size"].values + 1) * 18
        scatter = ax.scatter(
            x, y, c=c, s=s, label=task, alpha=alaph_values, edgecolors="none"
        )

    # add colored task legend
    ax.add_artist(ax.legend(title="Tasks", bbox_to_anchor=(1, 1.012), loc="upper left"))

    # add size legend
    handles, labels = scatter.legend_elements(prop="sizes", color="k", alpha=0.8)
    legend2 = ax.legend(
        handles,
        size_legend_label,
        bbox_to_anchor=(1, 0.5925),
        loc="upper left",
        title="Model sizes",
    )
    ax.add_artist(legend2)

    # add alpha legend
    alpha_legend = [None] * len(alpha_unique)

    for i, (a_value, a_lable) in enumerate(zip(alpha_unique, alpha_label)):
        alpha_legend[i] = Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=a_lable,
            alpha=a_value,
            markerfacecolor="k",
            markersize=10,
        )

    ax.add_artist(
        ax.legend(
            handles=alpha_legend,
            bbox_to_anchor=(1, 0.39),
            loc="upper left",
            title="Pretraining degree",
        )
    )

    plt.xlabel("x-0")
    plt.ylabel("f-x")
    plt.title(plot_title)

    path2folder = os.path.normpath(path2folder)

    print(f"Saving to {path2folder}...")

    for ext in PLOT_EXTS:
        plot_title_no_space = plot_title.replace(" ", "_")
        plt.savefig(
            os.path.join(path2folder, f"{plot_title_no_space}{ext}"),
            bbox_inches="tight",
        )

    return fig


def plot_layer_delta_hv(
    df: pd.DataFrame,
    layer_cut: int,
    arch: str,
    metric: str,
    path2folder: str = "results/summary",
):
    """A function for plotting and saving layer delta"""

    plot_title = f"{arch.upper()} layer {metric} at x = {layer_cut}"

    print(f"Plotting {plot_title}...")

    if arch == "esm":
        alpha = 0.8
    else:
        alpha = "ptp"

    delta_scatter = hv.render(
        hv.Scatter(df, kdims=["x-0"], vdims=["f-x", "task", "model_size", "ptp"],).opts(
            color="task",
            cmap={
                l: c
                for l, c in zip(
                    list(TASK_LEGEND_MAP.values()),
                    TASK_COLORS,
                )
            },
            alpha=alpha,
            line_width=2,
            width=800,
            height=400,
            legend_position="right",
            legend_offset=(1, 0),
            size=np.log(dim("model_size") + 1) * 1.5,
            title=plot_title,
        )
    )

    # turn off legend box line
    delta_scatter.legend.border_line_alpha = 0

    print(f"Saving to {path2folder}...")

    BokehSave(
        bokeh_plot=delta_scatter,
        path2folder=path2folder,
        plot_name=plot_title,
        plot_width=800,
    )