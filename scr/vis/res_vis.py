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

from scr.vis.vis_utils import BokehSave, save_plt
from scr.params.emb import MODEL_SIZE, MODEL_LAYER, ARCH_TYPE, CHECKPOINT_PERCENT
from scr.params.vis import (
    ORDERED_TASK_LIST,
    TASK_LEGEND_MAP,
    TASK_COLORS,
    TASK_SIMPLE_COLOR_MAP,
    PLOT_EXTS,
    ARCH_STYLE_DICT,
)
from scr.analysis.utils import INIT_DICT, INIT_SIMPLE_LIST, SIMPLE_METRIC_LIST
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

    def _get_best_metric_df_dicts(self) -> list[dict]:

        """
        A method for spliting the df into dict based on the best metric

        'all' means best within emb and onehot
        'emb' means best slicing out emb from all
        'carp' or 'esm' means best given arch
        """

        best_metric_df_dict = {"all": {}, "emb": {}}

        for arch in ARCH_TYPE:
            best_metric_df_dict[arch] = {}

        for m in SIMPLE_METRIC_LIST:

            slice_df = get_bestorlast_metric_df(df=self.prepped_df.copy(), metric=m)

            best_metric_df_dict["all"][m] = slice_df
            best_metric_df_dict["emb"][m] = slice_df[
                slice_df["ablation"] == "emb"
            ].copy()

            for arch in ARCH_TYPE:
                arch_df = get_bestorlast_metric_df(
                    df=self.prepped_df.copy(), metric=m, arch=arch
                )
                best_metric_df_dict[arch][m] = arch_df[
                    arch_df["ablation"] == "emb"
                ].copy()

        return best_metric_df_dict

    def _get_last_metric_df_dict(self) -> list[dict]:

        """
        A method for spliting the df into dict gettin last layer performance
        based on the metric

        'all' means best within emb and onehot
        'emb' means best slicing out emb from all
        'carp' or 'esm' means best given arch
        """

        last_metric_dict = {}

        for m in SIMPLE_METRIC_LIST:

            last_metric_dict[m] = {}

            for arch in ARCH_TYPE:
                last_metric_dict[m][arch] = get_bestorlast_metric_df(
                    df=self.prepped_df.copy(), metric=m, arch=arch, bestorlast="last"
                )

        return last_metric_dict

    def _append_randstat(self, emb_df: pd.DataFrame) -> pd.DataFrame:

        """
        A method for adding rand stat and delta onehot info to
        best performance based on given metric
        """

        # add rand stat
        for ab in INIT_SIMPLE_LIST:
            emb_df[ab] = np.nan
            emb_df[f"{ab} - onehot"] = np.nan

        emb_df["emb - onehot"] = np.nan

        emb_df = emb_df.reset_index(drop=True)

        # get the corresponding stat and rand value at the layer
        for i, row in emb_df.iterrows():

            # Convert the row to a dictionary
            row_dict = row.to_dict()

            # Select certain keys from the dictionary
            selected_keys = ["arch", "task", "metric", "model"]

            # pick the row to match
            row_to_match = {key: row_dict[key] for key in selected_keys}

            # get the onehot baseline
            onehot_val = get_layer_value(
                df=self.prepped_df,
                row_to_match={
                    key: row_dict[key] if i < 3 else "onehot"
                    for i, key in enumerate(selected_keys)
                },
                ablation="onehot",
                layer_numb=0,
            )

            # calc best emb perf del
            emb_df.at[i, "emb - onehot"] = emb_df.at[i, "best_value"] - onehot_val

            for ab in INIT_SIMPLE_LIST:

                randstat_val = get_layer_value(
                    df=self.prepped_df,
                    row_to_match=row_to_match,
                    ablation=ab,
                    layer_numb=row_dict["best_layer"],
                )

                emb_df.at[i, ab] = randstat_val
                emb_df.at[i, f"{ab} - onehot"] = randstat_val - onehot_val

        return emb_df

    def _append_ptp(self, emb_df: pd.DataFrame) -> pd.DataFrame:

        """A method for adding pretrain percent best performance"""

        for ptp in CHECKPOINT_PERCENT:
            emb_df[str(ptp)] = np.nan
            emb_df[f"{str(ptp)} - onehot"] = np.nan

        emb_df = emb_df.reset_index(drop=True)

        # get the corresponding stat and rand value at the layer
        for i, row in emb_df.iterrows():

            # Convert the row to a dictionary
            row_dict = row.to_dict()

            # Select certain keys from the dictionary
            selected_keys = ["arch", "task", "metric", "model"]

            # pick the row to match
            row_to_match = {key: row_dict[key] for key in selected_keys}

            # get the onehot baseline
            onehot_val = get_layer_value(
                df=self.prepped_df,
                row_to_match={
                    key: row_dict[key] if i < 3 else "onehot"
                    for i, key in enumerate(selected_keys)
                },
                ablation="onehot",
                layer_numb=0,
            )

            for ptp in CHECKPOINT_PERCENT:

                statorrand_val = get_layer_value(
                    df=self.prepped_df,
                    row_to_match=row_to_match,
                    ablation="emb",
                    ptp=ptp,
                    layer_numb=row_dict["best_layer"],
                )

                emb_df.at[i, str(ptp)] = statorrand_val
                emb_df.at[i, f"{str(ptp)} - onehot"] = statorrand_val - onehot_val

        return emb_df

    def plot_emb_onehot(self, metric: str = "test_performance_1") -> pd.DataFrame:

        """
        A method for getting df for emb vs onehot

        Args:
        - ifclean_metric: bool = True, simplify test performance
        """

        # now plot and save
        vs_plot = plot_emb_onehot(
            df=self.best_metric_df_dict["all"][metric],
            metric=metric,
            path2folder=checkNgen_folder(
                os.path.join(self._sum_folder, "embvsonetho", metric)
            ),
        )

        # get a bar plot with percent performance achieved
        emb_df = self.best_metric_df_dict["emb"][metric]
        emb_df["model_layer_percent"] = emb_df["best_layer"] / emb_df["model_layer"]

        bar_plot = plot_best_layer_bar(
            df=emb_df,
            metric=metric,
            path2folder=checkNgen_folder(
                os.path.join(self._sum_folder, "bestemblayer", metric)
            ),
        )

        # now add rand stat info to emb_df
        emb_df = self._append_randstat(emb_df=emb_df)

        randstat_plot_dict = {}

        for delta_onehot in [0, 1]:
            randstat_plot_dict[delta_onehot] = {}

            for randstat in INIT_SIMPLE_LIST + [""]:

                randstat_plot_dict[delta_onehot][randstat] = plot_randstat(
                    df=emb_df,
                    metric=metric,
                    randstat=randstat,
                    delta_onehot=delta_onehot,
                    path2folder=checkNgen_folder(
                        os.path.join(self._sum_folder, "randstat", metric)
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

        df = self.prepped_df.copy()

        # get rid of onehot
        df = df[df["ablation"] == "emb"].copy()

        if ifsimple or arch is None:

            slice_df = get_bestorlast_metric_df(df=df, metric=metric)

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
                metric=metric,
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
            delta_plot = plot_layer_delta_det(
                df=slice_df,
                layer_cut=layer_cut,
                arch=arch,
                metric=metric,
                path2folder=checkNgen_folder(
                    os.path.join(self._sum_folder, "layerdelta", metric)
                ),
            )

        return slice_df, delta_plot

    def plot_pretrain_degree(
        self, metric: str = "test_performance_1", arch: str = "carp"
    ):

        """A method for plotting the pretraining arch"""

        for delta_onehot in [0, 1]:
            plot_pretrain_degree(
                emb_df=self._append_ptp(self.best_metric_df_dict[arch][metric]),
                metric=metric,
                arch=arch,
                delta_onehot=delta_onehot,
                path2folder=checkNgen_folder(
                    os.path.join(self._sum_folder, "pretraindegree", metric, arch)
                ),
            )

    def plot_arch_size(
        self,
        metric: str = "test_performance_1",
    ):

        """A method for plotting arch size"""

        for delta_onehot in [0, 1]:
            for arch in ARCH_TYPE + [""]:
                plot_arch_size(
                    arch_df_dict=self.last_metric_df_dict,
                    metric=metric,
                    arch=arch,
                    delta_onehot=delta_onehot,
                    path2folder=checkNgen_folder(
                        os.path.join(self._sum_folder, "archsize", metric, arch)
                    ),
                )

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
    def prepped_df(self) -> pd.DataFrame:

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

    @property
    def best_metric_df_dict(self) -> dict:
        """Return a dict splicing the df with best metric based on best metric"""
        return self._get_best_metric_df_dicts()

    @property
    def last_metric_df_dict(self) -> dict:
        """Return a dict splicin the df with the last layer value given metric"""
        return self._get_last_metric_df_dict()


def simplify_test_metric(metric: str) -> str:

    """
    A function to unify metric for plotting
    """

    for t in ["test_performance_1", "test_performance_2", "test_loss"]:
        metric = metric.replace(t, "test performance")

    return metric


def get_bestorlast_metric_df(
    df: pd.DataFrame,
    metric: str = "test_performance_1",
    arch: str = "",
    bestorlast: str = "best",
) -> pd.DataFrame:

    """
    A function for cleaning up the df to get best layer based on chosen metric
    """

    slice_df = df[(df["metric"] == metric)].copy()

    slice_df = slice_df[
        (slice_df["ablation"] == "onehot") | (slice_df["ablation"] == "emb")
    ]

    # comb carp and esm
    if arch != "":
        slice_df = slice_df[(slice_df["arch"] == arch)].copy()

    if bestorlast == "best":

        if metric == "test_loss":
            # get the max perform layer
            slice_df["best_value"] = slice_df["value"].apply(np.min)
            slice_df["best_layer"] = slice_df["value"].apply(np.argmin)

            # Find the index of the maximum value in 'value_column' for each group
            min_indices = (
                slice_df.groupby(["task", "ablation"])["best_value"].idxmin().dropna()
            )

            # Use loc to select the rows corresponding to the max indices
            slice_df = slice_df.loc[min_indices]

        else:
            # get the max perform layer
            slice_df["best_value"] = slice_df["value"].apply(np.max)
            slice_df["best_layer"] = slice_df["value"].apply(np.argmax)

            # Find the index of the maximum value in 'value_column' for each group
            max_indices = (
                slice_df.groupby(["task", "ablation"])["best_value"].idxmax().dropna()
            )

            # Use loc to select the rows corresponding to the max indices
            slice_df = slice_df.loc[max_indices]

    elif bestorlast == "last":

        # get last layer
        slice_df["last_value"] = slice_df["value"].apply(
            lambda x: x[-1] if len(x) > 0 else None
        )

    else:
        print(f"{bestorlast} is not 'best' or 'last'")

    return slice_df.copy()


def get_layer_value(
    df: pd.DataFrame,
    row_to_match: dict,
    ablation: str,
    layer_numb: int,
    ptp: float = -1,
) -> float:

    """
    A function to get value of a given layer and other specifics

    Args:
    - ptp: float = -1, include ptp only when ptp not default -1
    """

    row_to_match["ablation"] = ablation

    # overwrite default
    if ptp > -1:
        row_to_match["ptp"] = ptp

    # Create a boolean mask for each condition
    conditions = [df[col] == value for col, value in row_to_match.items()]

    # Combine the conditions with AND (use np.all)
    mask = np.all(conditions, axis=0)

    # Use the mask to select the matching row(s)
    matching_rows = df[mask]

    assert len(matching_rows) == 1, f"{matching_rows} len not 1!"

    return matching_rows["value"].to_numpy()[0][int(layer_numb)]


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


def plot_best_layer_bar(
    df: pd.DataFrame, metric: str, path2folder: str = "results/summary/bestemblayer"
):

    """
    A function for plotting a bar plot for
    """

    plot_title = "Best {} achieved at percent depth of pretrain model".format(
        simplify_test_metric(metric)
    )

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

    save_plt(fig, plot_title=plot_title, path2folder=path2folder)

    return fig


def plot_emb_onehot(
    df: pd.DataFrame,
    metric: str,
    path2folder: str = "results/summary/embonehot",
):
    """A function for plotting best emb vs onehot"""

    plot_title = "Best {} embedding against onehot baseline".format(
        simplify_test_metric(metric)
    )

    print(f"Plotting {plot_title}...")

    fig, ax = plt.subplots()
    fig.set_size_inches(6, 6)

    # get the min x or y for the diagnol line
    diag_min = 1

    for (task, c) in TASK_SIMPLE_COLOR_MAP.items():

        sliced_df = df[df["task"] == task]

        x = sliced_df[sliced_df["ablation"] == "onehot"]["best_value"].values
        y = sliced_df[sliced_df["ablation"] == "emb"]["best_value"].values

        if metric != "test_loss":
            min_xy = min(min(y), min(x))
            if min_xy < diag_min:
                diag_min = min_xy

        scatter = ax.scatter(x, y, c=c, s=200, alpha=0.8, label=task, edgecolors="none")

    if metric != "test_loss":
        # diag min to smallest one decimal
        diag_min = math.floor(diag_min * 10) / 10

        # Add a diagonal line
        plt.plot(
            [diag_min, 1],
            [diag_min, 1],
            linestyle=":",
            color="grey",
        )

    ax.add_artist(ax.legend(title="Tasks", bbox_to_anchor=(1, 1.012), loc="upper left"))

    if metric == "test_loss":
        plt.xscale("log")
        plt.yscale("log")

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


def plot_randstat(
    df: pd.DataFrame,
    metric: str,
    randstat: str,
    delta_onehot: bool = True,
    path2folder: str = "results/summary/randstat",
):

    """
    A function for plotting emb vs rand or stat
    or rand vs stat
    """

    if randstat == "":
        comp_det = " vs ".join(list(INIT_DICT.values()))
        plot_title = "Best {} embedding same layer {}".format(
            simplify_test_metric(metric), comp_det
        )
        pathrandstat = "vs"
    else:
        plot_title = "Best {} embedding against same layer {}".format(
            simplify_test_metric(metric), INIT_DICT[randstat]
        )
        pathrandstat = randstat

    print(f"Plotting {plot_title}...")

    fig, ax = plt.subplots()
    fig.set_size_inches(6, 6)

    # get the min x or y for the diagnol line
    diag_min = 1

    if delta_onehot:
        diag_max = 0
        label_append = "- onehot"
        path_append = "onehot"

    else:
        diag_max = 1
        label_append = ""
        path_append = ""

    for (task, c) in TASK_SIMPLE_COLOR_MAP.items():

        task_df = df[df["task"] == task]

        if delta_onehot:
            if randstat == "":
                x = task_df["rand - onehot"].values
                y = task_df["stat - onehot"].values
            else:
                x = task_df["emb - onehot"].values
                y = task_df[f"{randstat} - onehot"].values

            if metric != "test_loss":

                max_xy = max(max(x), max(y))

                if max_xy > diag_max:
                    diag_max = max_xy

        else:
            if randstat == "":
                x = task_df["rand"].values
                y = task_df["stat"].values
            else:
                x = task_df["best_value"].values
                y = task_df[randstat].values

        if metric != "test_loss":

            min_xy = min(min(y), min(x))

            if min_xy < diag_min:
                diag_min = min_xy

            min_xy = min(min(y), min(x))
            if min_xy < diag_min:
                diag_min = min_xy

        scatter = ax.scatter(x, y, c=c, s=200, alpha=0.8, label=task, edgecolors="none")

    if metric != "test_loss":

        # diag min to smallest one decimal and max to largest
        diag_min = math.floor(diag_min * 10) / 10
        diag_max = math.ceil(diag_max * 10) / 10

        # Add a diagonal line
        plt.plot(
            [diag_min, diag_max],
            [diag_min, diag_max],
            linestyle=":",
            color="grey",
        )

    ax.add_artist(ax.legend(title="Tasks", bbox_to_anchor=(1, 1.012), loc="upper left"))

    if randstat == "":
        plt.xlabel(f"Best embedding random init {label_append}")
        plt.ylabel(f"Best embedding stat transfer {label_append}")
    else:
        plt.xlabel(f"Best embedding {label_append}")
        plt.ylabel(f"{INIT_DICT[randstat].capitalize()} {label_append}")

    plt.title(plot_title)

    path2folder = checkNgen_folder(
        os.path.normpath(os.path.join(path2folder, pathrandstat, path_append))
    )

    print(f"Saving to {path2folder}...")

    save_plt(fig, plot_title=plot_title, path2folder=path2folder)

    return fig


def plot_pretrain_degree(
    emb_df: pd.DataFrame,
    metric: str = "test_performance_1",
    arch: str = "carp",
    delta_onehot: bool = True,
    path2folder: str = "results/summary/pretraindegree",
):

    """A method for plotting the pretraining arch"""

    plot_title = "Best {} cross different pretrain degrees of {}".format(
        simplify_test_metric(metric), arch.upper()
    )

    if delta_onehot:
        melt_cols = ["task"] + [str(p) + " - onehot" for p in CHECKPOINT_PERCENT]
        label_append = " - onehot"
        path_append = "onehot"
        y_max = None
    else:
        melt_cols = ["task"] + [str(p) for p in CHECKPOINT_PERCENT]
        label_append = ""
        path_append = ""
        y_max = 1

    x_name = "Pretrain degree"
    y_name = "Best test performance" + label_append

    emb_df_melt = pd.melt(
        emb_df[melt_cols],
        id_vars="task",
        var_name=x_name,
        value_name=y_name,
    )

    # Plot dots with colors corresponding to the category
    fig, ax = plt.subplots()
    fig.set_size_inches(6, 6)

    for category, group in emb_df_melt.groupby("task"):

        ax.plot(
            [float(x.replace(label_append, "")) for x in group[x_name]],
            group[y_name],
            marker="o",
            markersize=12,
            alpha=0.8,
            label=category,
            color=TASK_SIMPLE_COLOR_MAP.get(category, "gray"),
            mec="none",
        )

    ax.set_xticks(CHECKPOINT_PERCENT)
    ax.set_xticklabels([str(tick) for tick in CHECKPOINT_PERCENT])

    ax.set_xlim(0, 1.125)

    if metric != "test_loss":
        ax.set_ylim(None, y_max)

    # Set labels and title
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.set_title(plot_title, pad=10)

    ax.add_artist(ax.legend(title="Tasks", bbox_to_anchor=(1, 1.012), loc="upper left"))

    path2folder = checkNgen_folder(
        os.path.join(os.path.normpath(path2folder), path_append)
    )

    print(f"Saving to {path2folder}...")

    save_plt(fig, plot_title=plot_title, path2folder=path2folder)

    return fig


def plot_arch_size(
    arch_df_dict: dict,
    metric: str = "test_performance_1",
    arch: str = "",
    delta_onehot: bool = True,
    path2folder: str = "results/summary/archsize",
):

    """
    A function for plotting performance vs arch size

    Args:
    - arch: str = "", for both arch combined
    """

    if arch == "":
        arch_name = "architectures"
        arch_list = ARCH_TYPE
    else:
        arch_name = arch.upper()
        arch_list = [arch]

    plot_title = "Last layer {} cross different sizes of pretrain {}".format(
        simplify_test_metric(metric), arch_name
    )

    if delta_onehot:
        label_append = " - onehot"
        path_append = "onehot"
        y_max = None
    else:
        label_append = ""
        path_append = ""
        y_max = 1

    # Plot dots with colors corresponding to the category
    fig, ax = plt.subplots()
    fig.set_size_inches(6, 6)

    for i, a in enumerate(arch_list):

        arch_df = arch_df_dict[metric][a]

        if a == "carp":
            # ignore ptp not 1 or 0
            # Select rows where 'Column2' has a value from the list
            arch_df = arch_df[
                arch_df["ptp"].isin(arch_df_dict[metric]["esm"]["ptp"].unique())
            ]

        # get rid of esm1b
        elif a == "esm":
            arch_df = arch_df[~arch_df["model"].isin(["esm1b_t33_650M_UR50S"])]

        else:
            print(f"{a} not in {ARCH_TYPE}")

        for category, group in arch_df.sort_values(["model_size"]).groupby("task"):

            # to not duplicate label
            if i == 0:
                label = category
            else:
                label = None

            if delta_onehot:

                group = group.reset_index(drop=True).copy()

                # Identify the onehot row
                onehot_row = group[group["ablation"] == "onehot"].index
                onehot_val = group.loc[
                    group["ablation"] == "onehot", "last_value"
                ].iloc[0]

                # subtract the onehot value from other rows
                group["last_value"] -= onehot_val

                # drop the onehot row
                group = group.drop(index=onehot_row)

            ax.plot(
                group["model_size"],
                group["last_value"],
                marker="o",
                markersize=12,
                alpha=0.8,
                label=label,
                color=TASK_SIMPLE_COLOR_MAP.get(category, "gray"),
                **ARCH_STYLE_DICT[a],
            )

    # add additional legend if for both
    if arch == "":
        arch_legend_dict = {}

        for a in ARCH_TYPE:

            if a == "carp":
                mfc = "none"
            else:
                mfc = "gray"

            arch_legend_dict[a] = Line2D(
                [0],
                [0],
                marker="o",
                color="gray",
                linestyle=ARCH_STYLE_DICT[a]["linestyle"],
                label=a.upper(),
                markerfacecolor="k",
                markersize=12,
                mfc=mfc,
            )

        ax.add_artist(
            ax.legend(
                handles=list(arch_legend_dict.values()),
                bbox_to_anchor=(1, 0.49),
                loc="upper left",
                title="Pretrained architectures",
            )
        )

    plt.xscale("log")

    if metric == "test_loss":
        plt.yscale("log")

    if metric != "test_loss":
        # Set y-axis limits
        ax.set_ylim(bottom=None, top=y_max)

    # Set labels and title
    ax.set_xlabel("Log model size (M)")
    ax.set_ylabel(f"Last layer test performance{label_append}")
    ax.set_title(plot_title)

    ax.add_artist(ax.legend(title="Tasks", bbox_to_anchor=(1, 1.012), loc="upper left"))

    path2folder = checkNgen_folder(
        os.path.join(os.path.normpath(path2folder), path_append)
    )

    print(f"Saving to {path2folder}...")

    save_plt(fig, plot_title=plot_title, path2folder=path2folder)

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

    plot_title = "Best {} at x = {}".format(simplify_test_metric(metric), layer_cut)

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

        if metric != "test_loss":
            min_xy = min(min(y), min(x))
            max_xy = max(max(x), max(y))

            if min_xy < diag_min:
                diag_min = min_xy

            if max_xy > diag_max:
                diag_max = max_xy

        scatter = ax.scatter(x, y, c=c, label=task, s=200, alpha=0.8, edgecolors="none")

    if metric != "test_loss":
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

    save_plt(fig, plot_title=plot_title, path2folder=path2folder)

    return fig


def plot_layer_delta_det(
    df: pd.DataFrame,
    layer_cut: int,
    arch: str,
    metric: str,
    path2folder: str = "results/summary/layerdelta",
):
    """A function for plotting and saving layer delta"""

    plot_title = "{} layer {} at x = {}".format(
        arch.upper(), simplify_test_metric(metric), layer_cut
    )

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

    save_plt(fig, plot_title=plot_title, path2folder=path2folder)

    return fig


def plot_layer_delta_hv(
    df: pd.DataFrame,
    layer_cut: int,
    arch: str,
    metric: str,
    path2folder: str = "results/summary",
):
    """A function for plotting and saving layer delta"""

    plot_title = "{} layer {} at x = {}".format(arch.upper(), metric, layer_cut)

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