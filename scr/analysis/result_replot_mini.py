"""A tempo clean up for summary plot"""

# a script for vis results from summary data df
from __future__ import annotations

import ast

import os
import itertools

import math

import numpy as np
import pandas as pd

from scipy.stats import spearmanr

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec

import seaborn as sns
import holoviews as hv
from holoviews import dim

hv.extension("bokeh")

from scr.vis.vis_utils import BokehSave, save_plt
from scr.params.emb import (
    MODEL_SIZE,
    EMB_MODEL_SIZE,
    EMB4TASK,
    MODEL_LAYER,
    EMB_MODEL_LAYER,
    CARP_MODEL_LAYER,
    CARP_INFO,
    ARCH_TYPE,
    ARCH_BAR_LAYER,
    ARCH_AB,
    ARCH_AB_DICT,
    CHECKPOINT_PERCENT,
    CARP_CHECKPOINT_LOSSES,
    EMB_SIMPLE_MAP,
    EMB_SIZE_SIMPLE,
    BASELINE_NAME_DICT,
    EMB_SIZE_NAME_SIMPLE
)
from scr.params.vis import (
    ORDERED_TASK_LIST,
    TASK_LEGEND_MAP,
    TASK_COLORS,
    TASK_SIMPLE_COLOR_MAP,
    PLOT_EXTS,
    CARP_ALPHA,
    ARCH_LINE_STYLE_DICT,
    ARCH_DOT_STYLE_DICT,
    ARCH_AB_DOT_STYLE_DICT,
    ARCH_SCATTER_STYLE_DICT,
    LAYER_ALPHAS,
)
from scr.analysis.utils import INIT_DICT, INIT_SIMPLE_LIST, SIMPLE_METRIC_LIST
from scr.utils import checkNgen_folder


task_cluster = {
    "well-aligned": ["SS3 - CB513", "SS3 - TS115", "SS3 - CASP12"],
    "benefit": ["Thermostability", "GB1 - low vs high", "AAV - two vs many", "AAV - one vs many", "Subcellular localization"],
    "nobenefit": ["GB1 - sampled", "GB1 - two vs rest"],
}

ASELINE_NAME_DICT = {
        "onehot": "One-hot",
        "rand": "Random init",
        "stat": "Stat transfer",
}

selected_emb = {"esm": "esm1b_t33_650M_UR50S", "carp": "carp_640M"}

# if for main text filter again for model in EMB4TASK
concise = True
large_esm = "esm1b_t33_650M_UR50S"

mild_3 = ["#f1d384", "#9dae88", "#a3bfd6"]  # lighter yellow green blue s m l
bright_3 = ["#F9BE00", "#73A950", "#00A1DF"]  # yellow green blue s m l
dark_3 = ["#cabc8c", "#4c5835", "#376977"]  # yellow green blue or this yellow - cabc8c
deep_3 = ["", "#005851", "#003B4C"]  # blue



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


def get_taskdf2plot(
    df: pd.DataFrame,
    task: str,
    metric: str,
    large_esm: str == "esm1b_t33_650M_UR50S",
    concise: bool = True,
):
    """
    A function define the bars to be plotted based on each task

    Args:
    - df: pd.DataFrame,
    - task: str,
    - large_esm: str == "esm1b_t33_650M_UR50S",
    - concise: bool = True
    """
    #  & (prepped_df["ptp"].isin([0, 1]))].copy()
    if "last_value" not in df.columns:
        df["last_value"] = df["value"].apply(
            lambda x: x[-1] if len(x) > 0 else None
        ).copy()

    task_df = df[(df["metric"] == metric) & (df["task"] == task)].copy()

    slice_df = task_df[
        (
            ((task_df["ablation"] == "emb"))  # get emb
            | (task_df["ablation"] == "onehot")  # get onehot
            | (
                (task_df["ablation"].isin(["rand", "stat"]))
                & (task_df["model"].isin([large_esm, "carp_640M"]))
            )
        )
    ].copy()

    # if further slice based on more concise
    select_model = slice_df["model"].isin(EMB4TASK)
    select_ptp = slice_df["ptp"].isin([0, 1])
    select_cp = (slice_df["arch"] != "esm") & (
        slice_df["ablation"].isin(["emb", "onehot"])
    )

    if concise:
        return (
            slice_df[select_model & select_ptp].copy(),
            slice_df[select_model & select_cp].copy(),
        )
    else:
        return (
            slice_df[select_ptp].copy(),
            slice_df[select_cp].copy(),
        )
    
def pair_taskwy(df: pd.DataFrame, val_col: str == "last_value") -> np.array:

    """A function to pair x for task based plots with y"""

    baseline_df = df[df["ablation"] != "emb"].copy()
    emb_df = df[df["ablation"] == "emb"].copy()

    baseline_df["ablation"] = pd.Categorical(
        baseline_df["ablation"],
        categories=list(BASELINE_NAME_DICT.keys()),
        ordered=True,
    )

    return list(baseline_df.sort_values("ablation")[val_col].values) + list(
        emb_df.sort_values("model_size")[val_col].values
    )

def eval_emb_vs_ab(
    df1: pd.DataFrame,
    randstat: str,
    error_margin: float = 0.1,
    large_esm: str = "esm1b_t33_650M_UR50S",
) -> float:
    """
    A function that counts largest emb vs ablation

    Args:
    - delta: float = 0.1, 5% increase at least
    """

    emb_vs_ab = 0
    # emb > rand * 1.05 & emb > stat * 1.05
    for largest in ["carp_640M", large_esm]:
        emb_perf = df1[(df1["model"] == largest) & (df1["ablation"] == "emb")][
            "last_value"
        ].values
        emb_ab = df1[(df1["model"] == largest) & (df1["ablation"] == randstat)][
            "last_value"
        ].values

        for ab in emb_ab:
            if emb_perf > ab * (1 + error_margin):
                emb_vs_ab += 1

    return emb_vs_ab / 2


# Check if the array is monotonically increasing within a 5% error margin
def is_monotonically_increasing_with_error(
    vals: np.array, error_margin: float = 0.1, tolerance: float = 0.2
) -> bool:
    """
    A function to see if a given array is monotonically increasing

    Args:
    - error_margin: float = 0.1, allowing each data point 5% error
    - tolerance: float = 0.2, allowing such 5% happen 20% out of the all data points
    """

    cap = len(vals) * tolerance
    exception_count = 0

    for i in range(1, len(vals)):
        if exception_count > cap:
            return False
        # Calculate the minimum value the next element should have to be considered as increasing
        min_value = vals[i - 1] * (1 - error_margin)
        if vals[i] < min_value:
            return False
        elif vals[i] < vals[i - 1]:
            exception_count += 1

    return True


def task_summary(
    task: str,
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    large_esm: str = "esm1b_t33_650M_UR50S",
    error_margin: float = 0.1,
    rho_cutoff: float = 0.9,
    tolerance: float = 0.2,
    window_size: int = 6,
) -> dict:

    """
    A function summarize a given task in terms of:

    Emb > OH
    Emb > Ab
    Scale with model size
    Scale with layer
    Scale with pretrain

    Args:
    - error_margin: float = 0.1, allowing each data point 5% error
    - rho_cutoff: float = 0.9, to what degree we call scale
    - tolerance: float = 0.2, allowing such 5% happen 20% out of the all data points
    - window_size: int = 5, moving average for the layer performance
    """

    summary_task_dict = {}

    # get onehot for the task
    onehot_val = df1[(df1["task"] == task) & (df1["model"] == "onehot")][
        "last_value"
    ].values[0]

    emb_df = df1[df1["ablation"] == "emb"].copy()

    # emb > onehot *1.05
    summary_task_dict["Transfer > One-hot"] = (
        sum(emb_df["last_value"] > onehot_val * (1 + error_margin)) / 6
    )
    summary_task_dict["Transfer > Random init"] = eval_emb_vs_ab(
        df1, "rand", error_margin
    )
    summary_task_dict["Transfer > Stat transfer"] = eval_emb_vs_ab(
        df1, "stat", error_margin
    )

    model_scale = 0
    layer_scale = 0

    for arch, large in zip(ARCH_TYPE, ["carp_640M", large_esm]):

        # do strickly greater than
        arch_model_scale = is_monotonically_increasing_with_error(
            emb_df[emb_df["arch"] == arch]
            .sort_values(by=["model_size"])["last_value"]
            .values,
            error_margin=0,
            tolerance=0,
        )
        summary_task_dict[f"Scale w {arch} size"] = arch_model_scale
        model_scale += arch_model_scale

        # make the layers smoother
        layer_perf = emb_df[(emb_df["model"] == large)]["value"].values[0]

        (
            layer_rho,
            summary_task_dict[f"{large} layer p"],
        ) = spearmanr(range(0, len(layer_perf)), layer_perf)

        if layer_rho >= rho_cutoff:
            arch_layer_scale = True
        else:
            arch_layer_scale = False

        summary_task_dict[f"{large} layer rho"] = layer_rho
        summary_task_dict[f"Scale w {large} layer"] = arch_layer_scale
        layer_scale += arch_layer_scale

    summary_task_dict["Scale with PLM sizes"] = model_scale / 2
    summary_task_dict["Scale with layer depths"] = layer_scale / 2

    pretrain_perf = (
        df2[df2["ablation"] == "emb"]
        .sort_values(by=["model_size", "ptp"])["last_value"]
        .values
    )

    (
        summary_task_dict["pretrain losses rho"],
        summary_task_dict["pretrain losses p"],
    ) = spearmanr(range(0, len(pretrain_perf)), pretrain_perf)

    if summary_task_dict["pretrain losses rho"] >= rho_cutoff:
        summary_task_dict["Scale with pretrain losses"] = True
    else:
        summary_task_dict["Scale with pretrain losses"] = False

    return summary_task_dict

def simplify_test_metric(metric: str) -> str:

    """
    A function to unify metric for plotting
    """

    for t in ["test_performance_1", "test_performance_2", "test_loss"]:
        metric = metric.replace(t, "test performance")

    return metric
