"""A tempo clean up for summary plot"""

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
    EMB_SIZE_SIMPLE
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

    @property
    def best_metric_df_dict(self) -> dict:
        """Return a dict splicing the df with best metric based on best metric"""
        return self._get_best_metric_df_dicts()

    @property
    def last_metric_df_dict(self) -> dict:
        """Return a dict splicin the df with the last layer value given metric"""
        return self._get_last_metric_df_dict()


def simplify_summeray_df(df: pd.DataFrame) -> pd.DataFrame:

    """
    A function to simply the last layer df
    """

    # Applying the rules to the respective columns
    for col in ["Emb > OH", "RI > OH", "ST > OH", "RI > ST"]:
        df[col] = df[col].apply(lambda value: True if value > 0 else False)

    # flip order
    for (col, ncol) in zip(["RI > Emb", "ST > Emb"], ["Emb > RI", "Emb > ST"]):
        df[ncol] = df[col].apply(lambda value: False if value > 0 else True)
        df = df.drop(columns=col)

    return df


def simplify_test_metric(metric: str) -> str:

    """
    A function to unify metric for plotting
    """

    for t in ["test_performance_1", "test_performance_2", "test_loss"]:
        metric = metric.replace(t, "test performance")

    return metric
