"""
A script for calculating statistical significance for random seed replicates
for random init and stat transfer
"""

from __future__ import annotations

import os
import ast
import warnings

import numpy as np
import pandas as pd

from scipy import stats

warnings.filterwarnings("ignore")

# ablations for rep stat testing
AB_STAT = ["rand", "stat"]


def perform_t_test(
    grouped_df,
    test_col: str = "last_layer",
    target_col: str = "emb_value",
    sig_cutoff: float = 0.05,
):

    # Use the target value specific to this group, assumed to be the same for all rows in the group
    target_value = grouped_df[target_col].iloc[0]
    # t_statistic, p_value = stats.ttest_1samp(grouped_df[test_col], target_value)
    # # For a one-tailed test, adjust p-value accordingly
    # one_tailed_p_value = p_value / 2 if t_statistic < 0 else 1 - (p_value / 2)
    t_statistic, one_tailed_p_value = stats.ttest_1samp(grouped_df[test_col], target_value, alternative="less")
    return pd.Series(
        {
            "mean": grouped_df[test_col].mean(),
            "std": grouped_df[test_col].std(),
            "n": len(grouped_df[test_col]),
            "t_statistic": t_statistic,
            "p_value": one_tailed_p_value,
            "significant": one_tailed_p_value < sig_cutoff,
        }
    )


class RepStat:
    """
    A class for getting replicate stats
    """

    def __init__(self, 
    summary_csv: str = "results/summary/all_results.csv",
    stat_csv: str = "results/summary/repstat.csv"):

        self._summary_csv = summary_csv
        self._stat_csv = stat_csv

        all_dfs = []

        for metric in ["test_performance_1", "test_performance_2"]:
            for ablation in AB_STAT:
                all_dfs.append(self._get_reptest(metric=metric, ablation=ablation))

        repstat_df = pd.concat(all_dfs, axis=0)

        repstat_df.applymap(lambda x: f"{x:.4f}" if isinstance(x, (float, int)) else x)

        repstat_df["significant"] = repstat_df.apply(
            lambda row: np.nan
            if row.drop("significant").isnull().any()
            else row["significant"],
            axis=1,
        )

        self._repstat_df = repstat_df.copy()

        repstat_df.to_csv(self._stat_csv, index=False)

    def _get_reptest(self, metric: str, ablation: str) -> pd.DataFrame:

        """ """

        assert ablation in AB_STAT, f"{ablation} not in ['rand', 'stat']"

        emb_df = (
            self.df[
                (self.df["ablation"] == "emb")
                & (self.df["metric"] == metric)
                & (self.df["ptp"] == 1)
            ]
            .drop(columns=["embseed", "ablation", "value"])
            .rename(columns={"last_layer": "emb_value"})
        )

        ab_df = self.df[
            (self.df["ablation"] == ablation) & (self.df["metric"] == metric)
        ]

        merge_df = pd.merge(
            ab_df.drop(columns=["value"]),
            emb_df,
            on=["arch", "task", "model", "metric"],
            how="left",
        ).dropna()

        tested_df = (
            merge_df.groupby(["arch", "task", "model", "metric"])
            .apply(perform_t_test)
            .reset_index()
        )
        tested_df["ablation"] = ablation

        return tested_df.copy()

    @property
    def df(self):
        """Return the full df with seeds"""
        # get last layer value
        df = pd.read_csv(self._summary_csv)
        df["last_layer"] = (
            df["value"]
            .apply(lambda x: ast.literal_eval(x)[-1] if x else None)
            .replace(0, np.nan)
        )
        return df.copy()

    @property
    def repstat_df(self):
        """"""
        return self._repstat_df