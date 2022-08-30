"""For dataset vis"""

from __future__ import annotations

import pandas as pd

import iqplot

from scr.params.vis import PLOT_EXTS
from scr.vis.vis_utils import BokehSave
from scr.utils import get_task_data_split, read_std_csv

class DatasetECDF(BokehSave):
    def __init__(
        self,
        dataset_path: str,
        path2folder: str = "results/dataset_vis",
        plot_exts: list = PLOT_EXTS,
        plot_height: int = 300,
        plot_width: int = 450,
        axis_font_size: str = "10pt",
        title_font_size: str = "10pt",
        x_name: str = "fitness",
        y_name: str = "ecdf",
        gridoff: bool = True,
    ) -> None:

        df = read_std_csv(dataset_path)

        df.loc[df["validation"] == True, "set"] = "val"

        self.bokeh_plot = iqplot.ecdf(
                df,
                q="target",
                cats="set",
                conf_int=True,
                # style="staircase",
                order=["train", "val", "test"],
                legend_location="bottom_right",
                marker_kwargs={"alpha": 0.5},
                fill_kwargs={"fill_alpha": 0.1}
                # line_kwargs={"line_width": 2.5},
            )

        super(DatasetECDF, self).__init__(
            bokeh_plot=self.bokeh_plot,
            path2folder=path2folder,
            plot_name="-".join(get_task_data_split(dataset_path)),
            plot_exts=plot_exts,
            plot_height=plot_height,
            plot_width=plot_width,
            axis_font_size=axis_font_size,
            title_font_size=title_font_size,
            x_name=x_name,
            y_name=y_name,
            gridoff=gridoff,
        )