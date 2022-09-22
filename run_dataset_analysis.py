"""A script for running dataset analysis"""

from __future__ import annotations

from scr.preprocess.data_process import TaskProcess
from scr.vis.dataset_vis import DatasetStripHistogram

TaskProcess(forceregen=True, showplot=False)

# DatasetStripHistogram(
#     dataset_folder="data/proeng/gb1",
#     split_order=["sampled", "two_vs_rest", "low_vs_high"],
#     plot_width=800,
#     showplot=False,
# )
# DatasetStripHistogram(
#     dataset_folder="data/proeng/aav",
#     split_order=["one_vs_many", "two_vs_many"],
#     showplot=False,
# )
# DatasetStripHistogram(
#     dataset_folder="data/proeng/thermo", 
#     plot_width=400,
#     showplot=False)