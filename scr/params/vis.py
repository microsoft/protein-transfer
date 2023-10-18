"""Parameters for plotting"""

from __future__ import annotations

# allowed dimension reduction types
ALLOWED_DIM_RED_TYPES = ["pca", "tsne", "umap"]
PLOT_EXTS = [".png", ".svg"]

PRESENTATION_PALETTE_SATURATE_DICT = {
    "blue": "#4bacc6",
    "orange": "#f79646",
    "green":"#9bbb59",
    "purple":"#8064a2",
    "gray":"#666666",
}

# blue, orange, green, purple, gray
PRESENTATION_PALETTE_SATURATE = list(PRESENTATION_PALETTE_SATURATE_DICT.keys())

# lgihter organes: 45%, 30%, 15%
# from https://www.w3schools.com/colors/colors_picker.asp
CHECKPOINT_COLOR = {
    0.5: "#dc6809",
    0.25: "#934506",
    0.125: "#492303"
}

# the order for plotting legend
ORDERED_TASK_LIST = [
    "proeng_gb1_sampled_mean",
    "proeng_gb1_low_vs_high_mean",
    "proeng_gb1_two_vs_rest_mean",
    "proeng_aav_two_vs_many_mean",
    "proeng_aav_one_vs_many_mean",
    "proeng_thermo_mixed_split_mean",
    "annotation_scl_balanced_mean",
    "structure_ss3_casp12_noflatten",
    "structure_ss3_cb513_noflatten",
    "structure_ss3_ts115_noflatten"
]

# assume same order as the above
ORDERED_TASK_LIST_SIMPLE =[
    "GB1 - sampled",
    "GB1 - low vs high",
    "GB1 - two vs rest",
    "AAV - two vs many",
    "AAV - one vs many",
    "Thermostability",
    "Subcellular localization",
    "SS3 - CASP12",
    "SS3 - CB513",
    "SS3 - TS115"
]

TASK_LEGEND_MAP = {k: v for k, v in zip(ORDERED_TASK_LIST, ORDERED_TASK_LIST_SIMPLE)}