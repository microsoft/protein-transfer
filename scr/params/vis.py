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