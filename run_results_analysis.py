"""Script for running results analysis and plotting"""

from scr.analysis.perlayer import LayerLoss
from scr.analysis.result_reorg import ResultReorg
from scr.vis.res_vis import PlotLayerDelta

"""
print("Running results analysis and plotting for sklearn CARP...")

LayerLoss(
    input_path="results/sklearn-carp",
    output_path="results/sklearn-carp_layer"
)

print("Running results analysis and plotting for sklearn ESM...")

LayerLoss(
    input_path="results/sklearn-esm",
    output_path="results/sklearn-esm_layer",
    add_checkpoint=False
)

print("Running results analysis and plotting for pytorch CARP...")

LayerLoss(
    input_path="results/pytorch-carp",
    output_path="results/pytorch-carp_layer"
)

print("Running results analysis and plotting for pytorch ESM...")

LayerLoss(
    input_path="results/pytorch-esm",
    output_path="results/pytorch-esm_layer",
    add_checkpoint=False
)
"""

# ResultReorg()

PlotLayerDelta().plot_sub_df(
    layer_cut=2,
    metric = "test_performance_1",
    ablation = "emb",
    arch = "carp",);

PlotLayerDelta().plot_sub_df(
    layer_cut=3,
    metric = "test_performance_1",
    ablation = "emb",
    arch = "carp",);

PlotLayerDelta().plot_sub_df(
    layer_cut=4,
    metric = "test_performance_1",
    ablation = "emb",
    arch = "carp",);