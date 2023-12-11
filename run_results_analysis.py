"""Script for running results analysis and plotting"""

from scr.analysis.perlayer import LayerLoss
from scr.analysis.result_reorg import ResultReorg
from scr.vis.res_vis import PlotLayerDelta, PlotResultScatter
from scr.params.vis import ARCH_CUT_DICT

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

plot_class = PlotResultScatter()

for metric in ["test_performance_1", "test_performance_2"]:

    for arch in ["esm", "carp"]:
        plot_class.plot_emb_onhot(
            metric = metric,
            arch = arch
        );

        for cut in ARCH_CUT_DICT[arch]:
            plot_class.plot_layer_delta(
                layer_cut=cut,
                metric=metric,
                ablation="emb",
                arch=arch
            );