"""Script for running results analysis and plotting"""

from scr.analysis.perlayer import LayerLoss

LayerLoss(
    input_path="results/sklearn-carp",
    output_path="results/sklearn-carp_layer"
)

LayerLoss(
    input_path="results/sklearn-esm",
    output_path="results/sklearn-esm_layer"
)