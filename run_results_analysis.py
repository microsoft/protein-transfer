"""Script for running results analysis and plotting"""

from scr.analysis.perlayer import LayerLoss
from scr.analysis.result_reorg import ResultReorg

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

ResultReorg()