"""Script for running results analysis and plotting"""

from scr.analysis.perlayer import LayerLoss

print("Running results analysis and plotting for CARP...")

LayerLoss(
    input_path="results/sklearn-carp",
    output_path="results/sklearn-carp_layer"
)

# print("Running results analysis and plotting for ESM...")

# LayerLoss(
#     input_path="results/sklearn-esm",
#     output_path="results/sklearn-esm_layer",
#     add_checkpoint=False
# )