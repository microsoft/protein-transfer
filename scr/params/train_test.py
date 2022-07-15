"""Parameters for training and testing"""

import torch

# Set up cuda variables
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_WORKERS = 8

RAND_SEED = 42