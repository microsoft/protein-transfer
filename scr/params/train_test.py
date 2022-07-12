"""Parameters for training and testing"""

import torch

# Set up cuda variables
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"