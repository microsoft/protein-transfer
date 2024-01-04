"""Embedding constants"""

ARCH_TYPE = ["esm", "carp"]

ARCH_CUT_DICT = {
    "": [2, 3, 4, 6],
    "carp": [2, 4, 6, 12],
    "esm": [2, 3, 4, 6]
}

# the longest seq for esm1b is 1022 with CLS and EOS
MAX_SEQ_LEN = 1022

# encoder_name: (embedding dim, num layers, token dim)
TRANSFORMER_INFO = {
    "esm1_t6_43M_UR50S": (768, 6, 2),
    "esm1_t12_85M_UR50S": (768, 12, 2),
    "esm1_t34_670M_UR50S": (1280, 34, 2),
    "esm1b_t33_650M_UR50S": (1280, 33, 2),
}

# encoder_name: (d_model, n_layers)
CARP_INFO = {
    "carp_600k": (128, 16),
    "carp_38M": (1024, 16),
    "carp_76M": (1024, 32),
    "carp_640M": (1280, 56),
}

# model parameter number in M
MODEL_SIZE = {
    "esm1_t6_43M_UR50S": 43,
    "esm1_t12_85M_UR50S": 85,
    "esm1_t34_670M_UR50S": 670,
    "esm1b_t33_650M_UR50S": 650,
    "carp_600k": 0.6,
    "carp_38M": 38,
    "carp_76M": 76,
    "carp_640M": 640,
    "onehot": 0.02
}

MODEL_LAYER = {
    model_name: model_dets[1]
    for info_dict in [TRANSFORMER_INFO, CARP_INFO, {"onehot": (1, 1)}]
    for model_name, model_dets in info_dict.items()
}

# TODO integrate to be auto fromr sheet
CARP_CHECKPOINTS = {
    "carp_600k": {0.5: 239263, 0.25: 114344, 0.125: 52039},
    "carp_38M": {0.5: 517622, 0.25: 256897, 0.125: 129575},
    "carp_76M": {0.5: 327960, 0.25: 162959, 0.125: 83180},
    "carp_640M": {0.5: 311757, 0.25: 154698, 0.125: 78810},
}

CARP_600K = {
    "experiment": "pretrain",
    "task": "mlm",
    "dataset": "uniref50",
    "d_embed": 8,
    "d_model": 128,
    "activation": "gelu",
    "slim": True,
    "epochs": 100,
    "n_layers": 16,
    "kernel_size": 5,
    "r": 128,
    "max_tokens": 600000,
    "max_batch_size": 1600,
    "bucket_size": 10000,
    "opt_level": "O2",
    "lr": 1e-3,
    "warmup_steps": 16000,
    "train_steps": 8000,
}

CARP_38M = {
    "experiment": "pretrain",
    "task": "mlm",
    "dataset": "uniref50",
    "d_embed": 8,
    "d_model": 1024,
    "activation": "gelu",
    "slim": True,
    "epochs": 100,
    "n_layers": 16,
    "kernel_size": 5,
    "r": 128,
    "max_tokens": 40000,
    "max_batch_size": 800,
    "bucket_size": 1000,
    "opt_level": "O2",
    "lr": 1e-3,
    "warmup_steps": 16000,
    "train_steps": 8000,
}

CARP_76M = {
    "experiment": "pretrain",
    "task": "mlm",
    "dataset": "uniref50",
    "d_embed": 8,
    "d_model": 1024,
    "activation": "gelu",
    "slim": True,
    "epochs": 100,
    "n_layers": 32,
    "kernel_size": 5,
    "r": 128,
    "max_tokens": 60000,
    "max_batch_size": 800,
    "bucket_size": 1000,
    "opt_level": "O2",
    "lr": 1e-3,
    "warmup_steps": 16000,
    "train_steps": 8000,
}

CARP_640M = {
    "experiment": "pretrain",
    "task": "mlm",
    "dataset": "uniref50",
    "d_embed": 8,
    "d_model": 1280,
    "activation": "gelu",
    "slim": False,
    "epochs": 100,
    "n_layers": 56,
    "kernel_size": 5,
    "r": 128,
    "max_tokens": 10000,
    "max_batch_size": 800,
    "bucket_size": 1000,
    "opt_level": "O2",
    "lr": 4e-4,
    "warmup_steps": 16000,
    "train_steps": 8000,
}