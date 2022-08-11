"""Embedding constants"""

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