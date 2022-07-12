"""Embedding constants"""


# (embedding dim, num layers, token dim)
TRANSFORMER_INFO = {
    "esm1_t6_43M_UR50S": (768, 6, 2),
    "esm1_t12_85M_UR50S": (768, 12, 2),
    "esm1b_t33_650M_UR50S": (1280, 33, 2),
    "esm1_t34_670M_UR50S": (1280, 34, 2),
}