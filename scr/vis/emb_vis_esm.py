"""Visulization for the ESM embeddings"""

from __future__ import annotations

from collections import Sequence

import torch

from scr.vis.emb_vis import Emb_Vis

ESM_MODEL_LIST = [
    "esm1b_t33_650M_UR50S",
    "esm1_t34_670M_UR50S",
    "esm1_t12_85M_UR50S",
    "esm1_t6_43M_UR50S",
]


def esm_emb_vis(
    esm_model_list: list(str) = ESM_MODEL_LIST,
    emb_start_ind: int = 0,
    emb_end_ind: int = 20,
) -> list(Emb_Vis):
    """
    Plot and save the esm embedding

    Args:
    - esm_model_list: list(str), a list of esm models
    - emb_start_ind: int, the start index of the esm embedding
    - emb_end_ind: int, the ending index of the esm embedding

    Returns:
    - emb_vis_class_list: list(Emb_Vis), a lsit of Emb_Vis class
    """

    emb_vis_class_list = [None] * len(esm_model_list)

    for m, esm_model in enumerate(esm_model_list):
        model, _ = torch.hub.load("facebookresearch/esm:main", esm_model)
        aa_emb = (
            model.embed_tokens.weight.detach()
            .cpu()
            .numpy()[emb_start_ind:emb_end_ind, :]
        )

        print(f"Plotting and saving {esm_model}")
        emb_vis_class_list[m] = Emb_Vis(
            aa_emb=aa_emb,
            emb_name=f"{esm_model} input amino acids",
            subfolder="esm",
        )

    return emb_vis_class_list


"""
# https://github.com/facebookresearch/esm/blob/main/esm/constants.py
proteinseq_toks = {
    "toks": [
        "L",
        "A",
        "G",
        "V",
        "S",
        "E",
        "R",
        "T",
        "I",
        "D",
        "P",
        "K",
        "Q",
        "N",
        "F",
        "Y",
        "M",
        "H",
        "W",
        "C",
        "X",
        "B",
        "U",
        "Z",
        "O",
        ".",
        "-",
    ]
}

# https://github.com/facebookresearch/esm/blob/main/esm/data.py
def from_architecture(cls, name: str) -> "Alphabet":
    if name in ("ESM-1", "protein_bert_base"):
        standard_toks = proteinseq_toks["toks"]
        prepend_toks: Tuple[str, ...] = ("<null_0>", "<pad>", "<eos>", "<unk>")
        append_toks: Tuple[str, ...] = ("<cls>", "<mask>", "<sep>")
        prepend_bos = True
        append_eos = False
        use_msa = False
    elif name in ("ESM-1b", "roberta_large"):
        standard_toks = proteinseq_toks["toks"]
        prepend_toks = ("<cls>", "<pad>", "<eos>", "<unk>")
        append_toks = ("<mask>",)
        prepend_bos = True
        append_eos = True
        use_msa = False
    elif name in ("MSA Transformer", "msa_transformer"):
        standard_toks = proteinseq_toks["toks"]
        prepend_toks = ("<cls>", "<pad>", "<eos>", "<unk>")
        append_toks = ("<mask>",)
        prepend_bos = True
        append_eos = False
        use_msa = True

    return cls(
        standard_toks, prepend_toks, append_toks, prepend_bos, append_eos, use_msa
    )
    """