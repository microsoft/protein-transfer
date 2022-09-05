"""Add encoding classes with class methods"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Collection
from collections import Iterable, Sequence

import math
import random
import numpy as np
from tqdm import tqdm

import torch
from torch.nn import Parameter
from torch.nn.init import (
    xavier_uniform_,
    xavier_normal_,
    kaiming_uniform_,
    uniform_,
    normal_,
    constant_,
    _calculate_fan_in_and_fan_out,
)
from sequence_models.pretrained import load_model_and_alphabet

from scr.params.aa import AA_NUMB, AA_TO_IND
from scr.params.emb import TRANSFORMER_INFO, CARP_INFO
from scr.params.sys import DEVICE, RAND_SEED


# seed everything
random.seed(RAND_SEED)
np.random.seed(RAND_SEED)
torch.manual_seed(RAND_SEED)
torch.cuda.manual_seed(RAND_SEED)
torch.cuda.manual_seed_all(RAND_SEED)
torch.backends.cudnn.deterministic = True


def cal_bound(model: torch.nn.Module, layer_name: str):
    """Return bound for reinit given model and layer name"""
    assert "bias" in layer_name, f"no bias in {layer_name}"
    fan_in, _ = _calculate_fan_in_and_fan_out(
        model.state_dict()[layer_name.replace("bias", "weight")]
    )
    return 1 / math.sqrt(fan_in) if fan_in > 0 else 0


class AbstractEncoder(ABC):
    """
    An abstract encoder class to fill in for different kinds of encoders

    All encoders will have an "encode" function
    """

    def __init__(
        self,
        encoder_name: str = "",
        reset_param: bool = False,
        resample_param: bool = False,
    ):

        """
        Args:
        - encoder_name: str, the name of the encoder, default empty for onehot
        - reset_param: bool = False, if update the full model to xavier_uniform_
        - resample_param: bool = False, if update the full model to xavier_normal_
        """

        self._encoder_name = encoder_name

        assert reset_param * resample_param != 1, "Choose reset OR resample param"

        self._reset_param = reset_param
        self._resample_param = resample_param

    def reset_resample_param(self, model: torch.nn.Module):
        """
        Initiate parameters in the PyTorch model. Following:
        https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#Transformer

        Args:
        - model: torch.nn.Module, the input model

        Returns:
        - torch.nn.Module, the model with all params set with xavier_uniform
        """
        if self._reset_param:
            print(f"Reinit params for {self._encoder_name} ...")

            """
            ESM1b:

            layers.n.self_attn.k_proj.weight: dim 2         [nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))]
            layers.n.self_attn.v_proj.weight: dim 2         [nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))]
            layers.n.self_attn.q_proj.weight: dim 2         [nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))]
            layers.n.self_attn.out_proj.weight: dim 2       [nn.init.xavier_uniform_(self.out_proj.weight)]
            
            layers.n.self_attn.k_proj.bias: dim 1           [nn.init.uniform_(self.bias, -bound, bound)]
            layers.n.self_attn.v_proj.bias: dim 1           [nn.init.uniform_(self.bias, -bound, bound)]
            layers.n.self_attn.q_proj.bias: dim 1           [nn.init.uniform_(self.bias, -bound, bound)]
            
            layers.n.self_attn.out_proj.bias: dim 1         [nn.init.constant_(self.out_proj.bias, 0.0)]
            
            layers.n.self_attn_layer_norm.weight: dim 1     [nn.Parameter(torch.ones(hidden_size))]
            layers.n.final_layer_norm.weight: dim 1         [nn.Parameter(torch.ones(hidden_size))]
            emb_layer_norm_before.weight: dim 1             [nn.Parameter(torch.ones(hidden_size))]
            emb_layer_norm_after.weight: dim 1              [nn.Parameter(torch.ones(hidden_size))]
            
            layers.n.self_attn_layer_norm.bias: dim 1       [nn.Parameter(torch.zeros(hidden_size))]
            layers.n.final_layer_norm.bias: dim 1           [nn.Parameter(torch.zeros(hidden_size))]
            emb_layer_norm_before.bias: dim 1               [nn.Parameter(torch.zeros(hidden_size))]
            emb_layer_norm_after.bias: dim 1                [nn.Parameter(torch.zeros(hidden_size))]
            
            layers.n.fc1.weight: dim 2                      [nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))]
            layers.n.fc2.weight: dim 2                      [nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))]
            contact_head.regression.weight: dim 2           [nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))]
            
            layers.n.fc1.bias: dim 1                        [nn.init.uniform_(self.bias, -bound, bound)]
            layers.n.fc2.bias: dim 1                        [nn.init.uniform_(self.bias, -bound, bound)]
            contact_head.regression.bias: dim 1             [nn.init.uniform_(self.bias, -bound, bound)]
            
            embed_positions.weight: dim 2                   [nn.init.normal_(self.weight)]
            """

            for layer_name, p in model.state_dict().items():
                # what esm1b and esm1 have in common
                if "_proj" in layer_name:
                    if "weight" in layer_name:
                        if "out" in layer_name:
                            xavier_uniform_(p)
                        else:
                            xavier_uniform_(p, gain=1 / math.sqrt(2))
                    elif "bias" in layer_name:
                        if "out" in layer_name:
                            constant_(p, 0.0)
                        else:
                            bound = cal_bound(model=model, layer_name=layer_name)
                            uniform_(p, -bound, bound)

                # esm1b enced up using LayerNorm so the same
                if "layer_norm" in layer_name:
                    if "weight" in layer_name:
                        Parameter(torch.ones_like(p))
                    elif "bias" in layer_name:
                        Parameter(torch.zeros_like(p))

                if ("layers" and "fc" in layer_name) or ("contact_head" in layer_name):
                    if "weight" in layer_name:
                        kaiming_uniform_(p, a=math.sqrt(5))
                    elif "bias" in layer_name:
                        bound = cal_bound(model=model, layer_name=layer_name)
                        uniform_(p, -bound, bound)

                if "esm1b_" in self._encoder_name:

                    if "embed_positions" in layer_name:
                        normal_(p)

                    if layer_name == "lm_head.weight":
                        xavier_uniform_(p)

                    if layer_name == "lm_head.bias" or "lm_head.layer_norm.bias":
                        Parameter(torch.zeros_like(p))

                    if "dense" in layer_name:
                        if "weight" in layer_name:
                            kaiming_uniform_(p, a=math.sqrt(5))
                        elif "bias" in layer_name:
                            bound = cal_bound(model=model, layer_name=layer_name)
                            uniform_(p, -bound, bound)

                elif "esm1_" and "bias_" in self._encoder_name:
                    xavier_normal_(p)

        elif self._resample_param:
            print(f"Resample params for {self._encoder_name} ...")

            resample_state = model.state_dict()
            for layer_name, p in model.state_dict().items():
                if (
                    ("embed_tokens" not in layer_name)
                    and ("embed_out" not in layer_name)
                    and ("_float_tensor" not in layer_name)
                ):

                    if len(p.shape) == 1:
                        resample_state[layer_name] = p[torch.randperm(p.shape[0])]
                        """
                        layers.n.self_attn.k_proj.bias: torch.Size([embed_dim])
                        layers.n.self_attn.v_proj.bias: torch.Size([embed_dim])
                        layers.n.self_attn.q_proj.bias: torch.Size([embed_dim])
                        layers.n.self_attn.out_proj.bias: torch.Size([embed_dim])
                        layers.n.self_attn_layer_norm.weight: torch.Size([embed_dim])
                        layers.n.self_attn_layer_norm.bias: torch.Size([embed_dim])
                        layers.n.fc1.bias: torch.Size([fc_dim])
                        layers.n.fc2.bias: torch.Size([embed_dim])
                        layers.n.final_layer_norm.weight: torch.Size([embed_dim])
                        layers.n.final_layer_norm.bias: torch.Size([embed_dim])
                        """
                    elif 1 in p.shape:
                        """
                        layers.n.self_attn.bias_k: torch.Size([1, 1, embed_dim])
                        layers.n.self_attn.bias_v: torch.Size([1, 1, embed_dim])
                        contact_head.regression.weight: torch.Size([1, reg_dim])
                        """
                        if "bias_" in layer_name:
                            resample_state[layer_name] = p[
                                :, :, torch.randperm(self._embed_dim)
                            ]
                        elif "regression.weight" in layer_name:
                            resample_state[layer_name] = p[
                                :, torch.randperm(p.shape[-1])
                            ]

                    elif (
                        "k_proj.weight" or "q_proj.weight" or "fc1.weight" in layer_name
                    ):
                        resample_state[layer_name] = p[torch.randperm(p.shape[0]), :]
                        """
                        layers.n.self_attn.k_proj.weight: torch.Size([embed_dim, embed_dim])
                        layers.n.self_attn.v_proj.weight: torch.Size([embed_dim, embed_dim])
                        layers.n.self_attn.q_proj.weight: torch.Size([embed_dim, embed_dim])
                        layers.n.self_attn.out_proj.weight: torch.Size([embed_dim, embed_dim])
                        layers.n.fc1.weight: torch.Size([fc_dim, embed_dim])
                        layers.n.fc2.weight: torch.Size([embed_dim, fc_dim])
                        """

                    elif (
                        "v_proj.weight"
                        or "out_proj.weight"
                        or "fc2.weight" in layer_name
                    ):

                        resample_state[layer_name] = p[:, torch.randperm(p.shape[1])]

            model.load_state_dict(resample_state)

            return model

    def encode(
        self,
        mut_seqs: Sequence[str] | str,
        batch_size: int = 0,
        flatten_emb: bool | str = False,
        mut_names: Sequence[str] | str | None = None,
    ) -> Iterable[np.ndarray]:
        """
        A function takes a list of sequences to yield a batch of encoded elements

        Args:
        - mut_seqs: list of str or str, mutant sequences of the same length
        - batch_size: int, set to 0 to encode all in a single batch
        - flatten_emb: bool or str, if and how (one of ["max", "mean"]) to flatten the embedding
        - mut_names: list of str or str or None, mutant names

        Returns:
        - generator: dict with layer number as keys and
            encoded flattened sequence with or without labels as value
        """

        if isinstance(mut_seqs, str):
            mut_seqs = [mut_seqs]

        # If the batch size is 0, then encode all at once in a single batch
        if batch_size == 0:
            yield self._encode_batch(
                mut_seqs=mut_seqs, flatten_emb=flatten_emb, mut_names=mut_names
            )

        # Otherwise, yield chunks of encoded sequence
        else:

            for i in tqdm(range(0, len(mut_seqs), batch_size)):

                # figure out what mut_names to feed in
                if mut_names is None:
                    mut_name_batch = mut_names
                else:
                    mut_name_batch = mut_names[i : i + batch_size]

                yield self._encode_batch(
                    mut_seqs=mut_seqs[i : i + batch_size],
                    flatten_emb=flatten_emb,
                    mut_names=mut_name_batch,
                )

    def flatten_encode(
        self,
        encoded_mut_seqs: np.ndarray,
        flatten_emb: bool | str,
        mut_seqs: Sequence[str] | str,
    ) -> np.ndarray:
        """
        Flatten the embedding or just return the encoded mutants.

        Args:
        - encoded_mut_seqs: np.ndarray, shape [batch_size, seq_len, embed_dim]
        - flatten_emb: bool or str, if and how (one of ["max", "mean"]) to flatten the embedding
            - True -> shape [batch_size, seq_len * embed_dim]
            - "max" or "mean" -> shape [batch_size, embed_dim]
            - False or everything else -> [batch_size, seq_len, embed_dim]

        Returns:
        - np.ndarray, shape depends on flatten_emb parameter
        """

        assert (
            encoded_mut_seqs.shape[-1] == self._embed_dim
        ), f"encode last dim {encoded_mut_seqs.shape[-1]} != embed dim {self._embed_dim}"

        if flatten_emb in [True, "flatten", "flattened", ""]:
            # shape [batch_size, seq_len * embed_dim]
            return encoded_mut_seqs.reshape(encoded_mut_seqs.shape[0], -1)

        elif isinstance(flatten_emb, str):
            # init out put seq_reps should be in dim [batch_size, embed_dim]
            seq_reps = np.empty((encoded_mut_seqs.shape[0], self._embed_dim))
            for i, encoded_mut_seq in enumerate(encoded_mut_seqs):
                if flatten_emb == "mean":
                    seq_reps[i] = encoded_mut_seq[: len(mut_seqs[i])].mean(0)
                elif flatten_emb == "max":
                    seq_reps[i] = encoded_mut_seq[: len(mut_seqs[i])].max(0)

            return seq_reps

        else:
            # print("No embedding flattening")
            # [batch_size, seq_len, embed_dim]
            return encoded_mut_seqs

    @abstractmethod
    def _encode_batch(
        mut_seqs: Sequence[str] | str,
        flatten_emb: bool | str,
        mut_names: Sequence[str] | str | None = None,
    ) -> np.ndarray:
        """
        Encode a single batch of mut_seqs
        """
        pass

    @property
    def embed_dim(self) -> int:
        """The dim of the embedding"""
        return self._embed_dim

    @property
    def max_emb_layer(self) -> int:
        """The max layer nubmer of the embedding"""
        return self._max_emb_layer

    @property
    def include_input_layer(self) -> bool:
        """If include the input layer when counting the max layer number"""
        return self._include_input_layer

    @property
    def total_emb_layer(self) -> int:
        """Total embedding layer number"""
        return self._max_emb_layer + self._include_input_layer

    @property
    def encoder_name(self) -> str:
        """The name of the encoding method"""
        return self._encoder_name


class OnehotEncoder(AbstractEncoder):
    """
    Build a onehot encoder
    """

    def __init__(
        self,
        max_seq_len: int,
        encoder_name: str = "",
        reset_param: bool = False,
        resample_param: bool = False,
    ):
        """
        Args
        - encoder_name: str, the name of the encoder, one of the keys of CARP_INFO
        - max_seq_len: int, the longest sequence length
        - reset_param: bool = False, if update the full model to xavier_uniform_
        - resample_param: bool = False, if update the full model to xavier_normal_
        """
        super().__init__(encoder_name, reset_param, resample_param)

        self.max_seq_len = max_seq_len

        if encoder_name not in (TRANSFORMER_INFO.keys() and CARP_INFO.keys()):
            self._encoder_name = "onehot"
            self._embed_dim, self._max_emb_layer = AA_NUMB, 0
            self._include_input_layer = True

        # load model from torch.hub
        print(
            f"Generating {self._encoder_name} upto {self._max_emb_layer} layer embedding ..."
        )

        if reset_param or resample_param:
            self._reset_param = False
            self._resample_param = False
            print(
                f"Onehot encoding reset or resample param not allowed. /n \
                    Setting both to {self._reset_param} ..."
            )

    def _encode_batch(
        self,
        mut_seqs: Sequence[str] | str,
        flatten_emb: bool | str,
        mut_names: Sequence[str] | str | None = None,
    ) -> np.ndarray:

        encoded_mut_seqs = []

        for mut_seq in mut_seqs:
            # padding: (top, bottom), (left, right)
            encoded_mut_seqs.append(
                np.pad(
                    np.array(np.eye(AA_NUMB)[[AA_TO_IND[aa] for aa in mut_seq]]),
                    pad_width=((0, self.max_seq_len - len(mut_seq)), (0, 0)),
                )
            )

        return {
            0: self.flatten_encode(
                encoded_mut_seqs=np.array(encoded_mut_seqs),
                flatten_emb=flatten_emb,
                mut_seqs=mut_seqs,
            )
        }


class ESMEncoder(AbstractEncoder):
    """
    Build an ESM encoder
    """

    def __init__(
        self,
        encoder_name: str,
        reset_param: bool = False,
        resample_param: bool = False,
        iftrimCLS: bool = True,
        iftrimEOS: bool = True,
    ):
        """
        Args
        - encoder_name: str, the name of the encoder, one of the keys of TRANSFORMER_INFO
        - reset_param: bool = False, if update the full model to xavier_uniform_
        - resample_param: bool = False, if update the full model to xavier_normal_
        - iftrimCLS: bool, whether to trim the first classifification token
        - iftrimEOS: bool, whether to trim the end of sequence token, if exists
        """

        super().__init__(encoder_name, reset_param, resample_param)

        self._iftrimCLS = iftrimCLS
        self._iftrimEOS = iftrimEOS

        # get transformer dim and layer info
        self._embed_dim, self._max_emb_layer, _ = TRANSFORMER_INFO[self._encoder_name]

        # esm has the input representation
        self._include_input_layer = True

        # load model from torch.hub
        print(
            f"Generating {self._encoder_name} upto {self._max_emb_layer} layer embedding ..."
        )

        self.model, self.alphabet = torch.hub.load(
            "facebookresearch/esm:main", model=self._encoder_name
        )
        self.batch_converter = self.alphabet.get_batch_converter()

        # if reset or resample weights
        self.model = self.reset_resample_param(model=self.model)

        # set model to eval mode
        self.model.eval()
        self.model.to(DEVICE)

        expected_num_layers = int(self._encoder_name.split("_")[-3][1:])
        assert (
            expected_num_layers == self._max_emb_layer
        ), "Wrong ESM model name or layer"

    def _encode_batch(
        self,
        mut_seqs: Sequence[str] | str,
        flatten_emb: bool | str,
        mut_names: Sequence[str] | str | None = None,
    ) -> np.ndarray:
        """
        Encodes a batch of mutant sequences.

        Args:
        - mut_seqs: list of str or str, mutant sequences of the same length
        - flatten_emb: bool or str, if and how (one of ["max", "mean"]) to flatten the embedding
        - mut_names: list of str or str or None, mutant names

        Returns:
        - np.ndarray or a tuple(np.ndarray, list[str]) where the list is batch_labels
        """

        if isinstance(mut_names, str):
            mut_names = [mut_names]

        # pair the mut_names and mut_seqs
        if mut_names is not None:
            assert len(mut_names) == len(
                mut_seqs
            ), "mutant_name and mut_seqs different length"
            mut_seqs = [(n, m) for (n, m) in zip(mut_names, mut_seqs)]
        else:
            mut_seqs = [("", m) for m in mut_seqs]

        # convert raw mutant sequences to tokens
        batch_labels, _, batch_tokens = self.batch_converter(mut_seqs)
        batch_tokens = batch_tokens.to(DEVICE)

        # Turn off gradients and pass the batch through
        with torch.no_grad():
            # shape [batch_size, seq_len + pad, embed_dim]
            """if batch_tokens.shape[1] > TRANSFORMER_MAX_SEQ_LEN:
            print(f"Sequence exceeds {TRANSFORMER_MAX_SEQ_LEN}, taking the beginning and the end")
            batch_tokens = batch_tokens[:, :TRANSFORMER_MAX_SEQ_LEN]"""

            dict_encoded_mut_seqs = self.model(
                batch_tokens, repr_layers=list(range(self._max_emb_layer + 1))
            )["representations"]

        for layer, encoded_mut_seqs in dict_encoded_mut_seqs.items():

            encoded_mut_seqs = encoded_mut_seqs.cpu().numpy()
            # https://github.com/facebookresearch/esm/blob/main/esm/data.py
            # from_architecture

            # trim off initial classification token [CLS]
            # both "ESM-1" and "ESM-1b" have prepend_bos = True
            if self._iftrimCLS and self._encoder_name.split("_")[0] in [
                "esm1",
                "esm1b",
            ]:
                encoded_mut_seqs = encoded_mut_seqs[:, 1:, :]

            # trim off end-of-sequence token [EOS]
            # only "ESM-1b" has append_eos = True
            if self._iftrimEOS and self._encoder_name.split("_")[0] == "esm1b":
                encoded_mut_seqs = encoded_mut_seqs[:, :-1, :]

            if mut_names is not None:
                dict_encoded_mut_seqs[layer] = (
                    self.flatten_encode(
                        encoded_mut_seqs=encoded_mut_seqs,
                        flatten_emb=flatten_emb,
                        mut_seqs=mut_seqs,
                    ),
                    batch_labels,
                )
            else:
                dict_encoded_mut_seqs[layer] = self.flatten_encode(
                    encoded_mut_seqs=encoded_mut_seqs,
                    flatten_emb=flatten_emb,
                    mut_seqs=mut_seqs,
                )

        return dict_encoded_mut_seqs


class CARPEncoder(AbstractEncoder):
    """
    Build a CARP encoder
    """

    def __init__(
        self,
        encoder_name: str,
        reset_param: bool = False,
        resample_param: bool = False,
    ):
        """
        Args
        - encoder_name: str, the name of the encoder, one of the keys of CARP_INFO
        - reset_param: bool = False, if update the full model to xavier_uniform_
        - resample_param: bool = False, if update the full model to xavier_normal_
        """

        super().__init__(encoder_name, reset_param, resample_param)

        self.model, self.collater = load_model_and_alphabet(self._encoder_name)

        # if reset or resample weights
        self.model = self.reset_resample_param(model=self.model)

        # set model to eval mode
        self.model.eval()
        self.model.to(DEVICE)

        self._embed_dim, self._max_emb_layer = CARP_INFO[self._encoder_name]

        # carp does not have the input representation
        self._include_input_layer = False

        # load model from torch.hub
        print(
            f"Generating {self._encoder_name} upto {self._max_emb_layer} layer embedding ..."
        )

    def _encode_batch(
        self,
        mut_seqs: Sequence[str] | str,
        flatten_emb: bool | str,
        mut_names: Sequence[str] | str | None = None,
    ) -> np.ndarray:
        """
        Encodes a batch of mutant sequences.

        Args:
        - mut_seqs: list of str or str, mutant sequences of the same length
        - flatten_emb: bool or str, if and how (one of ["max", "mean"]) to flatten the embedding
        - mut_names: list of str or str or None, mutant names

        Returns:
        - np.ndarray or a tuple(np.ndarray, list[str]) where the list is batch_labels
        """

        mut_seqs = [[m] for m in mut_seqs]

        x = self.collater(mut_seqs)[0]

        # alternatively check out the article called:
        # The One PyTorch Trick Which You Should Know
        # How hooks can improve your workflow significantly

        activation = {}

        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()

            return hook

        # convert raw mutant sequences to tokens
        for layer_numb in list(range(self._max_emb_layer)):
            self.model.model.embedder.layers[layer_numb].register_forward_hook(
                get_activation(layer_numb)
            )

        rep = self.model(x)

        for layer_numb, encoded_mut_seqs in activation.items():
            activation[layer_numb] = self.flatten_encode(
                encoded_mut_seqs=encoded_mut_seqs.cpu().numpy(),
                flatten_emb=flatten_emb,
                mut_seqs=mut_seqs,
            )

        return activation


def get_emb_info(encoder_name: str) -> Collection(str, AbstractEncoder, int):

    """
    A function return processed encoder_name and total_emb_layer

    Args:
    - encoder_name: str, input encoder_name

    Returns:
    - encoder_name: str, change anything not a transformer or carp encoder to onehot
    - encoder_class: AbstractEncoder, encoder class
    - total_emb_layer: int, number of embedding layers
    """

    if encoder_name in TRANSFORMER_INFO.keys():
        total_emb_layer = TRANSFORMER_INFO[encoder_name][1] + 1
        encoder_class = ESMEncoder
    elif encoder_name in CARP_INFO.keys():
        total_emb_layer = CARP_INFO[encoder_name][1]
        encoder_class = CARPEncoder
    else:
        # for onehot
        encoder_name = "onehot"
        encoder_class = OnehotEncoder
        total_emb_layer = 1

    return encoder_name, encoder_class, total_emb_layer