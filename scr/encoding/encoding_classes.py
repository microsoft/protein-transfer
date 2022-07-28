"""Add encoding classes with class methods"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Iterable, Sequence

import numpy as np
from tqdm import tqdm

import torch
from torch.nn.init import xavier_uniform_, xavier_normal_
from sequence_models.pretrained import load_model_and_alphabet

from scr.params.aa import AA_NUMB, AA_TO_IND
from scr.params.emb import TRANSFORMER_INFO, TRANSFORMER_MAX_SEQ_LEN, CARP_INFO
from scr.params.sys import DEVICE


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

            for p in model.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)

        elif self._resample_param:
            print(f"Resample params for {self._encoder_name} ...")

            for p in model.parameters():
                if p.dim() > 1:
                    xavier_normal_(p)

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
        self, encoded_mut_seqs: np.ndarray, flatten_emb: bool | str
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
        assert encoded_mut_seqs.shape[2] == self._embed_dim, "Wrong embed dim"

        if (
            flatten_emb in [True, "flatten", "flattened", ""]
            or self._encoder_name == "onehot"
        ):
            # shape [batch_size, seq_len * embed_dim]
            return encoded_mut_seqs.reshape(encoded_mut_seqs.shape[0], -1)

        elif isinstance(flatten_emb, str):
            if flatten_emb == "mean":
                # [batch_size, embed_dim]
                return encoded_mut_seqs.mean(axis=1)
            elif flatten_emb == "max":
                # [batch_size, embed_dim]
                return encoded_mut_seqs.max(axis=1)

        else:
            print("No embedding flattening")
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
        encoder_name: str = "",
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

        if encoder_name not in (TRANSFORMER_INFO.keys() and CARP_INFO.keys()):
            self._encoder_name = "onehot"
            self._embed_dim, self._max_emb_layer = AA_NUMB, 0
            self._include_input_layer = False

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
            encoded_mut_seqs.append(np.eye(AA_NUMB)[[AA_TO_IND[aa] for aa in mut_seq]])
        return {0: self.flatten_encode(np.array(encoded_mut_seqs), flatten_emb)}


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
            if batch_tokens.shape[1] > TRANSFORMER_MAX_SEQ_LEN:
                print(f"Sequence exceeds {TRANSFORMER_MAX_SEQ_LEN}, chopping the end")
                batch_tokens = batch_tokens[:, :TRANSFORMER_MAX_SEQ_LEN]

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
                    self.flatten_encode(encoded_mut_seqs, flatten_emb),
                    batch_labels,
                )
            else:
                dict_encoded_mut_seqs[layer] = self.flatten_encode(
                    encoded_mut_seqs, flatten_emb
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
                encoded_mut_seqs.cpu().numpy(), flatten_emb
            )

        return activation