"""Add encoding classes with class methods"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence

import numpy as np
from tqdm import tqdm

import torch

from scr.params.emb import TRANSFORMER_INFO, TRANSFORMER_MAX_SEQ_LEN
from scr.params.train_test import DEVICE
from scr.preprocess.seq_loader import SeqLoader


class AbstractEncoder(ABC):
    """
    An abstract encoder class to fill in for different kinds of encoders

    All encoders will have an "encode" function
    """

    def __init__(self, encoder_name: str, embed_layer: int):

        """
        Args:
        - encoder_name: str, the name of the encoder
        - embed_layer: int, the layer number of the embedding
        """

        self._encoder_name = encoder_name
        self._embed_layer = embed_layer

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
        """

        if isinstance(mut_seqs, str):
            mut_seqs = [mut_seqs]

        # If the batch size is 0, then encode all at once in a single batch
        if batch_size == 0:
            yield self._encode_batch(mut_seqs=mut_seqs, flatten_emb=flatten_emb, mut_names=mut_names)

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

    @abstractmethod
    def _flatten_encode(
        self, mut_seqs: np.ndarray, flatten_emb: bool | str
    ) -> np.ndarray:
        """
        Flatten the embedding.

        Args
        - mut_seqs: np.ndarray, shape [batch_size, seq_len, embed_dim]
        - flatten_emb: bool or str, if and how (one of ["max", "mean"]) to flatten the embedding
        """
        pass

    @property
    def embed_dim(self) -> int:
        """The dim of the embedding"""
        return self._embed_dim

    @property
    def embed_layer(self) -> int:
        """The layer nubmer of the embedding"""
        return self._embed_layer

    @property
    def encoder_name(self) -> str:
        """The name of the encoding method"""
        return self._encoder_name


class ESMEncoder(AbstractEncoder):
    """
    Build an ESM encoder
    """

    def __init__(
        self,
        encoder_name: str,
        embed_layer: int,
        iftrimCLS: bool = True,
        iftrimEOS: bool = True,
    ):
        """
        Args
        - encoder_name: str, the name of the encoder, one of the keys of TRANSFORMER_INFO
        - embed_layer: int, the layer number of the embedding
        - iftrimCLS: bool, whether to trim the first classifification token
        - iftrimEOS: bool, whether to trim the end of sequence token, if exists
        """

        super().__init__(encoder_name, embed_layer)

        self._iftrimCLS = iftrimCLS
        self._iftrimEOS = iftrimEOS

        # load model from torch.hub
        print(f"Loading {self._encoder_name} using {self._embed_layer} layer embedding")
        self.model, self.alphabet = torch.hub.load(
            "facebookresearch/esm:main", model=self._encoder_name
        )
        self.batch_converter = self.alphabet.get_batch_converter()

        # set model to eval mode
        self.model.eval()
        self.model.to(DEVICE)

        self._embed_dim, self._max_emb_layer, _ = TRANSFORMER_INFO[self._encoder_name]

        assert (
            self._embed_layer <= self._max_emb_layer
        ), f"{self._embed_layer} exceeds {self._max_emb_layer}"

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

            encoded_mut_seqs = (
                self.model(batch_tokens, repr_layers=[self._embed_layer])[
                    "representations"
                ][self._embed_layer]
                .cpu()
                .numpy()
            )

        # https://github.com/facebookresearch/esm/blob/main/esm/data.py
        # from_architecture

        # trim off initial classification token [CLS]
        # both "ESM-1" and "ESM-1b" have prepend_bos = True
        if self._iftrimCLS and self._encoder_name.split("_")[0] in ["esm1", "esm1b"]:
            encoded_mut_seqs = encoded_mut_seqs[:, 1:, :]

        # trim off end-of-sequence token [EOS]
        # only "ESM-1b" has append_eos = True
        if self._iftrimEOS and self._encoder_name.split("_")[0] == "esm1b":
            encoded_mut_seqs = encoded_mut_seqs[:, :-1, :]

        if mut_names is not None:
            return self._flatten_encode(encoded_mut_seqs, flatten_emb), batch_labels
        else:
            return self._flatten_encode(encoded_mut_seqs, flatten_emb)

    def _flatten_encode(
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

        if flatten_emb is True:
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