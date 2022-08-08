"""Script for run sklearn (currently ridge) models"""

from __future__ import annotations

import os
import random
import numpy as np

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

from scr.utils import get_folder_file_names, pickle_save, ndcg_scale
from scr.params.emb import TRANSFORMER_INFO, CARP_INFO
from scr.params.sys import RAND_SEED, SKLEARN_ALPHAS
from scr.encoding.encoding_classes import ESMEncoder, CARPEncoder, OnehotEncoder
from scr.preprocess.data_process import ProtranDataset

# seed
random.seed(RAND_SEED)
np.random.seed(RAND_SEED)

class RunRidge:
    """A class for running ridge regression"""

    def __init__(
        self,
        dataset_path: str,
        encoder_name: str,
        reset_param: bool = False,
        resample_param: bool = False,
        embed_batch_size: int = 128,
        flatten_emb: bool | str = False,
        embed_path: str | None = None,
        seq_start_idx: bool | int = False,
        seq_end_idx: bool | int = False,
        alphas: np.ndarray | int = SKLEARN_ALPHAS,
        ridge_state: int = RAND_SEED,
        ridge_params: dict | None = None,
        all_result_folder: str = "results/sklearn",
        **encoder_params,
    ) -> None:

        """
        Args:
        - dataset_path: str, full path to the dataset, in pkl or panda readable format
            columns include: sequence, target, set, validation,
            mut_name (optional), mut_numb (optional)
        - encoder_name: str, the name of the encoder
        - reset_param: bool = False, if update the full model to xavier_uniform_
        - resample_param: bool = False, if update the full model to xavier_normal_
        - embed_batch_size: int, set to 0 to encode all in a single batch
        - flatten_emb: bool or str, if and how (one of ["max", "mean"]) to flatten the embedding
        - embed_path: str = None, path to presaved embedding
        - seq_start_idx: bool | int = False, the index for the start of the sequence
        - seq_end_idx: bool | int = False, the index for the end of the sequence
        - alphas: np.ndarray, arrays of alphas to be tested
        - ridge_state: int = RAND_SEED, seed the ridge regression
        - ridge_params: dict | None = None, other ridge regression args
        - all_result_folder: str = "results/train_val_test", the parent folder for all results
        - encoder_params: kwarg, additional parameters for encoding
        """

        self.dataset_path = dataset_path
        self.encoder_name = encoder_name
        self.reset_param = reset_param
        self.resample_param = resample_param
        self.flatten_emb = flatten_emb

        if not isinstance(alphas, np.ndarray):
            alphas = np.array([alphas])
        self.alphas = alphas

        self.ridge_state = ridge_state
        self.ridge_params = ridge_params
        self.all_result_folder = all_result_folder

        if self.reset_param and "-rand" not in self.all_result_folder:
            self.all_result_folder = f"{self.all_result_folder}-rand"

        if self.resample_param and "-stat" not in self.all_result_folder:
            self.all_result_folder = f"{self.all_result_folder}-stat"

        # loader has ALL embedding layers
        self.train_ds, self.val_ds, self.test_ds = (
            ProtranDataset(
                dataset_path=dataset_path,
                subset=subset,
                encoder_name=encoder_name,
                reset_param=reset_param,
                resample_param=resample_param,
                embed_batch_size=embed_batch_size,
                flatten_emb=flatten_emb,
                embed_path=embed_path,
                seq_start_idx=seq_start_idx,
                seq_end_idx=seq_end_idx,
                **encoder_params,
            )
            for subset in ["train", "val", "test"]
        )

        all_ridge_results = {}

        if self.encoder_name in TRANSFORMER_INFO.keys():
            total_emb_layer = TRANSFORMER_INFO[encoder_name][1] + 1
        elif self.encoder_name in CARP_INFO.keys():
            total_emb_layer = CARP_INFO[encoder_name][1]
        else:
            # for onehot
            self.encoder_name = "onehot"
            total_emb_layer = 1

        """total_emb_layer = encoder_class(
                encoder_name=self.encoder_name,
                reset_param=reset_param,
                resample_param=resample_param,
                **encoder_params
            ).total_emb_layer"""

        for layer in range(total_emb_layer):
            all_ridge_results[layer] = self.run_ridge_layer(
                embed_layer=layer,
            )

        self._all_ridge_results = all_ridge_results

    def sk_test(
        self, model: sklearn.linear_model, ds: ProtranDataset, embed_layer: int
    ):
        """
        A function for testing sklearn models for a specific layer of embeddings

        Args:
        - model: sklearn.linear_model, trained model
        - ds: ProtranDataset, train, val, or test dataset
        - embed_layer: int, specific layer of the embedding

        Returns:
        - np.concatenate(pred): np.ndarray, 1D predicted fitness values
        - np.concatenate(true): np.ndarry, 1D true fitness values
        """

        return (
            model.predict(
                getattr(ds, "layer" + str(embed_layer))
            ).squeeze(),
            ds.y.squeeze(),
        )

    def pick_model(
        self,
        embed_layer: int,
    ):
        """
        A function for picking the best model for given alaphs, meaning
        lower train_mse and higher test_ndcg
        NOTE: alphas tuning is NOT currently optimal

        Args:
        - embed_layer: int, specific layer of the embedding

        Returns:
        - sklearn.linear_model, the model with the best alpha
        """

        # init values for comparison
        best_mse = np.Inf
        best_ndcg = -1
        best_rho = -1
        best_model = None

        # loop through all alphas
        for alpha in self.alphas:

            # init model for each alpha
            if self.ridge_params is None:
                self.ridge_params = {}
            model = Ridge(
                alpha=alpha, random_state=self.ridge_state, **self.ridge_params
            )

            # fit the model for a given layer of embedding
            fitness_scaler = StandardScaler()
            model.fit(
                getattr(self.train_ds, "layer" + str(embed_layer)),
                fitness_scaler.fit_transform(self.train_ds.y),
            )

            # eval the model with train and test
            train_pred, train_true = self.sk_test(
                model, self.train_ds, embed_layer=embed_layer
            )
            val_pred, val_true = self.sk_test(
                model, self.val_ds, embed_layer=embed_layer
            )

            # calc the metrics
            train_mse = mean_squared_error(train_true, train_pred)
            val_ndcg = ndcg_scale(val_true, val_pred)
            val_rho = spearmanr(val_true, val_pred)[0]

            # update the model if it has lower train_mse and higher val_ndcg
            if train_mse < best_mse and val_ndcg > best_ndcg:
                best_model = model
                best_mse = train_mse
                best_ndcg = val_ndcg
                best_rho = val_rho

        print(f"best model is {best_model}")
        return best_model

    def run_ridge_layer(
        self,
        embed_layer: int,
    ):

        """
        A function for running ridge regression for a given layer of embedding

        Args:
        - embed_layer: int, specific layer of the embedding

        Returns:
        - dict, with the keys and dict values
            "train": {"mse": float,
                    "pred": np.ndarray,
                    "true": np.ndarray,
                    "ndcg": float,
                    "rho": SpearmanrResults(correlation=float, pvalue=float)}
            "val":   {"mse": float,
                    "pred": np.ndarray,
                    "true": np.ndarray,
                    "ndcg": float,
                    "rho": SpearmanrResults(correlation=float, pvalue=float)}
            "test":  {"mse": float,
                    "pred": np.ndarray,
                    "true": np.ndarray,
                    "ndcg": float,
                    "rho": SpearmanrResults(correlation=float, pvalue=float)}
        """

        # train and get the best alpha
        best_model = self.pick_model(
            embed_layer=embed_layer,
        )

        # init dict for resulted outputs
        result_dict = {}

        # now test the model with the test data
        for subset, ds in zip(
            ["train", "val", "test"],
            [self.train_ds, self.val_ds, self.test_ds],
        ):
            pred, true = self.sk_test(best_model, ds, embed_layer=embed_layer)

            result_dict[subset] = {
                "mse": mean_squared_error(true, pred),
                "pred": pred,
                "true": true,
                "ndcg": ndcg_scale(true, pred),
                "rho": spearmanr(true, pred),
            }

        dataset_subfolder, file_name = get_folder_file_names(
            parent_folder=self.all_result_folder,
            dataset_path=self.dataset_path,
            encoder_name=self.encoder_name,
            embed_layer=embed_layer,
            flatten_emb=self.flatten_emb,
        )

        print(f"Saving results for {file_name} to: {dataset_subfolder}...")
        pickle_save(
            what2save=result_dict,
            where2save=os.path.join(dataset_subfolder, file_name + ".pkl"),
        )

        return result_dict

    @property
    def all_ridge_results(self):
        """
        Returns:
        - dict, with the keys and dict values
            "layer#": {
                        "train": {"mse": float,
                                "pred": np.ndarray,
                                "true": np.ndarray,
                                "ndcg": float,
                                "rho": SpearmanrResults(correlation=float, pvalue=float)}
                        "val":   {"mse": float,
                                "pred": np.ndarray,
                                "true": np.ndarray,
                                "ndcg": float,
                                "rho": SpearmanrResults(correlation=float, pvalue=float)}
                        "test":  {"mse": float,
                                "pred": np.ndarray,
                                "true": np.ndarray,
                                "ndcg": float,
                                "rho": SpearmanrResults(correlation=float, pvalue=float)}
                        }
        """
        return self._all_ridge_results