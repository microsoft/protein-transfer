"""Script for run sklearn (currently ridge) models"""

from __future__ import annotations

import os
import random
import numpy as np

from sklearn.linear_model import Ridge
from sklearn.metrics import ndcg_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

from torch.utils.data import DataLoader

from scr.utils import get_folder_file_names, pickle_save
from scr.params.sys import RAND_SEED, SKLEARN_ALPHAS
from scr.params.emb import TRANSFORMER_INFO, CARP_INFO
from scr.encoding.encoding_classes import ESMEncoder, CARPEncoder
from scr.preprocess.data_process import split_protrain_loader

# seed 
random.seed(RAND_SEED)
np.random.seed(RAND_SEED)


def sk_test(model: sklearn.linear_model, loader: DataLoader, embed_layer: int):
    """
    A function for testing sklearn models for a specific layer of embeddings

    Args:
    - model: sklearn.linear_model, trained model
    - loader: DataLoader, train, val, or test data loader
    - embed_layer: int, specific layer of the embedding

    Returns:
    - np.concatenate(pred): np.ndarray, 1D predicted fitness values
    - np.concatenate(true): np.ndarry, 1D true fitness values
    """
    pred = []
    true = []
    for (y, sequence, mut_name, mut_numb, *layer_emb) in loader:
        pred.append(model.predict(layer_emb[embed_layer]).squeeze())
        true.append(y.cpu().squeeze().numpy())
    return np.concatenate(pred), np.concatenate(true)


def pick_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    embed_layer: int,
    alphas: np.ndarray,
    ridge_state: int = RAND_SEED,
    ridge_params: dict | None = None,
):
    """
    A function for picking the best model for given alaphs, meaning
    lower train_mse and higher test_ndcg
    NOTE: alphas tuning is NOT currently optimal

    Args:
    - train_loader: DataLoader, dataloader with encoded sequence of different layers
    - val_loader: DataLoader, dataloader with encoded sequence of different layers
    - embed_layer: int, specific layer of the embedding
    - alphas: np.ndarray, arrays of alphas to be tested
    - ridge_state: int = RAND_SEED, seed the ridge regression
    - ridge_params: dict | None = None, other ridge regression args

    Returns:
    - sklearn.linear_model, the model with the best alpha
    """

    # init values for comparison
    best_mse = np.Inf
    best_ndcg = -1
    best_rho = -1
    best_model = None

    # loop through all alphas
    for alpha in alphas:

        # init model for each alpha
        if ridge_params is None:
            ridge_params = {}
        model = Ridge(alpha=alpha, random_state=ridge_state, **ridge_params)

        # fit the model for a given layer of embedding
        for (y, sequence, mut_name, mut_numb, *layer_emb) in train_loader:
            model.fit(layer_emb[embed_layer], y)

        # eval the model with train and test
        train_pred, train_true = sk_test(model, train_loader, embed_layer=embed_layer)
        val_pred, val_true = sk_test(model, val_loader, embed_layer=embed_layer)

        # calc the metrics
        train_mse = mean_squared_error(train_true, train_pred)
        val_ndcg = ndcg_score(val_true[None, :], val_pred[None, :])
        val_rho = spearmanr(val_true, val_pred)[0]

        # update the model if it has lower train_mse and higher val_ndcg
        if train_mse < best_mse and val_ndcg > best_ndcg:
            best_model = model
            best_mse = train_mse
            best_ndcg = val_ndcg
            best_rho = val_rho

    return best_model


def run_ridge_layer(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    dataset_path: str,
    encoder_name: str,
    embed_layer: int,
    flatten_emb: bool | str = False,
    alphas: np.ndarray = SKLEARN_ALPHAS,
    ridge_state: int = RAND_SEED,
    ridge_params: dict | None = None,
    all_result_folder: str = "results/sklearn",
):

    """
    A function for running ridge regression for a given layer of embedding

    Args:
    - train_loader: DataLoader, dataloader with encoded sequence of different layers
    - val_loader: DataLoader, dataloader with encoded sequence of different layers
    - test_loader: DataLoader, dataloader with encoded sequence of different layers
    - dataset_path: str, full path to the dataset, in pkl or panda readable format
        columns include: sequence, target, set, validation,
        mut_name (optional), mut_numb (optional)
    - encoder_name: str, the name of the encoder
    - embed_layer: int, specific layer of the embedding
    - flatten_emb: bool or str, if and how (one of ["max", "mean"]) to flatten the embedding
    - alphas: np.ndarray, arrays of alphas to be tested
    - ridge_state: int = RAND_SEED, seed the ridge regression
    - ridge_params: dict | None = None, other ridge regression args
    - all_result_folder: str = "results/sklearn", the parent folder for all results

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
    best_model = pick_model(
        train_loader=train_loader,
        val_loader=val_loader,
        embed_layer=embed_layer,
        alphas=alphas,
        ridge_state=ridge_state,
        ridge_params=ridge_params,
    )

    # init dict for resulted outputs
    result_dict = {}

    # now test the model with the test data
    for subset, loader in zip(
        ["train", "val", "test"], [train_loader, val_loader, test_loader]
    ):
        pred, true = sk_test(best_model, val_loader, embed_layer=embed_layer)

        result_dict[subset] = {
            "mse": mean_squared_error(true, pred),
            "pred": pred,
            "true": true,
            "ndcg": ndcg_score(true[None, :], pred[None, :]),
            "rho": spearmanr(true, pred),
        }

    dataset_subfolder, file_name = get_folder_file_names(
        parent_folder=all_result_folder,
        dataset_path=dataset_path,
        encoder_name=encoder_name,
        embed_layer=embed_layer,
        flatten_emb=flatten_emb,
    )

    print(f"Saving results for {file_name} to: {dataset_subfolder}...")
    pickle_save(
        what2save=result_dict,
        where2save=os.path.join(dataset_subfolder, file_name + ".pkl"),
    )

    return result_dict

def run_ridge(
    dataset_path: str,
    encoder_name: str,
    embed_batch_size: int = 128,
    flatten_emb: bool | str = False,
    embed_path: str | None = None,
    seq_start_idx: bool | int = False,
    seq_end_idx: bool | int = False,
    loader_batch_size: int = 64,
    worker_seed: int = RAND_SEED,
    alphas: np.ndarray = SKLEARN_ALPHAS,
    ridge_state: int = RAND_SEED,
    ridge_params: dict | None = None,
    all_result_folder: str = "results/sklearn",
    **encoder_params,
) -> dict:

    """
    A function for running sklearn model

    Args:
    - dataset_path: str, full path to the dataset, in pkl or panda readable format
        columns include: sequence, target, set, validation, 
        mut_name (optional), mut_numb (optional)
    - encoder_name: str, the name of the encoder
    - embed_batch_size: int, set to 0 to encode all in a single batch
    - flatten_emb: bool or str, if and how (one of ["max", "mean"]) to flatten the embedding
    - embed_path: str = None, path to presaved embedding
    - seq_start_idx: bool | int = False, the index for the start of the sequence
    - seq_end_idx: bool | int = False, the index for the end of the sequence
    - loader_batch_size: int, the batch size for train, val, and test dataloader
    - worker_seed: int, the seed for dataloader
    - alphas: np.ndarray, arrays of alphas to be tested
    - ridge_state: int = RAND_SEED, seed the ridge regression
    - ridge_params: dict | None = None, other ridge regression args
    - all_result_folder: str = "results/train_val_test", the parent folder for all results
    - encoder_params: kwarg, additional parameters for encoding

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

    # loader has ALL embedding layers
    train_loader, val_loader, test_loader = split_protrain_loader(
        dataset_path=dataset_path,
        encoder_name=encoder_name,
        embed_batch_size=embed_batch_size,
        flatten_emb=flatten_emb,
        embed_path=embed_path,
        seq_start_idx=seq_start_idx,
        seq_end_idx=seq_end_idx,
        subset_list=["train", "val", "test"],
        loader_batch_size=loader_batch_size,
        worker_seed=worker_seed,
        **encoder_params,
    )

    all_ridge_results = {}

    if encoder_name in TRANSFORMER_INFO.keys():
        total_emb_layer = ESMEncoder(encoder_name=encoder_name).total_emb_layer
    elif encoder_name in CARP_INFO.keys():
        total_emb_layer = CARPEncoder(encoder_name=encoder_name).total_emb_layer

    for layer in range(total_emb_layer):
        all_ridge_results[layer] = run_ridge_layer(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            dataset_path=dataset_path,
            encoder_name=encoder_name,
            embed_layer=layer,
            flatten_emb=flatten_emb,
            alphas=alphas,
            ridge_state=ridge_state,
            ridge_params=ridge_params,
            all_result_folder=all_result_folder,
        )

    return all_ridge_results