"""Script for running pytorch models"""

from __future__ import annotations

import os
from concurrent import futures
from functools import partial

import torch
import torch.nn as nn

from sklearn.metrics import ndcg_score
from scipy.stats import spearmanr

from scr.params.sys import RAND_SEED, DEVICE
from scr.params.emb import TRANSFORMER_INFO

from scr.preprocess.data_process import split_protrain_loader
from scr.encoding.encoding_classes import get_emb_info
from scr.model.pytorch_model import LinearRegression
from scr.model.train_test import train, test
from scr.vis.learning_vis import plot_lc
from scr.utils import get_folder_file_names, pickle_save, get_default_output_path


def run_pytorch(
    dataset_path: str,
    encoder_name: str,
    reset_param: bool = False,
    resample_param: bool = False,
    embed_batch_size: int = 128,
    flatten_emb: bool | str = False,
    embed_folder: str | None = None,
    seq_start_idx: bool | int = False,
    seq_end_idx: bool | int = False,
    loader_batch_size: int = 64,
    worker_seed: int = RAND_SEED,
    if_encode_all: bool = True,
    learning_rate: float = 1e-4,
    lr_decay: float = 0.1,
    epochs: int = 100,
    early_stop: bool = True,
    tolerance: int = 10,
    min_epoch: int = 5,
    device: torch.device | str = DEVICE,
    all_plot_folder: str = "results/learning_curves",
    all_result_folder: str = "results/train_val_test",
    **encoder_params,
) -> dict:

    """
    A function for running pytorch model

    Args:
    - dataset_path: str, full path to the dataset, in pkl or panda readable format
        columns include: sequence, target, set, validation, mut_name (optional), mut_numb (optional)
    - encoder_name: str, the name of the encoder
    
    - embed_batch_size: int, set to 0 to encode all in a single batch
    - flatten_emb: bool or str, if and how (one of ["max", "mean"]) to flatten the embedding
    - embed_folder: str = None, path to presaved embedding
    - seq_start_idx: bool | int = False, the index for the start of the sequence
    - seq_end_idx: bool | int = False, the index for the end of the sequence
    - loader_batch_size: int, the batch size for train, val, and test dataloader
    - worker_seed: int, the seed for dataloader
    - learning_rate: float
    - lr_decay: float, factor by which to decay LR on plateau
    - epochs: int, number of epochs to train for
    - device: torch.device or str
    - all_plot_folder: str, the parent folder path for saving all the learning curves
    - all_result_folder: str = "results/train_val_test", the parent folder for all results
    - encoder_params: kwarg, additional parameters for encoding

    Returns:
    - result_dict: dict, with the keys and dict values
        "losses": {"train_losses": np.ndarray, "val_losses": np.ndarray}
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

    train_loader, val_loader, test_loader = split_protrain_loader(
        dataset_path=dataset_path,
        encoder_name=encoder_name,
        reset_param=reset_param,
        resample_param=resample_param,
        embed_batch_size=embed_batch_size,
        flatten_emb=flatten_emb,
        embed_folder=embed_folder,
        seq_start_idx=seq_start_idx,
        seq_end_idx=seq_end_idx,
        subset_list=["train", "val", "test"],
        loader_batch_size=loader_batch_size,
        worker_seed=worker_seed,
        if_encode_all=if_encode_all,
        **encoder_params,
    )

    encoder_name, encoder_class, total_emb_layer = get_emb_info(encoder_name)

    # for each layer train the model and save the model
    for embed_layer in range(total_emb_layer):

        model = LinearRegression(
            input_dim=TRANSFORMER_INFO[encoder_name][0], output_dim=1
        )
        model.to(device, non_blocking=True)

        criterion = nn.MSELoss()
        criterion.to(device, non_blocking=True)

        train_losses, val_losses = train(
            model=model,            
            criterion=criterion,
            train_loader=train_loader,
            val_loader=val_loader,
            encoder_name=encoder_name,
            embed_layer=embed_layer,
            reset_param=reset_param,
            resample_param=resample_param,
            embed_batch_size=embed_batch_size,
            flatten_emb=flatten_emb,
            device=device,
            learning_rate=learning_rate,
            lr_decay=lr_decay,
            epochs=epochs,
            early_stop=early_stop,
            tolerance=tolerance,
            min_epoch=min_epoch,
            **encoder_params
        )

        # record the losses
        result_dict = {
            "losses": {"train_losses": train_losses, "val_losses": val_losses}
        }

        plot_lc(
            train_losses=train_losses,
            val_losses=val_losses,
            dataset_path=dataset_path,
            encoder_name=encoder_name,
            embed_layer=embed_layer,
            flatten_emb=flatten_emb,
            all_plot_folder=all_plot_folder,
        )

        # now test the model with the test data
        for subset, loader in zip(
            ["train", "val", "test"], [train_loader, val_loader, test_loader]
        ):
            mse, pred, true = test(
                model=model, loader=loader, device=device, criterion=criterion
            )

            result_dict[subset] = {
                "mse": mse,
                "pred": pred,
                "true": true,
                "ndcg": ndcg_score(true[None, :], pred[None, :]),
                "rho": spearmanr(true, pred),
            }

        dataset_subfolder, file_name = get_folder_file_names(
            parent_folder=get_default_output_path(all_result_folder),
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