from re import T
from scr.params.sys import SKLEARN_ALPHAS, RAND_SEED
from scr.params.emb import TRANSFORMER_INFO
from scr.model.run_sklearn import RunRidge

for encoder_name in TRANSFORMER_INFO.keys():
    RunRidge(
        dataset_path="data/proeng/thermo/mixed_split.csv",
        encoder_name=encoder_name,
        reset_param= True,
        resample_param = False,
        embed_batch_size = 128,
        flatten_emb = "mean",
        embed_folder = "embeddings-rand/proeng/thermo/mixed_split",
        all_embed_layers=False,
        seq_start_idx = False,
        seq_end_idx = False,
        if_encode_all=False,
        alphas = SKLEARN_ALPHAS,
        ridge_state= RAND_SEED,
        ridge_params= None,
        all_result_folder = "test/sklearn",
        # **encoder_params,
    )

for encoder_name in TRANSFORMER_INFO.keys():
    RunRidge(
        dataset_path="data/proeng/thermo/mixed_split.csv",
        encoder_name=encoder_name,
        reset_param= False,
        resample_param = True,
        embed_batch_size = 128,
        flatten_emb = "mean",
        embed_folder = "embeddings-stat/proeng/thermo/mixed_split",
        all_embed_layers=False,
        seq_start_idx = False,
        seq_end_idx = False,
        if_encode_all=False,
        alphas = SKLEARN_ALPHAS,
        ridge_state= RAND_SEED,
        ridge_params= None,
        all_result_folder = "test/sklearn",
        # **encoder_params,
    )