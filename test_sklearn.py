from scr.params.sys import SKLEARN_ALPHAS, RAND_SEED
from scr.params.emb import TRANSFORMER_INFO
from scr.model.run_sklearn import RunSK

"""
for ds_split in ["low_vs_high", "sampled", "two_vs_rest"]:
    # for encoder_name in ["onehot"]:
    for encoder_name in ["onehot"] + list(TRANSFORMER_INFO.keys()):
        if encoder_name == "onehot":
            embed_batch_size = 0
            flatten_emb = "flatten"
        else:
            embed_batch_size = 64
            flatten_emb = "mean"

        RunSK(
            dataset_path=f"data/proeng/gb1/{ds_split}.csv",
            encoder_name=encoder_name,
            reset_param=False,
            resample_param=False,
            embed_batch_size=embed_batch_size,
            flatten_emb=flatten_emb,
            embed_folder=None,
            all_embed_layers=True,
            seq_start_idx=0,
            seq_end_idx=56,
            if_encode_all=True,
            alphas=SKLEARN_ALPHAS,
            sklearn_state=RAND_SEED,
            # sklearn_params={"normalize":True},
            all_result_folder="results/sklearn-scaley-noloader-fixembpool",
            #**encoder_params,
        )
        """

for encoder_name in TRANSFORMER_INFO.keys():
    RunSK(
        dataset_path="data/proeng/thermo/mixed_split.csv",
        encoder_name=encoder_name,
        reset_param= False,
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
