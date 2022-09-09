from scr.params.sys import SKLEARN_ALPHAS, RAND_SEED
from scr.params.emb import TRANSFORMER_INFO
from scr.model.run_sklearn import RunSK

for ds_split in ["low_vs_high", "sampled", "two_vs_rest"]:
    for encoder_name in list(TRANSFORMER_INFO.keys()):
        RunSK(
            dataset_path=f"data/proeng/gb1/{ds_split}.csv",
            encoder_name=encoder_name,
            reset_param=False,
            resample_param=True,
            embed_batch_size=64,
            flatten_emb="mean",
            embed_folder=f"/home/t-fli/amlt/genemb-stat-shuffleall/gb1-{ds_split}-{encoder_name}-mean-stat/embeddings-stat/proeng/gb1/{ds_split}",
            all_embed_layers=False,
            seq_start_idx=0,
            seq_end_idx=56,
            if_encode_all=False,
            alphas=SKLEARN_ALPHAS,
            sklearn_state=RAND_SEED,
            sklearn_params={"normalize":True},
            all_result_folder="results/sklearn-scalex-scaley-noloader-fixembpool",
            #**encoder_params,
        )

for ds_split in ["one_vs_many", "two_vs_many"]:
    for encoder_name in list(TRANSFORMER_INFO.keys()):
        RunSK(
            dataset_path=f"data/proeng/aav/{ds_split}.csv",
            encoder_name=encoder_name,
            reset_param=False,
            resample_param=True,
            embed_batch_size=64,
            flatten_emb="mean",
            embed_folder=f"/home/t-fli/amlt/genemb-stat-shuffleall/aav-{ds_split}-{encoder_name}-mean-stat/embeddings-stat/proeng/aav/{ds_split}",
            all_embed_layers=False,
            seq_start_idx=False,
            seq_end_idx=False,
            if_encode_all=False,
            alphas=SKLEARN_ALPHAS,
            sklearn_state=RAND_SEED,
            sklearn_params={"normalize":True},
            all_result_folder="results/sklearn-scalex-scaley-noloader-fixembpool",
            #**encoder_params,
        )

for ds_split in ["mixed_split"]:
    for encoder_name in list(TRANSFORMER_INFO.keys()):
        RunSK(
            dataset_path=f"data/proeng/thermo/{ds_split}.csv",
            encoder_name=encoder_name,
            reset_param=False,
            resample_param=True,
            embed_batch_size=64,
            flatten_emb="mean",
            embed_folder=f"/home/t-fli/amlt/genemb-stat-shuffleall/thermo-{encoder_name}-mean-stat/embeddings-stat/proeng/thermo/mixed_split",
            all_embed_layers=False,
            seq_start_idx=False,
            seq_end_idx=False,
            if_encode_all=False,
            alphas=SKLEARN_ALPHAS,
            sklearn_state=RAND_SEED,
            sklearn_params={"normalize":True},
            all_result_folder="results/sklearn-scalex-scaley-noloader-fixembpool",
            #**encoder_params,
        )