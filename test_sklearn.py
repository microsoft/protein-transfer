from tkinter import TRUE
from scr.params.sys import SKLEARN_ALPHAS, RAND_SEED
from scr.params.emb import TRANSFORMER_INFO, CARP_INFO
from scr.model.run_sklearn import RunSK

"""
for ds_split in ["low_vs_high", "sampled", "two_vs_rest"]:
    # for encoder_name in ["onehot"]:
    for encoder_name in list(CARP_INFO.keys()):
    # for encoder_name in ["onehot"] + list(TRANSFORMER_INFO.keys()):
        if encoder_name == "onehot":
            embed_batch_size = 0
            flatten_emb = "flatten"
        else:
            embed_batch_size = 64
            flatten_emb = "mean"

        RunSK(
            dataset_path=f"data/proeng/gb1/{ds_split}.csv",
            encoder_name=encoder_name,
            reset_param=True,
            resample_param=False,
            embed_batch_size=embed_batch_size,
            flatten_emb=flatten_emb,
            embed_folder=f"/home/t-fli/amlt/carp_emb/gb1-{ds_split}/embeddings/proeng/gb1/{ds_split}",
            all_embed_layers=False,
            seq_start_idx=0,
            seq_end_idx=56,
            if_encode_all=False,
            alphas=SKLEARN_ALPHAS,
            sklearn_state=RAND_SEED,
            sklearn_params={"normalize":True},
            # all_result_folder="test/sklearn-scaley-noloader-fixembpool",
            all_result_folder="test/sklearn-scalex-scaley-noloader-fixembpool",
            #**encoder_params,
)"""

"""for ds_split in ["low_vs_high", "sampled", "two_vs_rest"]:
    # for encoder_name in ["onehot"]:
    for encoder_name in list(CARP_INFO.keys()):
    # for encoder_name in ["onehot"] + list(TRANSFORMER_INFO.keys()):
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
            embed_folder=f"/home/t-fli/amlt/carp_emb/gb1-{ds_split}/embeddings/proeng/gb1/{ds_split}",
            all_embed_layers=False,
            seq_start_idx=0,
            seq_end_idx=56,
            if_encode_all=False,
            alphas=SKLEARN_ALPHAS,
            sklearn_state=RAND_SEED,
            sklearn_params={"normalize":True},
            # all_result_folder="test/sklearn-scaley-noloader-fixembpool",
            all_result_folder="test/sklearn-scalex-scaley-noloader-fixembpool",
            #**encoder_params,
        )"""
"""
for ds_split in ["low_vs_high", "sampled", "two_vs_rest"]:
    # for encoder_name in ["onehot"]:
    for encoder_name in list(CARP_INFO.keys()):
    # for encoder_name in ["onehot"] + list(TRANSFORMER_INFO.keys()):
        if encoder_name == "onehot":
            embed_batch_size = 0
            flatten_emb = "flatten"
        else:
            embed_batch_size = 64
            flatten_emb = "mean"

        RunSK(
            dataset_path=f"data/proeng/gb1/{ds_split}.csv",
            encoder_name=encoder_name,
            reset_param=True,
            resample_param=False,
            embed_batch_size=embed_batch_size,
            flatten_emb=flatten_emb,
            embed_folder=f"/home/t-fli/amlt/carp_emb_cuda_individual/gb1-{ds_split}-{encoder_name}-mean-rand/embeddings-rand/proeng/gb1/{ds_split}",
            all_embed_layers=False,
            seq_start_idx=0,
            seq_end_idx=56,
            if_encode_all=False,
            alphas=SKLEARN_ALPHAS,
            sklearn_state=RAND_SEED,
            sklearn_params={"normalize":True},
            # all_result_folder="test/sklearn-scaley-noloader-fixembpool",
            all_result_folder="test/sklearn-scalex-scaley-noloader-fixembpool",
            #**encoder_params,
        )"""
"""
for ds_split in ["low_vs_high", "sampled", "two_vs_rest"]:
    # for encoder_name in ["onehot"]:
    for encoder_name in list(CARP_INFO.keys()):
    # for encoder_name in ["onehot"] + list(TRANSFORMER_INFO.keys()):
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
            resample_param=True,
            embed_batch_size=embed_batch_size,
            flatten_emb=flatten_emb,
            embed_folder=f"/home/t-fli/amlt/carp_emb_cuda_individual/gb1-{ds_split}-{encoder_name}-mean-stat/embeddings-stat/proeng/gb1/{ds_split}",
            all_embed_layers=False,
            seq_start_idx=0,
            seq_end_idx=56,
            if_encode_all=False,
            alphas=SKLEARN_ALPHAS,
            sklearn_state=RAND_SEED,
            sklearn_params={"normalize":True},
            # all_result_folder="test/sklearn-scaley-noloader-fixembpool",
            all_result_folder="results/sklearn-scalex-scaley-noloader-carp",
            #**encoder_params,
        )

for ds_split in ["mixed_split"]:
    # for encoder_name in ["onehot"]:
    for encoder_name in list(CARP_INFO.keys()):
    # for encoder_name in ["onehot"] + list(TRANSFORMER_INFO.keys()):
        if encoder_name == "onehot":
            embed_batch_size = 0
            flatten_emb = "flatten"
        else:
            embed_batch_size = 64
            flatten_emb = "mean"

        RunSK(
            dataset_path=f"data/proeng/thermo/{ds_split}.csv",
            encoder_name=encoder_name,
            reset_param=False,
            resample_param=True,
            embed_batch_size=embed_batch_size,
            flatten_emb=flatten_emb,
            embed_folder=f"/home/t-fli/amlt/carp_emb_cuda_individual/thermo-{encoder_name}-mean-stat/embeddings-stat/proeng/thermo/mixed_split",
            all_embed_layers=False,
            seq_start_idx=False,
            seq_end_idx=False,
            if_encode_all=False,
            alphas=SKLEARN_ALPHAS,
            sklearn_state=RAND_SEED,
            sklearn_params={"normalize":True},
            all_result_folder="results/sklearn-scalex-scaley-noloader-carp",
             #all_result_folder="test/sklearn-scaley-noloader-fixembpool"
            #**encoder_params,
        )"""

"""
for ds_split in ["one_vs_many", "two_vs_many"]:
    # for encoder_name in ["onehot"]:
    for encoder_name in ["carp_640M"]:
    # "carp_640M"
    # for encoder_name in ["onehot"] + list(TRANSFORMER_INFO.keys()):
        if encoder_name == "onehot":
            embed_batch_size = 0
            flatten_emb = "flatten"
        else:
            embed_batch_size = 64
            flatten_emb = "mean"

        RunSK(
            dataset_path=f"data/proeng/aav/{ds_split}.csv",
            encoder_name=encoder_name,
            reset_param=False,
            resample_param=False,
            embed_batch_size=embed_batch_size,
            flatten_emb=flatten_emb,
            embed_folder=f"/home/t-fli/amlt/carp_emb_cuda_individual/aav-{ds_split}-{encoder_name}-mean/embeddings/proeng/aav/{ds_split}",
            all_embed_layers=False,
            seq_start_idx=False,
            seq_end_idx=False,
            if_encode_all=False,
            alphas=SKLEARN_ALPHAS,
            sklearn_state=RAND_SEED,
            sklearn_params={"normalize":True},
            # all_result_folder="test/sklearn-scaley-noloader-fixembpool",
            # all_result_folder="test/sklearn-scalex-scaley-noloader-fixembpool",
            all_result_folder="results/sklearn-scalex-scaley-noloader-carp"
            #**encoder_params,
        )


for ds_split in ["one_vs_many", "two_vs_many"]:
    # for encoder_name in ["onehot"]:
    for encoder_name in ["carp_640M"]:
    # "carp_640M"
    # for encoder_name in ["onehot"] + list(TRANSFORMER_INFO.keys()):
        if encoder_name == "onehot":
            embed_batch_size = 0
            flatten_emb = "flatten"
        else:
            embed_batch_size = 64
            flatten_emb = "mean"

        RunSK(
            dataset_path=f"data/proeng/aav/{ds_split}.csv",
            encoder_name=encoder_name,
            reset_param=True,
            resample_param=False,
            embed_batch_size=embed_batch_size,
            flatten_emb=flatten_emb,
            embed_folder=f"/home/t-fli/amlt/carp_emb_cuda_individual/aav-{ds_split}-{encoder_name}-mean-rand/embeddings-rand/proeng/aav/{ds_split}",
            all_embed_layers=False,
            seq_start_idx=False,
            seq_end_idx=False,
            if_encode_all=False,
            alphas=SKLEARN_ALPHAS,
            sklearn_state=RAND_SEED,
            sklearn_params={"normalize":True},
            # all_result_folder="test/sklearn-scaley-noloader-fixembpool",
            # all_result_folder="test/sklearn-scalex-scaley-noloader-fixembpool",
            all_result_folder="results/sklearn-scalex-scaley-noloader-carp"
            #**encoder_params,
        )"""
"""
for ds_split in ["two_vs_many"]:
    # for encoder_name in ["onehot"]:
    for encoder_name in ["carp_640M"]:
    # "carp_640M"
    # for encoder_name in ["onehot"] + list(TRANSFORMER_INFO.keys()):
        if encoder_name == "onehot":
            embed_batch_size = 0
            flatten_emb = "flatten"
        else:
            embed_batch_size = 64
            flatten_emb = "mean"

        RunSK(
            dataset_path=f"data/proeng/aav/{ds_split}.csv",
            encoder_name=encoder_name,
            reset_param=False,
            resample_param=True,
            embed_batch_size=embed_batch_size,
            flatten_emb=flatten_emb,
            embed_folder=f"/home/t-fli/amlt/carp_emb_cuda_individual/aav-{ds_split}-{encoder_name}-mean-stat/embeddings-stat/proeng/aav/{ds_split}",
            all_embed_layers=False,
            seq_start_idx=False,
            seq_end_idx=False,
            if_encode_all=False,
            alphas=SKLEARN_ALPHAS,
            sklearn_state=RAND_SEED,
            sklearn_params={"normalize":True},
            # all_result_folder="test/sklearn-scaley-noloader-fixembpool",
            # all_result_folder="test/sklearn-scalex-scaley-noloader-fixembpool",
            all_result_folder="results/sklearn-scalex-scaley-noloader-carp"
            #**encoder_params,
        )"""


for ds_split in ["one_vs_many"]:
    # for encoder_name in ["onehot"]:
    for encoder_name in ["carp_640M"]:
    # "carp_640M"
    # for encoder_name in ["onehot"] + list(TRANSFORMER_INFO.keys()):
        if encoder_name == "onehot":
            embed_batch_size = 0
            flatten_emb = "flatten"
        else:
            embed_batch_size = 64
            flatten_emb = "mean"

        RunSK(
            dataset_path=f"data/proeng/aav/{ds_split}.csv",
            encoder_name=encoder_name,
            reset_param=False,
            resample_param=True,
            embed_batch_size=embed_batch_size,
            flatten_emb=flatten_emb,
            embed_folder=f"/home/t-fli/amlt/carp_emb_cuda_individual/aav-{ds_split}-{encoder_name}-mean-stat/embeddings-stat/proeng/aav/{ds_split}",
            all_embed_layers=False,
            seq_start_idx=False,
            seq_end_idx=False,
            if_encode_all=False,
            alphas=SKLEARN_ALPHAS,
            sklearn_state=RAND_SEED,
            sklearn_params={"normalize":True},
            # all_result_folder="test/sklearn-scaley-noloader-fixembpool",
            # all_result_folder="test/sklearn-scalex-scaley-noloader-fixembpool",
            all_result_folder="results/sklearn-scalex-scaley-noloader-carp"
            #**encoder_params,
        )

"""
for ds_split in ["one_vs_many", "two_vs_many"]:
    # for encoder_name in ["onehot"]:
    for encoder_name in ["carp_600k", "carp_38M", "carp_76M"]:
    # "carp_640M"
    # for encoder_name in ["onehot"] + list(TRANSFORMER_INFO.keys()):
        if encoder_name == "onehot":
            embed_batch_size = 0
            flatten_emb = "flatten"
        else:
            embed_batch_size = 64
            flatten_emb = "mean"

        RunSK(
            dataset_path=f"data/proeng/aav/{ds_split}.csv",
            encoder_name=encoder_name,
            reset_param=False,
            resample_param=False,
            embed_batch_size=embed_batch_size,
            flatten_emb=flatten_emb,
            embed_folder=f"/home/t-fli/amlt/carp_emb_cuda_individual/aav-{ds_split}-{encoder_name}-mean/embeddings/proeng/aav/{ds_split}",
            all_embed_layers=False,
            seq_start_idx=False,
            seq_end_idx=False,
            if_encode_all=False,
            alphas=SKLEARN_ALPHAS,
            sklearn_state=RAND_SEED,
            sklearn_params={"normalize":True},
            # all_result_folder="test/sklearn-scaley-noloader-fixembpool",
            # all_result_folder="test/sklearn-scalex-scaley-noloader-fixembpool",
            all_result_folder="results/sklearn-scalex-scaley-noloader-carp"
            #**encoder_params,
        )


for ds_split in ["one_vs_many", "two_vs_many"]:
    # for encoder_name in ["onehot"]:
    for encoder_name in ["carp_600k", "carp_38M", "carp_76M"]:
    # "carp_640M"
    # for encoder_name in ["onehot"] + list(TRANSFORMER_INFO.keys()):
        if encoder_name == "onehot":
            embed_batch_size = 0
            flatten_emb = "flatten"
        else:
            embed_batch_size = 64
            flatten_emb = "mean"

        RunSK(
            dataset_path=f"data/proeng/aav/{ds_split}.csv",
            encoder_name=encoder_name,
            reset_param=True,
            resample_param=False,
            embed_batch_size=embed_batch_size,
            flatten_emb=flatten_emb,
            embed_folder=f"/home/t-fli/amlt/carp_emb_cuda_individual/aav-{ds_split}-{encoder_name}-mean-rand/embeddings-rand/proeng/aav/{ds_split}",
            all_embed_layers=False,
            seq_start_idx=False,
            seq_end_idx=False,
            if_encode_all=False,
            alphas=SKLEARN_ALPHAS,
            sklearn_state=RAND_SEED,
            sklearn_params={"normalize":True},
            # all_result_folder="test/sklearn-scaley-noloader-fixembpool",
            # all_result_folder="test/sklearn-scalex-scaley-noloader-fixembpool",
            all_result_folder="results/sklearn-scalex-scaley-noloader-carp"
            #**encoder_params,
        )


for ds_split in ["one_vs_many", "two_vs_many"]:
    # for encoder_name in ["onehot"]:
    for encoder_name in ["carp_600k", "carp_38M", "carp_76M"]:
    # "carp_640M"
    # for encoder_name in ["onehot"] + list(TRANSFORMER_INFO.keys()):
        if encoder_name == "onehot":
            embed_batch_size = 0
            flatten_emb = "flatten"
        else:
            embed_batch_size = 64
            flatten_emb = "mean"

        RunSK(
            dataset_path=f"data/proeng/aav/{ds_split}.csv",
            encoder_name=encoder_name,
            reset_param=False,
            resample_param=True,
            embed_batch_size=embed_batch_size,
            flatten_emb=flatten_emb,
            embed_folder=f"/home/t-fli/amlt/carp_emb_cuda_individual/aav-{ds_split}-{encoder_name}-mean-stat/embeddings-stat/proeng/aav/{ds_split}",
            all_embed_layers=False,
            seq_start_idx=False,
            seq_end_idx=False,
            if_encode_all=False,
            alphas=SKLEARN_ALPHAS,
            sklearn_state=RAND_SEED,
            sklearn_params={"normalize":True},
            # all_result_folder="test/sklearn-scaley-noloader-fixembpool",
            # all_result_folder="test/sklearn-scalex-scaley-noloader-fixembpool",
            all_result_folder="results/sklearn-scalex-scaley-noloader-carp"
            #**encoder_params,
        )"""
# scl will not converge
"""
for ds_split in ["balanced"]:
    # for encoder_name in ["onehot"]:
    for encoder_name in list(CARP_INFO.keys()):
    # for encoder_name in ["onehot"] + list(TRANSFORMER_INFO.keys()):
        if encoder_name == "onehot":
            embed_batch_size = 0
            flatten_emb = "flatten"
        else:
            embed_batch_size = 64
            flatten_emb = "mean"

        RunSK(
            dataset_path=f"data/annotation/scl/{ds_split}.csv",
            encoder_name=encoder_name,
            reset_param=False,
            resample_param=False,
            embed_batch_size=embed_batch_size,
            flatten_emb=flatten_emb,
            embed_folder=f"/home/t-fli/amlt/carp_emb_cuda_individual/scl-{encoder_name}-mean/embeddings/annotation/scl/balanced",
            all_embed_layers=False,
            seq_start_idx=False,
            seq_end_idx=False,
            if_encode_all=False,
            alphas=SKLEARN_ALPHAS,
            sklearn_state=RAND_SEED,
            # sklearn_params={"normalize":True},
            # all_result_folder="test/sklearn-scaley-noloader-fixembpool",
            all_result_folder="test/sklearn-scaley-noloader-fixembpool",
            #**encoder_params,
        )"""


# NEED TO RUN THE 640 ONE
"""
for ds_split in ["one_vs_many", "two_vs_many"]:
    # for encoder_name in ["onehot"]:
    for encoder_name in list(CARP_INFO.keys()):
    # for encoder_name in ["onehot"] + list(TRANSFORMER_INFO.keys()):
        if encoder_name == "onehot":
            embed_batch_size = 0
            flatten_emb = "flatten"
        else:
            embed_batch_size = 64
            flatten_emb = "mean"

        RunSK(
            dataset_path=f"data/proeng/aav/{ds_split}.csv",
            encoder_name=encoder_name,
            reset_param=False,
            resample_param=False,
            embed_batch_size=embed_batch_size,
            flatten_emb=flatten_emb,
            embed_folder=f"/home/t-fli/amlt/carp_emb_cuda_individual/aav-{ds_split}-{encoder_name}-mean/embeddings/proeng/aav/{ds_split}",
            all_embed_layers=False,
            seq_start_idx=False,
            seq_end_idx=False,
            if_encode_all=False,
            alphas=SKLEARN_ALPHAS,
            sklearn_state=RAND_SEED,
            sklearn_params={"normalize":True},
            # all_result_folder="test/sklearn-scaley-noloader-fixembpool",
            all_result_folder="test/sklearn-scalex-scaley-noloader-fixembpool",
            #**encoder_params,
        )"""

"""
for ds_split in ["mixed_split"]:
    # for encoder_name in ["onehot"]:
    for encoder_name in list(CARP_INFO.keys()):
    # for encoder_name in ["onehot"] + list(TRANSFORMER_INFO.keys()):
        if encoder_name == "onehot":
            embed_batch_size = 0
            flatten_emb = "flatten"
        else:
            embed_batch_size = 64
            flatten_emb = "mean"

        RunSK(
            dataset_path=f"data/proeng/thermo/{ds_split}.csv",
            encoder_name=encoder_name,
            reset_param=False,
            resample_param=False,
            embed_batch_size=embed_batch_size,
            flatten_emb=flatten_emb,
            embed_folder=f"/home/t-fli/amlt/carp_emb_cuda_individual/thermo-{encoder_name}-mean/embeddings/proeng/thermo/mixed_split",
            all_embed_layers=False,
            seq_start_idx=False,
            seq_end_idx=False,
            if_encode_all=False,
            alphas=SKLEARN_ALPHAS,
            sklearn_state=RAND_SEED,
            # sklearn_params={"normalize":True},
            # all_result_folder="test/sklearn-scaley-noloader-fixembpool",
            all_result_folder="test/sklearn-scaley-noloader-fixembpool",
            #**encoder_params,
        )"""

"""for ds_split in ["mixed_split"]:
    # for encoder_name in ["onehot"]:
    for encoder_name in list(CARP_INFO.keys()):
    # for encoder_name in ["onehot"] + list(TRANSFORMER_INFO.keys()):
        if encoder_name == "onehot":
            embed_batch_size = 0
            flatten_emb = "flatten"
        else:
            embed_batch_size = 64
            flatten_emb = "mean"

        RunSK(
            dataset_path=f"data/proeng/thermo/{ds_split}.csv",
            encoder_name=encoder_name,
            reset_param=True,
            resample_param=False,
            embed_batch_size=embed_batch_size,
            flatten_emb=flatten_emb,
            embed_folder=f"/home/t-fli/amlt/carp_emb_cuda_individual/thermo-{encoder_name}-mean-rand/embeddings-rand/proeng/thermo/mixed_split",
            all_embed_layers=False,
            seq_start_idx=False,
            seq_end_idx=False,
            if_encode_all=False,
            alphas=SKLEARN_ALPHAS,
            sklearn_state=RAND_SEED,
            sklearn_params={"normalize":True},
            all_result_folder="results/sklearn-scalex-scaley-noloader-carp",
            #**encoder_params,
        )"""

"""
for ds_split in ["low_vs_high", "sampled", "two_vs_rest"]:
    # for encoder_name in ["onehot"]:
    for encoder_name in list(TRANSFORMER_INFO.keys()):
    # for encoder_name in ["onehot"] + list(TRANSFORMER_INFO.keys()):
        if encoder_name == "onehot":
            embed_batch_size = 0
            flatten_emb = "flatten"
        else:
            embed_batch_size = 64
            flatten_emb = "mean"

        RunSK(
            dataset_path=f"data/proeng/gb1/{ds_split}.csv",
            encoder_name=encoder_name,
            reset_param=True,
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
            all_result_folder="test/sklearn-scaley-noloader-fixembpool",
            #**encoder_params,
        )"""

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
"""
for encoder_name in TRANSFORMER_INFO.keys():
    RunSK(
        dataset_path="data/proeng/thermo/mixed_split.csv",
        encoder_name=encoder_name,
        reset_param= False,
        resample_param = False,
        embed_batch_size = 128,
        flatten_emb = "mean",
        embed_folder = "embeddings/proeng/thermo/mixed_split",
        all_embed_layers=False,
        seq_start_idx = False,
        seq_end_idx = False,
        if_encode_all=False,
        alphas = SKLEARN_ALPHAS,
        sklearn_state= RAND_SEED,
        sklearn_params= None,
        all_result_folder = "test/sklearn-scaley-noloader-fixembpool",
        # **encoder_params,
    )

for encoder_name in TRANSFORMER_INFO.keys():
    RunSK(
        dataset_path="data/proeng/thermo/mixed_split.csv",
        encoder_name=encoder_name,
        reset_param= False,
        resample_param = False,
        embed_batch_size = 128,
        flatten_emb = "mean",
        embed_folder = "embeddings/proeng/thermo/mixed_split",
        all_embed_layers=False,
        seq_start_idx = False,
        seq_end_idx = False,
        if_encode_all=False,
        alphas = SKLEARN_ALPHAS,
        sklearn_state= {"normalize":True},
        sklearn_params= None,
        all_result_folder = "test/sklearn-scalex-scaley-noloader-fixembpool",
        # **encoder_params,
    )
"""
"""
for encoder_name in TRANSFORMER_INFO.keys():
    RunSK(
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
        sklearn_state= RAND_SEED,
        sklearn_params= None,
        all_result_folder = "test/sklearn-scaley-noloader-fixembpool",
        # **encoder_params,
    )

for encoder_name in TRANSFORMER_INFO.keys():
    RunSK(
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
        sklearn_state= {"normalize":True},
        sklearn_params= None,
        all_result_folder = "test/sklearn-scalex-scaley-noloader-fixembpool",
        # **encoder_params,
    )"""

"""
RunSK(
    dataset_path="data/proeng/thermo/mixed_split.csv",
    encoder_name="onehot",
    reset_param= False,
    resample_param = False,
    embed_batch_size = 128,
    flatten_emb = "flatten",
    embed_folder = "embeddings/proeng/thermo/mixed_split",
    all_embed_layers=False,
    seq_start_idx = False,
    seq_end_idx = False,
    if_encode_all=False,
    alphas = SKLEARN_ALPHAS,
    sklearn_state= RAND_SEED,
    sklearn_params= None,
    all_result_folder = "test/sklearn-scaley-noloader-fixembpool",
    # **encoder_params,
)

RunSK(
    dataset_path="data/proeng/thermo/mixed_split.csv",
    encoder_name="onehot",
    reset_param= False,
    resample_param = False,
    embed_batch_size = 128,
    flatten_emb = "flatten",
    embed_folder = "embeddings/proeng/thermo/mixed_split",
    all_embed_layers=False,
    seq_start_idx = False,
    seq_end_idx = False,
    if_encode_all=False,
    alphas = SKLEARN_ALPHAS,
    sklearn_state= RAND_SEED,
    sklearn_params= {"normalize":True},
    all_result_folder = "test/sklearn-scalex-scaley-noloader-fixembpool",
    # **encoder_params,
)
"""
"""
RunSK(
    dataset_path="data/proeng/thermo/mixed_split.csv",
    encoder_name="onehot",
    reset_param= True,
    resample_param = False,
    embed_batch_size = 128,
    flatten_emb = "flatten",
    embed_folder = "embeddings-rand/proeng/thermo/mixed_split",
    all_embed_layers=False,
    seq_start_idx = False,
    seq_end_idx = False,
    if_encode_all=False,
    alphas = SKLEARN_ALPHAS,
    sklearn_state= RAND_SEED,
    sklearn_params= None,
    all_result_folder = "test/sklearn-scaley-noloader-fixembpool",
    # **encoder_params,
)"""
"""
RunSK(
    dataset_path="data/proeng/thermo/mixed_split.csv",
    encoder_name="onehot",
    reset_param= True,
    resample_param = False,
    embed_batch_size = 128,
    flatten_emb = "flatten",
    embed_folder = "embeddings-rand/proeng/thermo/mixed_split",
    all_embed_layers=False,
    seq_start_idx = False,
    seq_end_idx = False,
    if_encode_all=False,
    alphas = SKLEARN_ALPHAS,
    sklearn_state= RAND_SEED,
    sklearn_params= {"normalize":True},
    all_result_folder = "test/sklearn-scalex-scaley-noloader-fixembpool",
    # **encoder_params,
)"""
"""
for encoder_name in TRANSFORMER_INFO.keys():
    RunSK(
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
        sklearn_state= RAND_SEED,
        sklearn_params= {"normalize":True},
        all_result_folder = "test/sklearn-scalex-scaley-noloader-fixembpool",
        # **encoder_params,
    )


for encoder_name in TRANSFORMER_INFO.keys():
    RunSK(
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
        sklearn_state= RAND_SEED,
        # sklearn_params= {"normalize":True},
        all_result_folder = "test/sklearn-scaley-noloader-fixembpool",
        # **encoder_params,
    )"""

"""
for encoder_name in TRANSFORMER_INFO.keys():
    RunSK(
        dataset_path="data/annotation/scl/balanced.csv",
        encoder_name=encoder_name,
        reset_param= False,
        resample_param = False,
        embed_batch_size = 128,
        flatten_emb = "mean",
        embed_folder = "embeddings/annotation/scl/balanced",
        all_embed_layers=False,
        seq_start_idx = False,
        seq_end_idx = False,
        if_encode_all=False,
        alphas = SKLEARN_ALPHAS,
        sklearn_state= RAND_SEED,
        sklearn_params= None,
        all_result_folder = "test/sklearn-scaley-noloader-fixembpool",
        # **encoder_params,
    )

for encoder_name in TRANSFORMER_INFO.keys():
    RunSK(
        dataset_path="data/annotation/scl/balanced.csv",
        encoder_name=encoder_name,
        reset_param= False,
        resample_param = False,
        embed_batch_size = 128,
        flatten_emb = "mean",
        embed_folder = "embeddings/annotation/scl/balanced",
        all_embed_layers=False,
        seq_start_idx = False,
        seq_end_idx = False,
        if_encode_all=False,
        alphas = SKLEARN_ALPHAS,
        sklearn_state= RAND_SEED,
        sklearn_params= {"normalize":True},
        all_result_folder = "test/sklearn-scalex-scaley-noloader-fixembpool",
        # **encoder_params,
    )"""