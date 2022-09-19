from scr.params.sys import DEVICE, RAND_SEED
from scr.params.emb import TRANSFORMER_INFO
from scr.model.run_pytorch import Run_Pytorch

for encoder_name in TRANSFORMER_INFO.keys():
    Run_Pytorch(
        dataset_path="data/annotation/scl/balanced.csv",
        encoder_name=encoder_name,
        reset_param = False,
        resample_param = True,
        embed_batch_size = 128,
        flatten_emb= "mean",
        # path needs to be confirmed
        embed_folder = f"/home/t-fli/amlt/esm-stat/scl-{encoder_name}-mean-stat/embeddings-stat/annotation/scl/balanced",
        seq_start_idx= False,
        seq_end_idx = False,
        loader_batch_size = 256,
        worker_seed = RAND_SEED,
        if_encode_all = False,
        if_multiprocess = True,
        learning_rate = 1e-4,
        lr_decay = 0.1,
        epochs = 100,
        early_stop = True,
        tolerance = 10,
        min_epoch = 5,
        device = DEVICE,
        all_plot_folder = "results/learning_curves_esm",
        all_result_folder = "results/pytorch_esm",
        # **encoder_params,
)