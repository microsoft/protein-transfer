from scr.params.sys import DEVICE, RAND_SEED
from scr.model.run_pytorch_model import run_pytorch

run_pytorch(
    dataset_path="data/proeng/thermo/mixed_split.csv",
    # encoder_name="esm1b_t33_650M_UR50S",
    encoder_name="esm1_t6_43M_UR50S",
    reset_param = False,
    resample_param = False,
    embed_batch_size = 128,
    flatten_emb= "mean",
    embed_folder= "embeddings/proeng/thermo/mixed_split",
    seq_start_idx= False,
    seq_end_idx = False,
    loader_batch_size = 64,
    worker_seed = RAND_SEED,
    if_encode_all = False,
    learning_rate = 1e-4,
    lr_decay = 0.1,
    epochs = 10,
    early_stop = True,
    tolerance = 10,
    min_epoch = 5,
    device = DEVICE,
    all_plot_folder = "test/learning_curves",
    all_result_folder = "test/train_val_test",
    # **encoder_params,
    )