# queue for running all sklearn models from embs
export CUDA_VISIBLE_DEVICES=1
### GB1 ###

# python run_protran_sklearn.py --dataset_path="data/proeng/gb1/sampled.csv" --encoder_name="esm1_t6_43M_UR50S" --checkpoint=1 --embed_batch_size=64 --flatten_emb="mean" --embed_folder="embeddings" --seq_end_idx=56 --all_embed_layers=True --sklearn_params='{"normalize":true}' --all_result_folder="results/sklearn-esm" --reset_param=True --embed_torch_seed=0
# python run_protran_sklearn.py --dataset_path="data/proeng/gb1/sampled.csv" --encoder_name="esm1_t6_43M_UR50S" --checkpoint=1 --embed_batch_size=64 --flatten_emb="mean" --embed_folder="embeddings" --seq_end_idx=56 --all_embed_layers=True --sklearn_params='{"normalize":true}' --all_result_folder="results/sklearn-esm" --resample_param=True --embed_torch_seed=0
# python run_protran_sklearn.py --dataset_path="data/proeng/gb1/sampled.csv" --encoder_name="esm1_t6_43M_UR50S" --checkpoint=1 --embed_batch_size=64 --flatten_emb="mean" --embed_folder="embeddings" --seq_end_idx=56 --all_embed_layers=True --sklearn_params='{"normalize":true}' --all_result_folder="results/sklearn-esm" --reset_param=True --embed_torch_seed=12345
# python run_protran_sklearn.py --dataset_path="data/proeng/gb1/sampled.csv" --encoder_name="esm1_t6_43M_UR50S" --checkpoint=1 --embed_batch_size=64 --flatten_emb="mean" --embed_folder="embeddings" --seq_end_idx=56 --all_embed_layers=True --sklearn_params='{"normalize":true}' --all_result_folder="results/sklearn-esm" --resample_param=True --embed_torch_seed=12345
# python run_protran_sklearn.py --dataset_path="data/proeng/gb1/sampled.csv" --encoder_name="esm1_t6_43M_UR50S" --checkpoint=1 --embed_batch_size=64 --flatten_emb="mean" --embed_folder="embeddings" --seq_end_idx=56 --all_embed_layers=True --sklearn_params='{"normalize":true}' --all_result_folder="results/sklearn-esm" --reset_param=True --embed_torch_seed=42
# python run_protran_sklearn.py --dataset_path="data/proeng/gb1/sampled.csv" --encoder_name="esm1_t6_43M_UR50S" --checkpoint=1 --embed_batch_size=64 --flatten_emb="mean" --embed_folder="embeddings" --seq_end_idx=56 --all_embed_layers=True --sklearn_params='{"normalize":true}' --all_result_folder="results/sklearn-esm" --resample_param=True --embed_torch_seed=42
# python run_protran_sklearn.py --dataset_path="data/proeng/gb1/sampled.csv" --encoder_name="esm1_t6_43M_UR50S" --checkpoint=1 --embed_batch_size=64 --flatten_emb="mean" --embed_folder="embeddings" --seq_end_idx=56 --all_embed_layers=True --sklearn_params='{"normalize":true}' --all_result_folder="results/sklearn-esm"

# python run_protran_sklearn.py --dataset_path="data/proeng/gb1/sampled.csv" --encoder_name="esm1_t12_85M_UR50S" --checkpoint=1 --embed_batch_size=64 --flatten_emb="mean" --embed_folder="embeddings" --seq_end_idx=56 --all_embed_layers=True --sklearn_params='{"normalize":true}' --all_result_folder="results/sklearn-esm" --reset_param=True --embed_torch_seed=0
# python run_protran_sklearn.py --dataset_path="data/proeng/gb1/sampled.csv" --encoder_name="esm1_t12_85M_UR50S" --checkpoint=1 --embed_batch_size=64 --flatten_emb="mean" --embed_folder="embeddings" --seq_end_idx=56 --all_embed_layers=True --sklearn_params='{"normalize":true}' --all_result_folder="results/sklearn-esm" --resample_param=True --embed_torch_seed=0
# python run_protran_sklearn.py --dataset_path="data/proeng/gb1/sampled.csv" --encoder_name="esm1_t12_85M_UR50S" --checkpoint=1 --embed_batch_size=64 --flatten_emb="mean" --embed_folder="embeddings" --seq_end_idx=56 --all_embed_layers=True --sklearn_params='{"normalize":true}' --all_result_folder="results/sklearn-esm" --reset_param=True --embed_torch_seed=12345
# python run_protran_sklearn.py --dataset_path="data/proeng/gb1/sampled.csv" --encoder_name="esm1_t12_85M_UR50S" --checkpoint=1 --embed_batch_size=64 --flatten_emb="mean" --embed_folder="embeddings" --seq_end_idx=56 --all_embed_layers=True --sklearn_params='{"normalize":true}' --all_result_folder="results/sklearn-esm" --resample_param=True --embed_torch_seed=12345
# python run_protran_sklearn.py --dataset_path="data/proeng/gb1/sampled.csv" --encoder_name="esm1_t12_85M_UR50S" --checkpoint=1 --embed_batch_size=64 --flatten_emb="mean" --embed_folder="embeddings" --seq_end_idx=56 --all_embed_layers=True --sklearn_params='{"normalize":true}' --all_result_folder="results/sklearn-esm" --reset_param=True --embed_torch_seed=42
# python run_protran_sklearn.py --dataset_path="data/proeng/gb1/sampled.csv" --encoder_name="esm1_t12_85M_UR50S" --checkpoint=1 --embed_batch_size=64 --flatten_emb="mean" --embed_folder="embeddings" --seq_end_idx=56 --all_embed_layers=True --sklearn_params='{"normalize":true}' --all_result_folder="results/sklearn-esm" --resample_param=True --embed_torch_seed=42
# python run_protran_sklearn.py --dataset_path="data/proeng/gb1/sampled.csv" --encoder_name="esm1_t12_85M_UR50S" --checkpoint=1 --embed_batch_size=64 --flatten_emb="mean" --embed_folder="embeddings" --seq_end_idx=56 --all_embed_layers=True --sklearn_params='{"normalize":true}' --all_result_folder="results/sklearn-esm"

# python run_protran_sklearn.py --dataset_path="data/proeng/gb1/sampled.csv" --encoder_name="esm1_t34_670M_UR50S" --checkpoint=1 --embed_batch_size=64 --flatten_emb="mean" --embed_folder="embeddings" --seq_end_idx=56 --all_embed_layers=True --sklearn_params='{"normalize":true}' --all_result_folder="results/sklearn-esm" --reset_param=True --embed_torch_seed=0
# python run_protran_sklearn.py --dataset_path="data/proeng/gb1/sampled.csv" --encoder_name="esm1_t34_670M_UR50S" --checkpoint=1 --embed_batch_size=64 --flatten_emb="mean" --embed_folder="embeddings" --seq_end_idx=56 --all_embed_layers=True --sklearn_params='{"normalize":true}' --all_result_folder="results/sklearn-esm" --resample_param=True --embed_torch_seed=0
# python run_protran_sklearn.py --dataset_path="data/proeng/gb1/sampled.csv" --encoder_name="esm1_t34_670M_UR50S" --checkpoint=1 --embed_batch_size=64 --flatten_emb="mean" --embed_folder="embeddings" --seq_end_idx=56 --all_embed_layers=True --sklearn_params='{"normalize":true}' --all_result_folder="results/sklearn-esm" --reset_param=True --embed_torch_seed=12345
# python run_protran_sklearn.py --dataset_path="data/proeng/gb1/sampled.csv" --encoder_name="esm1_t34_670M_UR50S" --checkpoint=1 --embed_batch_size=64 --flatten_emb="mean" --embed_folder="embeddings" --seq_end_idx=56 --all_embed_layers=True --sklearn_params='{"normalize":true}' --all_result_folder="results/sklearn-esm" --resample_param=True --embed_torch_seed=12345
# python run_protran_sklearn.py --dataset_path="data/proeng/gb1/sampled.csv" --encoder_name="esm1_t34_670M_UR50S" --checkpoint=1 --embed_batch_size=64 --flatten_emb="mean" --embed_folder="embeddings" --seq_end_idx=56 --all_embed_layers=True --sklearn_params='{"normalize":true}' --all_result_folder="results/sklearn-esm" --reset_param=True --embed_torch_seed=42
# python run_protran_sklearn.py --dataset_path="data/proeng/gb1/sampled.csv" --encoder_name="esm1_t34_670M_UR50S" --checkpoint=1 --embed_batch_size=64 --flatten_emb="mean" --embed_folder="embeddings" --seq_end_idx=56 --all_embed_layers=True --sklearn_params='{"normalize":true}' --all_result_folder="results/sklearn-esm" --resample_param=True --embed_torch_seed=42
# python run_protran_sklearn.py --dataset_path="data/proeng/gb1/sampled.csv" --encoder_name="esm1_t34_670M_UR50S" --checkpoint=1 --embed_batch_size=64 --flatten_emb="mean" --embed_folder="embeddings" --seq_end_idx=56 --all_embed_layers=True --sklearn_params='{"normalize":true}' --all_result_folder="results/sklearn-esm"

# python run_protran_sklearn.py --dataset_path="data/proeng/gb1/sampled.csv" --encoder_name="esm1b_t33_650M_UR50S" --checkpoint=1 --embed_batch_size=64 --flatten_emb="mean" --embed_folder="embeddings" --seq_end_idx=56 --all_embed_layers=True --sklearn_params='{"normalize":true}' --all_result_folder="results/sklearn-esm" --reset_param=True --embed_torch_seed=0
# python run_protran_sklearn.py --dataset_path="data/proeng/gb1/sampled.csv" --encoder_name="esm1b_t33_650M_UR50S" --checkpoint=1 --embed_batch_size=64 --flatten_emb="mean" --embed_folder="embeddings" --seq_end_idx=56 --all_embed_layers=True --sklearn_params='{"normalize":true}' --all_result_folder="results/sklearn-esm" --resample_param=True --embed_torch_seed=0
# python run_protran_sklearn.py --dataset_path="data/proeng/gb1/sampled.csv" --encoder_name="esm1b_t33_650M_UR50S" --checkpoint=1 --embed_batch_size=64 --flatten_emb="mean" --embed_folder="embeddings" --seq_end_idx=56 --all_embed_layers=True --sklearn_params='{"normalize":true}' --all_result_folder="results/sklearn-esm" --reset_param=True --embed_torch_seed=12345
# python run_protran_sklearn.py --dataset_path="data/proeng/gb1/sampled.csv" --encoder_name="esm1b_t33_650M_UR50S" --checkpoint=1 --embed_batch_size=64 --flatten_emb="mean" --embed_folder="embeddings" --seq_end_idx=56 --all_embed_layers=True --sklearn_params='{"normalize":true}' --all_result_folder="results/sklearn-esm" --resample_param=True --embed_torch_seed=12345
# python run_protran_sklearn.py --dataset_path="data/proeng/gb1/sampled.csv" --encoder_name="esm1b_t33_650M_UR50S" --checkpoint=1 --embed_batch_size=64 --flatten_emb="mean" --embed_folder="embeddings" --seq_end_idx=56 --all_embed_layers=True --sklearn_params='{"normalize":true}' --all_result_folder="results/sklearn-esm" --reset_param=True --embed_torch_seed=42
# python run_protran_sklearn.py --dataset_path="data/proeng/gb1/sampled.csv" --encoder_name="esm1b_t33_650M_UR50S" --checkpoint=1 --embed_batch_size=64 --flatten_emb="mean" --embed_folder="embeddings" --seq_end_idx=56 --all_embed_layers=True --sklearn_params='{"normalize":true}' --all_result_folder="results/sklearn-esm" --resample_param=True --embed_torch_seed=42
# python run_protran_sklearn.py --dataset_path="data/proeng/gb1/sampled.csv" --encoder_name="esm1b_t33_650M_UR50S" --checkpoint=1 --embed_batch_size=64 --flatten_emb="mean" --embed_folder="embeddings" --seq_end_idx=56 --all_embed_layers=True --sklearn_params='{"normalize":true}' --all_result_folder="results/sklearn-esm"



# python run_protran_sklearn.py --dataset_path="data/proeng/gb1/low_vs_high.csv" --encoder_name="esm1_t6_43M_UR50S" --checkpoint=1 --embed_batch_size=64 --flatten_emb="mean" --embed_folder="embeddings" --seq_end_idx=56 --all_embed_layers=True --sklearn_params='{"normalize":true}' --all_result_folder="results/sklearn-esm" --reset_param=True
# python run_protran_sklearn.py --dataset_path="data/proeng/gb1/low_vs_high.csv" --encoder_name="esm1_t6_43M_UR50S" --checkpoint=1 --embed_batch_size=64 --flatten_emb="mean" --embed_folder="embeddings" --seq_end_idx=56 --all_embed_layers=True --sklearn_params='{"normalize":true}' --all_result_folder="results/sklearn-esm" --resample_param=True
# python run_protran_sklearn.py --dataset_path="data/proeng/gb1/low_vs_high.csv" --encoder_name="esm1_t6_43M_UR50S" --checkpoint=1 --embed_batch_size=64 --flatten_emb="mean" --embed_folder="embeddings" --seq_end_idx=56 --all_embed_layers=True --sklearn_params='{"normalize":true}' --all_result_folder="results/sklearn-esm"

# python run_protran_sklearn.py --dataset_path="data/proeng/gb1/low_vs_high.csv" --encoder_name="esm1_t12_85M_UR50S" --checkpoint=1 --embed_batch_size=64 --flatten_emb="mean" --embed_folder="embeddings" --seq_end_idx=56 --all_embed_layers=True --sklearn_params='{"normalize":true}' --all_result_folder="results/sklearn-esm" --reset_param=True
# python run_protran_sklearn.py --dataset_path="data/proeng/gb1/low_vs_high.csv" --encoder_name="esm1_t12_85M_UR50S" --checkpoint=1 --embed_batch_size=64 --flatten_emb="mean" --embed_folder="embeddings" --seq_end_idx=56 --all_embed_layers=True --sklearn_params='{"normalize":true}' --all_result_folder="results/sklearn-esm" --resample_param=True
# python run_protran_sklearn.py --dataset_path="data/proeng/gb1/low_vs_high.csv" --encoder_name="esm1_t12_85M_UR50S" --checkpoint=1 --embed_batch_size=64 --flatten_emb="mean" --embed_folder="embeddings" --seq_end_idx=56 --all_embed_layers=True --sklearn_params='{"normalize":true}' --all_result_folder="results/sklearn-esm"

# python run_protran_sklearn.py --dataset_path="data/proeng/gb1/low_vs_high.csv" --encoder_name="esm1_t34_670M_UR50S" --checkpoint=1 --embed_batch_size=64 --flatten_emb="mean" --embed_folder="embeddings" --seq_end_idx=56 --all_embed_layers=True --sklearn_params='{"normalize":true}' --all_result_folder="results/sklearn-esm" --reset_param=True
# python run_protran_sklearn.py --dataset_path="data/proeng/gb1/low_vs_high.csv" --encoder_name="esm1_t34_670M_UR50S" --checkpoint=1 --embed_batch_size=64 --flatten_emb="mean" --embed_folder="embeddings" --seq_end_idx=56 --all_embed_layers=True --sklearn_params='{"normalize":true}' --all_result_folder="results/sklearn-esm" --resample_param=True
# python run_protran_sklearn.py --dataset_path="data/proeng/gb1/low_vs_high.csv" --encoder_name="esm1_t34_670M_UR50S" --checkpoint=1 --embed_batch_size=64 --flatten_emb="mean" --embed_folder="embeddings" --seq_end_idx=56 --all_embed_layers=True --sklearn_params='{"normalize":true}' --all_result_folder="results/sklearn-esm"

# python run_protran_sklearn.py --dataset_path="data/proeng/gb1/low_vs_high.csv" --encoder_name="esm1b_t33_650M_UR50S" --checkpoint=1 --embed_batch_size=64 --flatten_emb="mean" --embed_folder="embeddings" --seq_end_idx=56 --all_embed_layers=True --sklearn_params='{"normalize":true}' --all_result_folder="results/sklearn-esm" --reset_param=True
# python run_protran_sklearn.py --dataset_path="data/proeng/gb1/low_vs_high.csv" --encoder_name="esm1b_t33_650M_UR50S" --checkpoint=1 --embed_batch_size=64 --flatten_emb="mean" --embed_folder="embeddings" --seq_end_idx=56 --all_embed_layers=True --sklearn_params='{"normalize":true}' --all_result_folder="results/sklearn-esm" --resample_param=True
# python run_protran_sklearn.py --dataset_path="data/proeng/gb1/low_vs_high.csv" --encoder_name="esm1b_t33_650M_UR50S" --checkpoint=1 --embed_batch_size=64 --flatten_emb="mean" --embed_folder="embeddings" --seq_end_idx=56 --all_embed_layers=True --sklearn_params='{"normalize":true}' --all_result_folder="results/sklearn-esm"



# python run_protran_sklearn.py --dataset_path="data/proeng/gb1/two_vs_rest.csv" --encoder_name="esm1_t6_43M_UR50S" --checkpoint=1 --embed_batch_size=64 --flatten_emb="mean" --embed_folder="embeddings" --seq_end_idx=56 --all_embed_layers=True --sklearn_params='{"normalize":true}' --all_result_folder="results/sklearn-esm" --reset_param=True
# python run_protran_sklearn.py --dataset_path="data/proeng/gb1/two_vs_rest.csv" --encoder_name="esm1_t6_43M_UR50S" --checkpoint=1 --embed_batch_size=64 --flatten_emb="mean" --embed_folder="embeddings" --seq_end_idx=56 --all_embed_layers=True --sklearn_params='{"normalize":true}' --all_result_folder="results/sklearn-esm" --resample_param=True
# python run_protran_sklearn.py --dataset_path="data/proeng/gb1/two_vs_rest.csv" --encoder_name="esm1_t6_43M_UR50S" --checkpoint=1 --embed_batch_size=64 --flatten_emb="mean" --embed_folder="embeddings" --seq_end_idx=56 --all_embed_layers=True --sklearn_params='{"normalize":true}' --all_result_folder="results/sklearn-esm"

# python run_protran_sklearn.py --dataset_path="data/proeng/gb1/two_vs_rest.csv" --encoder_name="esm1_t12_85M_UR50S" --checkpoint=1 --embed_batch_size=64 --flatten_emb="mean" --embed_folder="embeddings" --seq_end_idx=56 --all_embed_layers=True --sklearn_params='{"normalize":true}' --all_result_folder="results/sklearn-esm" --reset_param=True
# python run_protran_sklearn.py --dataset_path="data/proeng/gb1/two_vs_rest.csv" --encoder_name="esm1_t12_85M_UR50S" --checkpoint=1 --embed_batch_size=64 --flatten_emb="mean" --embed_folder="embeddings" --seq_end_idx=56 --all_embed_layers=True --sklearn_params='{"normalize":true}' --all_result_folder="results/sklearn-esm" --resample_param=True
# python run_protran_sklearn.py --dataset_path="data/proeng/gb1/two_vs_rest.csv" --encoder_name="esm1_t12_85M_UR50S" --checkpoint=1 --embed_batch_size=64 --flatten_emb="mean" --embed_folder="embeddings" --seq_end_idx=56 --all_embed_layers=True --sklearn_params='{"normalize":true}' --all_result_folder="results/sklearn-esm"

# python run_protran_sklearn.py --dataset_path="data/proeng/gb1/two_vs_rest.csv" --encoder_name="esm1_t34_670M_UR50S" --checkpoint=1 --embed_batch_size=64 --flatten_emb="mean" --embed_folder="embeddings" --seq_end_idx=56 --all_embed_layers=True --sklearn_params='{"normalize":true}' --all_result_folder="results/sklearn-esm" --reset_param=True
# python run_protran_sklearn.py --dataset_path="data/proeng/gb1/two_vs_rest.csv" --encoder_name="esm1_t34_670M_UR50S" --checkpoint=1 --embed_batch_size=64 --flatten_emb="mean" --embed_folder="embeddings" --seq_end_idx=56 --all_embed_layers=True --sklearn_params='{"normalize":true}' --all_result_folder="results/sklearn-esm" --resample_param=True
# python run_protran_sklearn.py --dataset_path="data/proeng/gb1/two_vs_rest.csv" --encoder_name="esm1_t34_670M_UR50S" --checkpoint=1 --embed_batch_size=64 --flatten_emb="mean" --embed_folder="embeddings" --seq_end_idx=56 --all_embed_layers=True --sklearn_params='{"normalize":true}' --all_result_folder="results/sklearn-esm"

# python run_protran_sklearn.py --dataset_path="data/proeng/gb1/two_vs_rest.csv" --encoder_name="esm1b_t33_650M_UR50S" --checkpoint=1 --embed_batch_size=64 --flatten_emb="mean" --embed_folder="embeddings" --seq_end_idx=56 --all_embed_layers=True --sklearn_params='{"normalize":true}' --all_result_folder="results/sklearn-esm" --reset_param=True
# python run_protran_sklearn.py --dataset_path="data/proeng/gb1/two_vs_rest.csv" --encoder_name="esm1b_t33_650M_UR50S" --checkpoint=1 --embed_batch_size=64 --flatten_emb="mean" --embed_folder="embeddings" --seq_end_idx=56 --all_embed_layers=True --sklearn_params='{"normalize":true}' --all_result_folder="results/sklearn-esm" --resample_param=True
# python run_protran_sklearn.py --dataset_path="data/proeng/gb1/two_vs_rest.csv" --encoder_name="esm1b_t33_650M_UR50S" --checkpoint=1 --embed_batch_size=64 --flatten_emb="mean" --embed_folder="embeddings" --seq_end_idx=56 --all_embed_layers=True --sklearn_params='{"normalize":true}' --all_result_folder="results/sklearn-esm"


# ### AAV ###

# python run_protran_sklearn.py --dataset_path="data/proeng/aav/two_vs_many.csv" --encoder_name="esm1_t6_43M_UR50S" --checkpoint=1 --embed_batch_size=64 --flatten_emb="mean" --embed_folder="embeddings" --all_embed_layers=True --sklearn_params='{"normalize":true}' --all_result_folder="results/sklearn-esm" --reset_param=True
# python run_protran_sklearn.py --dataset_path="data/proeng/aav/two_vs_many.csv" --encoder_name="esm1_t6_43M_UR50S" --checkpoint=1 --embed_batch_size=64 --flatten_emb="mean" --embed_folder="embeddings" --all_embed_layers=True --sklearn_params='{"normalize":true}' --all_result_folder="results/sklearn-esm" --resample_param=True
# python run_protran_sklearn.py --dataset_path="data/proeng/aav/two_vs_many.csv" --encoder_name="esm1_t6_43M_UR50S" --checkpoint=1 --embed_batch_size=64 --flatten_emb="mean" --embed_folder="embeddings" --all_embed_layers=True --sklearn_params='{"normalize":true}' --all_result_folder="results/sklearn-esm"

# python run_protran_sklearn.py --dataset_path="data/proeng/aav/two_vs_many.csv" --encoder_name="esm1_t12_85M_UR50S" --checkpoint=1 --embed_batch_size=64 --flatten_emb="mean" --embed_folder="embeddings" --all_embed_layers=True --sklearn_params='{"normalize":true}' --all_result_folder="results/sklearn-esm" --reset_param=True
# python run_protran_sklearn.py --dataset_path="data/proeng/aav/two_vs_many.csv" --encoder_name="esm1_t12_85M_UR50S" --checkpoint=1 --embed_batch_size=64 --flatten_emb="mean" --embed_folder="embeddings" --all_embed_layers=True --sklearn_params='{"normalize":true}' --all_result_folder="results/sklearn-esm" --resample_param=True
# python run_protran_sklearn.py --dataset_path="data/proeng/aav/two_vs_many.csv" --encoder_name="esm1_t12_85M_UR50S" --checkpoint=1 --embed_batch_size=64 --flatten_emb="mean" --embed_folder="embeddings" --all_embed_layers=True --sklearn_params='{"normalize":true}' --all_result_folder="results/sklearn-esm"

# python run_protran_sklearn.py --dataset_path="data/proeng/aav/two_vs_many.csv" --encoder_name="esm1_t34_670M_UR50S" --checkpoint=1 --embed_batch_size=64 --flatten_emb="mean" --embed_folder="embeddings" --all_embed_layers=True --sklearn_params='{"normalize":true}' --all_result_folder="results/sklearn-esm" --reset_param=True
# python run_protran_sklearn.py --dataset_path="data/proeng/aav/two_vs_many.csv" --encoder_name="esm1_t34_670M_UR50S" --checkpoint=1 --embed_batch_size=64 --flatten_emb="mean" --embed_folder="embeddings" --all_embed_layers=True --sklearn_params='{"normalize":true}' --all_result_folder="results/sklearn-esm" --resample_param=True
# python run_protran_sklearn.py --dataset_path="data/proeng/aav/two_vs_many.csv" --encoder_name="esm1_t34_670M_UR50S" --checkpoint=1 --embed_batch_size=64 --flatten_emb="mean" --embed_folder="embeddings" --all_embed_layers=True --sklearn_params='{"normalize":true}' --all_result_folder="results/sklearn-esm"

# python run_protran_sklearn.py --dataset_path="data/proeng/aav/two_vs_many.csv" --encoder_name="esm1b_t33_650M_UR50S" --checkpoint=1 --embed_batch_size=64 --flatten_emb="mean" --embed_folder="embeddings" --all_embed_layers=True --sklearn_params='{"normalize":true}' --all_result_folder="results/sklearn-esm" --reset_param=True
# python run_protran_sklearn.py --dataset_path="data/proeng/aav/two_vs_many.csv" --encoder_name="esm1b_t33_650M_UR50S" --checkpoint=1 --embed_batch_size=64 --flatten_emb="mean" --embed_folder="embeddings" --all_embed_layers=True --sklearn_params='{"normalize":true}' --all_result_folder="results/sklearn-esm" --resample_param=True
# python run_protran_sklearn.py --dataset_path="data/proeng/aav/two_vs_many.csv" --encoder_name="esm1b_t33_650M_UR50S" --checkpoint=1 --embed_batch_size=64 --flatten_emb="mean" --embed_folder="embeddings" --all_embed_layers=True --sklearn_params='{"normalize":true}' --all_result_folder="results/sklearn-esm"


# python run_protran_sklearn.py --dataset_path="data/proeng/aav/one_vs_many.csv" --encoder_name="esm1_t6_43M_UR50S" --checkpoint=1 --embed_batch_size=64 --flatten_emb="mean" --embed_folder="embeddings" --all_embed_layers=True --sklearn_params='{"normalize":true}' --all_result_folder="results/sklearn-esm" --reset_param=True
# python run_protran_sklearn.py --dataset_path="data/proeng/aav/one_vs_many.csv" --encoder_name="esm1_t6_43M_UR50S" --checkpoint=1 --embed_batch_size=64 --flatten_emb="mean" --embed_folder="embeddings" --all_embed_layers=True --sklearn_params='{"normalize":true}' --all_result_folder="results/sklearn-esm" --resample_param=True
# python run_protran_sklearn.py --dataset_path="data/proeng/aav/one_vs_many.csv" --encoder_name="esm1_t6_43M_UR50S" --checkpoint=1 --embed_batch_size=64 --flatten_emb="mean" --embed_folder="embeddings" --all_embed_layers=True --sklearn_params='{"normalize":true}' --all_result_folder="results/sklearn-esm"

# python run_protran_sklearn.py --dataset_path="data/proeng/aav/one_vs_many.csv" --encoder_name="esm1_t12_85M_UR50S" --checkpoint=1 --embed_batch_size=64 --flatten_emb="mean" --embed_folder="embeddings" --all_embed_layers=True --sklearn_params='{"normalize":true}' --all_result_folder="results/sklearn-esm" --reset_param=True
# python run_protran_sklearn.py --dataset_path="data/proeng/aav/one_vs_many.csv" --encoder_name="esm1_t12_85M_UR50S" --checkpoint=1 --embed_batch_size=64 --flatten_emb="mean" --embed_folder="embeddings" --all_embed_layers=True --sklearn_params='{"normalize":true}' --all_result_folder="results/sklearn-esm" --resample_param=True
# python run_protran_sklearn.py --dataset_path="data/proeng/aav/one_vs_many.csv" --encoder_name="esm1_t12_85M_UR50S" --checkpoint=1 --embed_batch_size=64 --flatten_emb="mean" --embed_folder="embeddings" --all_embed_layers=True --sklearn_params='{"normalize":true}' --all_result_folder="results/sklearn-esm"

# python run_protran_sklearn.py --dataset_path="data/proeng/aav/one_vs_many.csv" --encoder_name="esm1_t34_670M_UR50S" --checkpoint=1 --embed_batch_size=64 --flatten_emb="mean" --embed_folder="embeddings" --all_embed_layers=True --sklearn_params='{"normalize":true}' --all_result_folder="results/sklearn-esm" --reset_param=True
# python run_protran_sklearn.py --dataset_path="data/proeng/aav/one_vs_many.csv" --encoder_name="esm1_t34_670M_UR50S" --checkpoint=1 --embed_batch_size=64 --flatten_emb="mean" --embed_folder="embeddings" --all_embed_layers=True --sklearn_params='{"normalize":true}' --all_result_folder="results/sklearn-esm" --resample_param=True
# python run_protran_sklearn.py --dataset_path="data/proeng/aav/one_vs_many.csv" --encoder_name="esm1_t34_670M_UR50S" --checkpoint=1 --embed_batch_size=64 --flatten_emb="mean" --embed_folder="embeddings" --all_embed_layers=True --sklearn_params='{"normalize":true}' --all_result_folder="results/sklearn-esm"

# python run_protran_sklearn.py --dataset_path="data/proeng/aav/one_vs_many.csv" --encoder_name="esm1b_t33_650M_UR50S" --checkpoint=1 --embed_batch_size=64 --flatten_emb="mean" --embed_folder="embeddings" --all_embed_layers=True --sklearn_params='{"normalize":true}' --all_result_folder="results/sklearn-esm" --reset_param=True
# python run_protran_sklearn.py --dataset_path="data/proeng/aav/one_vs_many.csv" --encoder_name="esm1b_t33_650M_UR50S" --checkpoint=1 --embed_batch_size=64 --flatten_emb="mean" --embed_folder="embeddings" --all_embed_layers=True --sklearn_params='{"normalize":true}' --all_result_folder="results/sklearn-esm" --resample_param=True
# python run_protran_sklearn.py --dataset_path="data/proeng/aav/one_vs_many.csv" --encoder_name="esm1b_t33_650M_UR50S" --checkpoint=1 --embed_batch_size=64 --flatten_emb="mean" --embed_folder="embeddings" --all_embed_layers=True --sklearn_params='{"normalize":true}' --all_result_folder="results/sklearn-esm"


# ### thermo ### 

# python run_protran_sklearn.py --dataset_path="data/proeng/thermo/mixed_split.csv" --encoder_name="esm1_t6_43M_UR50S" --checkpoint=1 --embed_batch_size=64 --flatten_emb="mean" --embed_folder="embeddings" --all_embed_layers=True --sklearn_params='{"normalize":true}' --all_result_folder="results/sklearn-esm" --reset_param=True
# python run_protran_sklearn.py --dataset_path="data/proeng/thermo/mixed_split.csv" --encoder_name="esm1_t6_43M_UR50S" --checkpoint=1 --embed_batch_size=64 --flatten_emb="mean" --embed_folder="embeddings" --all_embed_layers=True --sklearn_params='{"normalize":true}' --all_result_folder="results/sklearn-esm" --resample_param=True
# python run_protran_sklearn.py --dataset_path="data/proeng/thermo/mixed_split.csv" --encoder_name="esm1_t6_43M_UR50S" --checkpoint=1 --embed_batch_size=64 --flatten_emb="mean" --embed_folder="embeddings" --all_embed_layers=True --sklearn_params='{"normalize":true}' --all_result_folder="results/sklearn-esm"

# python run_protran_sklearn.py --dataset_path="data/proeng/thermo/mixed_split.csv" --encoder_name="esm1_t12_85M_UR50S" --checkpoint=1 --embed_batch_size=64 --flatten_emb="mean" --embed_folder="embeddings" --all_embed_layers=True --sklearn_params='{"normalize":true}' --all_result_folder="results/sklearn-esm" --reset_param=True
# python run_protran_sklearn.py --dataset_path="data/proeng/thermo/mixed_split.csv" --encoder_name="esm1_t12_85M_UR50S" --checkpoint=1 --embed_batch_size=64 --flatten_emb="mean" --embed_folder="embeddings" --all_embed_layers=True --sklearn_params='{"normalize":true}' --all_result_folder="results/sklearn-esm" --resample_param=True
# python run_protran_sklearn.py --dataset_path="data/proeng/thermo/mixed_split.csv" --encoder_name="esm1_t12_85M_UR50S" --checkpoint=1 --embed_batch_size=64 --flatten_emb="mean" --embed_folder="embeddings" --all_embed_layers=True --sklearn_params='{"normalize":true}' --all_result_folder="results/sklearn-esm"

# python run_protran_sklearn.py --dataset_path="data/proeng/thermo/mixed_split.csv" --encoder_name="esm1_t34_670M_UR50S" --checkpoint=1 --embed_batch_size=64 --flatten_emb="mean" --embed_folder="embeddings" --all_embed_layers=True --sklearn_params='{"normalize":true}' --all_result_folder="results/sklearn-esm" --reset_param=True
# python run_protran_sklearn.py --dataset_path="data/proeng/thermo/mixed_split.csv" --encoder_name="esm1_t34_670M_UR50S" --checkpoint=1 --embed_batch_size=64 --flatten_emb="mean" --embed_folder="embeddings" --all_embed_layers=True --sklearn_params='{"normalize":true}' --all_result_folder="results/sklearn-esm" --resample_param=True
# python run_protran_sklearn.py --dataset_path="data/proeng/thermo/mixed_split.csv" --encoder_name="esm1_t34_670M_UR50S" --checkpoint=1 --embed_batch_size=64 --flatten_emb="mean" --embed_folder="embeddings" --all_embed_layers=True --sklearn_params='{"normalize":true}' --all_result_folder="results/sklearn-esm"

# python run_protran_sklearn.py --dataset_path="data/proeng/thermo/mixed_split.csv" --encoder_name="esm1b_t33_650M_UR50S" --checkpoint=1 --embed_batch_size=64 --flatten_emb="mean" --embed_folder="embeddings" --all_embed_layers=True --sklearn_params='{"normalize":true}' --all_result_folder="results/sklearn-esm" --reset_param=True
# python run_protran_sklearn.py --dataset_path="data/proeng/thermo/mixed_split.csv" --encoder_name="esm1b_t33_650M_UR50S" --checkpoint=1 --embed_batch_size=64 --flatten_emb="mean" --embed_folder="embeddings" --all_embed_layers=True --sklearn_params='{"normalize":true}' --all_result_folder="results/sklearn-esm" --resample_param=True
# python run_protran_sklearn.py --dataset_path="data/proeng/thermo/mixed_split.csv" --encoder_name="esm1b_t33_650M_UR50S" --checkpoint=1 --embed_batch_size=64 --flatten_emb="mean" --embed_folder="embeddings" --all_embed_layers=True --sklearn_params='{"normalize":true}' --all_result_folder="results/sklearn-esm"



# python run_protran_sklearn.py --dataset_path="data/annotation/scl/balanced.csv" --encoder_name="esm1_t6_43M_UR50S" --checkpoint=1 --embed_batch_size=64 --flatten_emb="mean" --embed_folder="embeddings" --all_embed_layers=True --sklearn_params='{"normalize":true}' --all_result_folder="results/sklearn-esm" --reset_param=True
# python run_protran_sklearn.py --dataset_path="data/annotation/scl/balanced.csv" --encoder_name="esm1_t6_43M_UR50S" --checkpoint=1 --embed_batch_size=64 --flatten_emb="mean" --embed_folder="embeddings" --all_embed_layers=True --sklearn_params='{"normalize":true}' --all_result_folder="results/sklearn-esm" --resample_param=True
# python run_protran_sklearn.py --dataset_path="data/annotation/scl/balanced.csv" --encoder_name="esm1_t6_43M_UR50S" --checkpoint=1 --embed_batch_size=64 --flatten_emb="mean" --embed_folder="embeddings" --all_embed_layers=True --sklearn_params='{"normalize":true}' --all_result_folder="results/sklearn-esm"

# python run_protran_sklearn.py --dataset_path="data/annotation/scl/balanced.csv" --encoder_name="esm1_t12_85M_UR50S" --checkpoint=1 --embed_batch_size=64 --flatten_emb="mean" --embed_folder="embeddings" --all_embed_layers=True --sklearn_params='{"normalize":true}' --all_result_folder="results/sklearn-esm" --reset_param=True
# python run_protran_sklearn.py --dataset_path="data/annotation/scl/balanced.csv" --encoder_name="esm1_t12_85M_UR50S" --checkpoint=1 --embed_batch_size=64 --flatten_emb="mean" --embed_folder="embeddings" --all_embed_layers=True --sklearn_params='{"normalize":true}' --all_result_folder="results/sklearn-esm" --resample_param=True
# python run_protran_sklearn.py --dataset_path="data/annotation/scl/balanced.csv" --encoder_name="esm1_t12_85M_UR50S" --checkpoint=1 --embed_batch_size=64 --flatten_emb="mean" --embed_folder="embeddings" --all_embed_layers=True --sklearn_params='{"normalize":true}' --all_result_folder="results/sklearn-esm"

# python run_protran_sklearn.py --dataset_path="data/annotation/scl/balanced.csv" --encoder_name="esm1_t34_670M_UR50S" --checkpoint=1 --embed_batch_size=64 --flatten_emb="mean" --embed_folder="embeddings" --all_embed_layers=True --sklearn_params='{"normalize":true}' --all_result_folder="results/sklearn-esm" --reset_param=True
# python run_protran_sklearn.py --dataset_path="data/annotation/scl/balanced.csv" --encoder_name="esm1_t34_670M_UR50S" --checkpoint=1 --embed_batch_size=64 --flatten_emb="mean" --embed_folder="embeddings" --all_embed_layers=True --sklearn_params='{"normalize":true}' --all_result_folder="results/sklearn-esm" --resample_param=True
# python run_protran_sklearn.py --dataset_path="data/annotation/scl/balanced.csv" --encoder_name="esm1_t34_670M_UR50S" --checkpoint=1 --embed_batch_size=64 --flatten_emb="mean" --embed_folder="embeddings" --all_embed_layers=True --sklearn_params='{"normalize":true}' --all_result_folder="results/sklearn-esm"

# python run_protran_sklearn.py --dataset_path="data/annotation/scl/balanced.csv" --encoder_name="esm1b_t33_650M_UR50S" --checkpoint=1 --embed_batch_size=64 --flatten_emb="mean" --embed_folder="embeddings" --all_embed_layers=True --sklearn_params='{"normalize":true}' --all_result_folder="results/sklearn-esm" --reset_param=True
# python run_protran_sklearn.py --dataset_path="data/annotation/scl/balanced.csv" --encoder_name="esm1b_t33_650M_UR50S" --checkpoint=1 --embed_batch_size=64 --flatten_emb="mean" --embed_folder="embeddings" --all_embed_layers=True --sklearn_params='{"normalize":true}' --all_result_folder="results/sklearn-esm" --resample_param=True
# python run_protran_sklearn.py --dataset_path="data/annotation/scl/balanced.csv" --encoder_name="esm1b_t33_650M_UR50S" --checkpoint=1 --embed_batch_size=64 --flatten_emb="mean" --embed_folder="embeddings" --all_embed_layers=True --sklearn_params='{"normalize":true}' --all_result_folder="results/sklearn-esm"