# queue for running all pytorch models from embs

# export CUDA_VISIBLE_DEVICES=""
export CUDA_VISIBLE_DEVICES=1

### scl ###

# python run_protran_pytorch.py --dataset_path="data/annotation/scl/balanced.csv" --encoder_name="esm1_t6_43M_UR50S" --checkpoint=1 --embed_batch_size=64 --flatten_emb="mean" --embed_folder="embeddings" --loader_batch_size=256 --epochs=100 --all_plot_folder="results/pytorch_learning_curves-esm" --all_result_folder="results/pytorch-esm"
# python run_protran_pytorch.py --dataset_path="data/annotation/scl/balanced.csv" --encoder_name="esm1_t6_43M_UR50S" --checkpoint=1 --embed_batch_size=64 --flatten_emb="mean" --embed_folder="embeddings" --loader_batch_size=256 --epochs=100 --all_plot_folder="results/pytorch_learning_curves-esm" --all_result_folder="results/pytorch-esm" --reset_param=True
# python run_protran_pytorch.py --dataset_path="data/annotation/scl/balanced.csv" --encoder_name="esm1_t6_43M_UR50S" --checkpoint=1 --embed_batch_size=64 --flatten_emb="mean" --embed_folder="embeddings" --loader_batch_size=256 --epochs=100 --all_plot_folder="results/pytorch_learning_curves-esm" --all_result_folder="results/pytorch-esm" --resample_param=True

# python run_protran_pytorch.py --dataset_path="data/annotation/scl/balanced.csv" --encoder_name="esm1_t12_85M_UR50S" --checkpoint=1 --embed_batch_size=64 --flatten_emb="mean" --embed_folder="embeddings" --loader_batch_size=256 --epochs=100 --all_plot_folder="results/pytorch_learning_curves-esm" --all_result_folder="results/pytorch-esm"
# python run_protran_pytorch.py --dataset_path="data/annotation/scl/balanced.csv" --encoder_name="esm1_t12_85M_UR50S" --checkpoint=1 --embed_batch_size=64 --flatten_emb="mean" --embed_folder="embeddings" --loader_batch_size=256 --epochs=100 --all_plot_folder="results/pytorch_learning_curves-esm" --all_result_folder="results/pytorch-esm" --reset_param=True
# python run_protran_pytorch.py --dataset_path="data/annotation/scl/balanced.csv" --encoder_name="esm1_t12_85M_UR50S" --checkpoint=1 --embed_batch_size=64 --flatten_emb="mean" --embed_folder="embeddings" --loader_batch_size=256 --epochs=100 --all_plot_folder="results/pytorch_learning_curves-esm" --all_result_folder="results/pytorch-esm" --resample_param=True

# python run_protran_pytorch.py --dataset_path="data/annotation/scl/balanced.csv" --encoder_name="esm1_t34_670M_UR50S" --checkpoint=1 --embed_batch_size=64 --flatten_emb="mean" --embed_folder="embeddings" --loader_batch_size=256 --epochs=100 --all_plot_folder="results/pytorch_learning_curves-esm" --all_result_folder="results/pytorch-esm"
# python run_protran_pytorch.py --dataset_path="data/annotation/scl/balanced.csv" --encoder_name="esm1_t34_670M_UR50S" --checkpoint=1 --embed_batch_size=64 --flatten_emb="mean" --embed_folder="embeddings" --loader_batch_size=256 --epochs=100 --all_plot_folder="results/pytorch_learning_curves-esm" --all_result_folder="results/pytorch-esm" --reset_param=True &
# python run_protran_pytorch.py --dataset_path="data/annotation/scl/balanced.csv" --encoder_name="esm1_t34_670M_UR50S" --checkpoint=1 --embed_batch_size=64 --flatten_emb="mean" --embed_folder="embeddings" --loader_batch_size=256 --epochs=100 --all_plot_folder="results/pytorch_learning_curves-esm" --all_result_folder="results/pytorch-esm" --resample_param=True &

# python run_protran_pytorch.py --dataset_path="data/annotation/scl/balanced.csv" --encoder_name="esm1b_t33_650M_UR50S" --checkpoint=1 --embed_batch_size=64 --flatten_emb="mean" --embed_folder="embeddings" --loader_batch_size=256 --epochs=100 --all_plot_folder="results/pytorch_learning_curves-esm" --all_result_folder="results/pytorch-esm"
# python run_protran_pytorch.py --dataset_path="data/annotation/scl/balanced.csv" --encoder_name="esm1b_t33_650M_UR50S" --checkpoint=1 --embed_batch_size=64 --flatten_emb="mean" --embed_folder="embeddings" --loader_batch_size=256 --epochs=100 --all_plot_folder="results/pytorch_learning_curves-esm" --all_result_folder="results/pytorch-esm" --reset_param=True &
# python run_protran_pytorch.py --dataset_path="data/annotation/scl/balanced.csv" --encoder_name="esm1b_t33_650M_UR50S" --checkpoint=1 --embed_batch_size=64 --flatten_emb="mean" --embed_folder="embeddings" --loader_batch_size=256 --epochs=100 --all_plot_folder="results/pytorch_learning_curves-esm" --all_result_folder="results/pytorch-esm" --resample_param=True &

### ss3 ###
# python run_protran_pytorch.py --dataset_path="data/structure/secondary_structure/tape_ss3_processed.csv" --encoder_name="esm1_t6_43M_UR50S" --checkpoint=1 --embed_batch_size=64 --embed_folder="embeddings" --loader_batch_size=256 --epochs=100 --all_plot_folder="results/pytorch_learning_curves-esm" --all_result_folder="results/pytorch-esm"
python run_protran_pytorch.py --dataset_path="data/structure/secondary_structure/tape_ss3_processed.csv" --encoder_name="esm1_t6_43M_UR50S" --checkpoint=1 --embed_batch_size=64 --embed_folder="embeddings" --loader_batch_size=256 --epochs=100 --all_plot_folder="results/pytorch_learning_curves-esm" --all_result_folder="results/pytorch-esm" --reset_param=True --
python run_protran_pytorch.py --dataset_path="data/structure/secondary_structure/tape_ss3_processed.csv" --encoder_name="esm1_t6_43M_UR50S" --checkpoint=1 --embed_batch_size=64 --embed_folder="embeddings" --loader_batch_size=256 --epochs=100 --all_plot_folder="results/pytorch_learning_curves-esm" --all_result_folder="results/pytorch-esm" --resample_param=True

python run_protran_pytorch.py --dataset_path="data/structure/secondary_structure/tape_ss3_processed.csv" --encoder_name="" --checkpoint=1 --embed_batch_size=64 --loader_batch_size=256 --epochs=100 --all_plot_folder="results/pytorch_learning_curves-esm" --all_result_folder="results/pytorch-esm"
