# queue for generating and sorting all embs

### GB1 ###

python run_pregen_emb.py --dataset_path="data/proeng/gb1/sampled.csv" --encoder_name="esm1_t6_43M_UR50S" --flatten_emb="mean" --seq_end_idx=56 --reset_param=True
python run_pregen_emb.py --dataset_path="data/proeng/gb1/sampled.csv" --encoder_name="esm1_t6_43M_UR50S" --flatten_emb="mean" --seq_end_idx=56 --resample_param=True
python run_pregen_emb.py --dataset_path="data/proeng/gb1/sampled.csv" --encoder_name="esm1_t6_43M_UR50S" --flatten_emb="mean" --seq_end_idx=56

python run_pregen_emb.py --dataset_path="data/proeng/gb1/sampled.csv" --encoder_name="esm1_t12_85M_UR50S" --flatten_emb="mean" --seq_end_idx=56 --reset_param=True
python run_pregen_emb.py --dataset_path="data/proeng/gb1/sampled.csv" --encoder_name="esm1_t12_85M_UR50S" --flatten_emb="mean" --seq_end_idx=56 --resample_param=True
python run_pregen_emb.py --dataset_path="data/proeng/gb1/sampled.csv" --encoder_name="esm1_t12_85M_UR50S" --flatten_emb="mean" --seq_end_idx=56

python run_pregen_emb.py --dataset_path="data/proeng/gb1/sampled.csv" --encoder_name="esm1_t34_670M_UR50S" --flatten_emb="mean" --seq_end_idx=56 --reset_param=True
python run_pregen_emb.py --dataset_path="data/proeng/gb1/sampled.csv" --encoder_name="esm1_t34_670M_UR50S" --flatten_emb="mean" --seq_end_idx=56 --resample_param=True
python run_pregen_emb.py --dataset_path="data/proeng/gb1/sampled.csv" --encoder_name="esm1_t34_670M_UR50S" --flatten_emb="mean" --seq_end_idx=56

python run_pregen_emb.py --dataset_path="data/proeng/gb1/sampled.csv" --encoder_name="esm1b_t33_650M_UR50S" --flatten_emb="mean" --seq_end_idx=56 --reset_param=True
python run_pregen_emb.py --dataset_path="data/proeng/gb1/sampled.csv" --encoder_name="esm1b_t33_650M_UR50S" --flatten_emb="mean" --seq_end_idx=56 --resample_param=True
python run_pregen_emb.py --dataset_path="data/proeng/gb1/sampled.csv" --encoder_name="esm1b_t33_650M_UR50S" --flatten_emb="mean" --seq_end_idx=56



python run_pregen_emb.py --dataset_path="data/proeng/gb1/low_vs_high.csv" --encoder_name="esm1_t6_43M_UR50S" --flatten_emb="mean" --seq_end_idx=56 --reset_param=True
python run_pregen_emb.py --dataset_path="data/proeng/gb1/low_vs_high.csv" --encoder_name="esm1_t6_43M_UR50S" --flatten_emb="mean" --seq_end_idx=56 --resample_param=True
python run_pregen_emb.py --dataset_path="data/proeng/gb1/low_vs_high.csv" --encoder_name="esm1_t6_43M_UR50S" --flatten_emb="mean" --seq_end_idx=56

python run_pregen_emb.py --dataset_path="data/proeng/gb1/low_vs_high.csv" --encoder_name="esm1_t12_85M_UR50S" --flatten_emb="mean" --seq_end_idx=56 --reset_param=True
python run_pregen_emb.py --dataset_path="data/proeng/gb1/low_vs_high.csv" --encoder_name="esm1_t12_85M_UR50S" --flatten_emb="mean" --seq_end_idx=56 --resample_param=True
python run_pregen_emb.py --dataset_path="data/proeng/gb1/low_vs_high.csv" --encoder_name="esm1_t12_85M_UR50S" --flatten_emb="mean" --seq_end_idx=56

python run_pregen_emb.py --dataset_path="data/proeng/gb1/low_vs_high.csv" --encoder_name="esm1_t34_670M_UR50S" --flatten_emb="mean" --seq_end_idx=56 --reset_param=True
python run_pregen_emb.py --dataset_path="data/proeng/gb1/low_vs_high.csv" --encoder_name="esm1_t34_670M_UR50S" --flatten_emb="mean" --seq_end_idx=56 --resample_param=True
python run_pregen_emb.py --dataset_path="data/proeng/gb1/low_vs_high.csv" --encoder_name="esm1_t34_670M_UR50S" --flatten_emb="mean" --seq_end_idx=56

python run_pregen_emb.py --dataset_path="data/proeng/gb1/low_vs_high.csv" --encoder_name="esm1b_t33_650M_UR50S" --flatten_emb="mean" --seq_end_idx=56 --reset_param=True
python run_pregen_emb.py --dataset_path="data/proeng/gb1/low_vs_high.csv" --encoder_name="esm1b_t33_650M_UR50S" --flatten_emb="mean" --seq_end_idx=56 --resample_param=True
python run_pregen_emb.py --dataset_path="data/proeng/gb1/low_vs_high.csv" --encoder_name="esm1b_t33_650M_UR50S" --flatten_emb="mean" --seq_end_idx=56



python run_pregen_emb.py --dataset_path="data/proeng/gb1/two_vs_rest.csv" --encoder_name="esm1_t6_43M_UR50S" --flatten_emb="mean" --seq_end_idx=56 --reset_param=True
python run_pregen_emb.py --dataset_path="data/proeng/gb1/two_vs_rest.csv" --encoder_name="esm1_t6_43M_UR50S" --flatten_emb="mean" --seq_end_idx=56 --resample_param=True
python run_pregen_emb.py --dataset_path="data/proeng/gb1/two_vs_rest.csv" --encoder_name="esm1_t6_43M_UR50S" --flatten_emb="mean" --seq_end_idx=56

python run_pregen_emb.py --dataset_path="data/proeng/gb1/two_vs_rest.csv" --encoder_name="esm1_t12_85M_UR50S" --flatten_emb="mean" --seq_end_idx=56 --reset_param=True
python run_pregen_emb.py --dataset_path="data/proeng/gb1/two_vs_rest.csv" --encoder_name="esm1_t12_85M_UR50S" --flatten_emb="mean" --seq_end_idx=56 --resample_param=True
python run_pregen_emb.py --dataset_path="data/proeng/gb1/two_vs_rest.csv" --encoder_name="esm1_t12_85M_UR50S" --flatten_emb="mean" --seq_end_idx=56

python run_pregen_emb.py --dataset_path="data/proeng/gb1/two_vs_rest.csv" --encoder_name="esm1_t34_670M_UR50S" --flatten_emb="mean" --seq_end_idx=56 --reset_param=True
python run_pregen_emb.py --dataset_path="data/proeng/gb1/two_vs_rest.csv" --encoder_name="esm1_t34_670M_UR50S" --flatten_emb="mean" --seq_end_idx=56 --resample_param=True
python run_pregen_emb.py --dataset_path="data/proeng/gb1/two_vs_rest.csv" --encoder_name="esm1_t34_670M_UR50S" --flatten_emb="mean" --seq_end_idx=56

python run_pregen_emb.py --dataset_path="data/proeng/gb1/two_vs_rest.csv" --encoder_name="esm1b_t33_650M_UR50S" --flatten_emb="mean" --seq_end_idx=56 --reset_param=True
python run_pregen_emb.py --dataset_path="data/proeng/gb1/two_vs_rest.csv" --encoder_name="esm1b_t33_650M_UR50S" --flatten_emb="mean" --seq_end_idx=56 --resample_param=True
python run_pregen_emb.py --dataset_path="data/proeng/gb1/two_vs_rest.csv" --encoder_name="esm1b_t33_650M_UR50S" --flatten_emb="mean" --seq_end_idx=56



### AAV ###
python run_pregen_emb.py --dataset_path="data/proeng/aav/two_vs_many.csv" --encoder_name="esm1_t6_43M_UR50S" --flatten_emb="mean" --reset_param=True
python run_pregen_emb.py --dataset_path="data/proeng/aav/two_vs_many.csv" --encoder_name="esm1_t6_43M_UR50S" --flatten_emb="mean" --resample_param=True
python run_pregen_emb.py --dataset_path="data/proeng/aav/two_vs_many.csv" --encoder_name="esm1_t6_43M_UR50S" --flatten_emb="mean"

python run_pregen_emb.py --dataset_path="data/proeng/aav/two_vs_many.csv" --encoder_name="esm1_t12_85M_UR50S" --flatten_emb="mean" --reset_param=True
python run_pregen_emb.py --dataset_path="data/proeng/aav/two_vs_many.csv" --encoder_name="esm1_t12_85M_UR50S" --flatten_emb="mean" --resample_param=True
python run_pregen_emb.py --dataset_path="data/proeng/aav/two_vs_many.csv" --encoder_name="esm1_t12_85M_UR50S" --flatten_emb="mean"

python run_pregen_emb.py --dataset_path="data/proeng/aav/two_vs_many.csv" --encoder_name="esm1_t34_670M_UR50S" --flatten_emb="mean" --reset_param=True
python run_pregen_emb.py --dataset_path="data/proeng/aav/two_vs_many.csv" --encoder_name="esm1_t34_670M_UR50S" --flatten_emb="mean" --resample_param=True
python run_pregen_emb.py --dataset_path="data/proeng/aav/two_vs_many.csv" --encoder_name="esm1_t34_670M_UR50S" --flatten_emb="mean"

python run_pregen_emb.py --dataset_path="data/proeng/aav/two_vs_many.csv" --encoder_name="esm1b_t33_650M_UR50S" --flatten_emb="mean" --reset_param=True
python run_pregen_emb.py --dataset_path="data/proeng/aav/two_vs_many.csv" --encoder_name="esm1b_t33_650M_UR50S" --flatten_emb="mean" --resample_param=True
python run_pregen_emb.py --dataset_path="data/proeng/aav/two_vs_many.csv" --encoder_name="esm1b_t33_650M_UR50S" --flatten_emb="mean"



python run_pregen_emb.py --dataset_path="data/proeng/aav/one_vs_many.csv" --encoder_name="esm1_t6_43M_UR50S" --flatten_emb="mean" --reset_param=True
python run_pregen_emb.py --dataset_path="data/proeng/aav/one_vs_many.csv" --encoder_name="esm1_t6_43M_UR50S" --flatten_emb="mean" --resample_param=True
python run_pregen_emb.py --dataset_path="data/proeng/aav/one_vs_many.csv" --encoder_name="esm1_t6_43M_UR50S" --flatten_emb="mean"

python run_pregen_emb.py --dataset_path="data/proeng/aav/one_vs_many.csv" --encoder_name="esm1_t12_85M_UR50S" --flatten_emb="mean" --reset_param=True
python run_pregen_emb.py --dataset_path="data/proeng/aav/one_vs_many.csv" --encoder_name="esm1_t12_85M_UR50S" --flatten_emb="mean" --resample_param=True
python run_pregen_emb.py --dataset_path="data/proeng/aav/one_vs_many.csv" --encoder_name="esm1_t12_85M_UR50S" --flatten_emb="mean"

python run_pregen_emb.py --dataset_path="data/proeng/aav/one_vs_many.csv" --encoder_name="esm1_t34_670M_UR50S" --flatten_emb="mean" --reset_param=True
python run_pregen_emb.py --dataset_path="data/proeng/aav/one_vs_many.csv" --encoder_name="esm1_t34_670M_UR50S" --flatten_emb="mean" --resample_param=True
python run_pregen_emb.py --dataset_path="data/proeng/aav/one_vs_many.csv" --encoder_name="esm1_t34_670M_UR50S" --flatten_emb="mean"

python run_pregen_emb.py --dataset_path="data/proeng/aav/one_vs_many.csv" --encoder_name="esm1b_t33_650M_UR50S" --flatten_emb="mean" --reset_param=True
python run_pregen_emb.py --dataset_path="data/proeng/aav/one_vs_many.csv" --encoder_name="esm1b_t33_650M_UR50S" --flatten_emb="mean" --resample_param=True
python run_pregen_emb.py --dataset_path="data/proeng/aav/one_vs_many.csv" --encoder_name="esm1b_t33_650M_UR50S" --flatten_emb="mean"


### thermo ###

python run_pregen_emb.py --dataset_path="data/proeng/thermo/mixed_split.csv" --encoder_name="esm1_t6_43M_UR50S" --flatten_emb="mean" --reset_param=True
python run_pregen_emb.py --dataset_path="data/proeng/thermo/mixed_split.csv" --encoder_name="esm1_t6_43M_UR50S" --flatten_emb="mean" --resample_param=True
python run_pregen_emb.py --dataset_path="data/proeng/thermo/mixed_split.csv" --encoder_name="esm1_t6_43M_UR50S" --flatten_emb="mean"

python run_pregen_emb.py --dataset_path="data/proeng/thermo/mixed_split.csv" --encoder_name="esm1_t12_85M_UR50S" --flatten_emb="mean" --reset_param=True
python run_pregen_emb.py --dataset_path="data/proeng/thermo/mixed_split.csv" --encoder_name="esm1_t12_85M_UR50S" --flatten_emb="mean" --resample_param=True
python run_pregen_emb.py --dataset_path="data/proeng/thermo/mixed_split.csv" --encoder_name="esm1_t12_85M_UR50S" --flatten_emb="mean"

python run_pregen_emb.py --dataset_path="data/proeng/thermo/mixed_split.csv" --encoder_name="esm1_t34_670M_UR50S" --flatten_emb="mean" --reset_param=True
python run_pregen_emb.py --dataset_path="data/proeng/thermo/mixed_split.csv" --encoder_name="esm1_t34_670M_UR50S" --flatten_emb="mean" --resample_param=True
python run_pregen_emb.py --dataset_path="data/proeng/thermo/mixed_split.csv" --encoder_name="esm1_t34_670M_UR50S" --flatten_emb="mean"

python run_pregen_emb.py --dataset_path="data/proeng/thermo/mixed_split.csv" --encoder_name="esm1b_t33_650M_UR50S" --flatten_emb="mean" --reset_param=True
python run_pregen_emb.py --dataset_path="data/proeng/thermo/mixed_split.csv" --encoder_name="esm1b_t33_650M_UR50S" --flatten_emb="mean" --resample_param=True
python run_pregen_emb.py --dataset_path="data/proeng/thermo/mixed_split.csv" --encoder_name="esm1b_t33_650M_UR50S" --flatten_emb="mean"


### Annotation ### 

python run_pregen_emb.py --dataset_path="data/annotation/scl/balanced.csv" --encoder_name="esm1_t6_43M_UR50S" --flatten_emb="mean" --reset_param=True
python run_pregen_emb.py --dataset_path="data/annotation/scl/balanced.csv" --encoder_name="esm1_t6_43M_UR50S" --flatten_emb="mean" --resample_param=True
python run_pregen_emb.py --dataset_path="data/annotation/scl/balanced.csv" --encoder_name="esm1_t6_43M_UR50S" --flatten_emb="mean"

python run_pregen_emb.py --dataset_path="data/annotation/scl/balanced.csv" --encoder_name="esm1_t12_85M_UR50S" --flatten_emb="mean" --reset_param=True
python run_pregen_emb.py --dataset_path="data/annotation/scl/balanced.csv" --encoder_name="esm1_t12_85M_UR50S" --flatten_emb="mean" --resample_param=True
python run_pregen_emb.py --dataset_path="data/annotation/scl/balanced.csv" --encoder_name="esm1_t12_85M_UR50S" --flatten_emb="mean"

python run_pregen_emb.py --dataset_path="data/annotation/scl/balanced.csv" --encoder_name="esm1_t34_670M_UR50S" --flatten_emb="mean" --reset_param=True
python run_pregen_emb.py --dataset_path="data/annotation/scl/balanced.csv" --encoder_name="esm1_t34_670M_UR50S" --flatten_emb="mean" --resample_param=True
python run_pregen_emb.py --dataset_path="data/annotation/scl/balanced.csv" --encoder_name="esm1_t34_670M_UR50S" --flatten_emb="mean"

python run_pregen_emb.py --dataset_path="data/annotation/scl/balanced.csv" --encoder_name="esm1b_t33_650M_UR50S" --flatten_emb="mean" --reset_param=True
python run_pregen_emb.py --dataset_path="data/annotation/scl/balanced.csv" --encoder_name="esm1b_t33_650M_UR50S" --flatten_emb="mean" --resample_param=True
python run_pregen_emb.py --dataset_path="data/annotation/scl/balanced.csv" --encoder_name="esm1b_t33_650M_UR50S" --flatten_emb="mean"
