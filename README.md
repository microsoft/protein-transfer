# Code for Feature Reuse and Scaling: Understanding Transfer Learning with Protein Language Models 

Code for reproducing the analyses in our preprint "Feature Reuse and Scaling: Understanding Transfer Learning with Protein Language Models":
[PREPRINT LINK HERE]

For datasets and model checkpoints weights, see our Zenodo repository:
[ZENODO LINK HERE]

### Scripts to extract representations from pretrained models can be found under:
- run_pregen_embs.py - extracts representations from every layer in a model given a dataset and pretrained model
- execute_gen_emb_carp.sh - batch script to extract representations over all datasets and pretrained models for CARP
- execute_gen_emb_esm.sh - batch script to extract representations over all datasets and pretrained models for ESM

As these representations are time-consuming to extract, we provide them in our Zenodo repository ([ZENODO LINK HERE]). 

### Scripts to train linear models for each of the downstream tasks:
For the classification tasks (secondary structure and subcellular localization), we implement models in PyTorch:

- run_protran_pytorch.py - trains and evaluates classifiers for each layer in a model given a dataset and pretrained model
- execute_run_pytorch_carp.sh - batches over all datasets/models for CARP
- execute_run_pytorch_carp.sh - batches over all datasets/models for ESM

For the regression tasks (all other downstream tasks), we implement models in Scikit-Learn:

- run_protran_sklearn.py - trains and evaluates regressors for each layer in a model given a dataset and pretrained model
- execute_run_sklearn_carp.sh - batches over all datasets/models for CARP
- execute_run_sklearn_carp.sh - batches over all datasets/models for ESM

### Scripts to reproduce our analysis:
To produce the plots shown in our manuscript, run run_results_analysis.py

