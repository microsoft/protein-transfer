from scr.utils import pickle_load
from scr.params.emb import TRANSFORMER_INFO

import os
from glob import glob
import numpy as np

def parse_result_dicts(folder_path: str, metric_list: list[str] = ["train_mse", "test_ndcg", "test_rho"]):
    pkl_list = glob(f"{folder_path}/*.pkl")
    print(len(pkl_list))
    
    # get the max layer number for the array
    model_name, _, _ = os.path.splitext(os.path.basename(pkl_list[0]))[0].split("-")
    max_layer_numb = TRANSFORMER_INFO[model_name][1] + 1
    
    output_numb_dict = {metric: np.zeros([max_layer_numb]) for metric in metric_list}

    for pkl_file in pkl_list:
        layer_numb = int(os.path.splitext(os.path.basename(pkl_file))[0].split("-")[1].split("_")[-1])
        result_dict = pickle_load(pkl_file)
        
        for metric in metric_list:
            subset, kind = metric.split("_")
            if kind == "rho":
                output_numb_dict[metric][layer_numb] = result_dict[subset][kind][0]
            else:
                output_numb_dict[metric][layer_numb] = result_dict[subset][kind]
         #print(output_numb_dict[metric])
    return output_numb_dict