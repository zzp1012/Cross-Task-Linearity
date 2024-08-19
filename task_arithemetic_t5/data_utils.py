import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
from datasets import load_dataset
from collections import defaultdict
from utils import get_logger
from featuremap import *
dataset_names = [("imdb",), ("race", "all"), ("qasc",), ("multi_news",), ("squad",), ("allenai/common_gen",)]


def get_evaluate_fn(dataset_name):
    if len([name[0] for name in dataset_names if dataset_name in name[0]]) == 0:
        raise ValueError(f"dataset_name should be in {dataset_names[:0]}")
    return eval(f"eval_{dataset_name}")



def process_hf_dataset_to_torch_dataset(hf_ds, dataset_name):
    if dataset_name == "imdb":
        hf_ds = hf_ds['test']
    elif dataset_name == "race":
        hf_ds = hf_ds['test']
    elif dataset_name == "qasc":
        hf_ds = hf_ds['test']
    elif dataset_name == "multi_news":
        hf_ds = hf_ds['test']
    elif dataset_name == "multi_news":
        hf_ds = hf_ds['test']
    elif dataset_name == "squad":
        hf_ds = hf_ds['validation']
    elif dataset_name == "common_gen":
        hf_ds = hf_ds['validation']
    else:
        raise NotImplementedError(f"dataset_name {dataset_name} not implemented.")
    ds_torch = hf_ds.with_format("torch")
    return ds_torch

def load_hf_dataset(dataset_name):
    if len([name[0] for name in dataset_names if dataset_name in name[0]]) == 0:
        raise ValueError(f"dataset_name should be in {dataset_names[:0]}")
    dataset_tuple = [name for name in dataset_names if dataset_name in name[0]][0]
    hf_ds = load_dataset(*dataset_tuple)
    torch_ds = process_hf_dataset_to_torch_dataset(hf_ds, dataset_name)
    return torch_ds