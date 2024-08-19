import os
import re
import torch
from typing import Dict

# import internal libs
from src.task_vectors import TaskVector

def decode_model_name(model_name: str) -> tuple:
    """Decode the model name into scaling coefficient and task vectors info.

    Args:
        model_name: The model name.
        In the format of "scaling_coef+dataset1+dataset2+...+datasetN" or "scaling_coef_dataset1_dataset2_..._datasetN".
    
    Returns:
        scaling_coef: The scaling coefficient.
        task_vectors_info: A dictionary containing the task vectors information.
            where keys are the task names, and values are the labels indicating "add" or "minus".
    """
    # extract the scaling coefficient and task vectors info
    scaling_coef = re.search(r"(\d+\.\d+)", model_name).group()
    assert model_name[:len(scaling_coef)] == scaling_coef, \
        f"Invalid model_name {model_name}."
    
    # extract the task vectors info
    model_name = model_name.replace(scaling_coef, "")
    task_vectors_info = {}
    while len(model_name) > 0:
        # first char is the label, the first combination of letter and digit is the dataset name
        label, dataset_name = model_name[0], re.search(r"[a-zA-Z0-9]+", model_name).group()
        if label == "+":
            task_vectors_info[dataset_name] = "add"
        elif label == "_":
            task_vectors_info[dataset_name] = "minus"
        else:
            raise ValueError(f"Invalid model_name {model_name}.")
        model_name = model_name[1+len(dataset_name):]
    
    return float(scaling_coef), task_vectors_info


def load_image_encoder(model_root: str, 
                       task_vectors_info: Dict[str, str],
                       scaling_coef: float) -> torch.nn.Module:
    """Loads the image encoder from a pretrained model.

    Args:
        model_root: Path to the folder containing the pretrained model.
        task_vectors_info: A dictionary containing the task vectors information.
            where keys are the task names, and values are the labels indicating "add" or "minus".
    
    Returns:
        The image encoder.
    """
    assert scaling_coef >= -1.0 and scaling_coef <= 1.0, \
            f"scaling_coef should be in [-1.0, 1.0], but got {scaling_coef}."
    
    # create an empty task vector
    pretrained_checkpoint = os.path.join(model_root, "zeroshot.pt")
    task_vector = TaskVector(pretrained_checkpoint, pretrained_checkpoint)

    # iterate over the task vectors info
    for dataset_name, label in task_vectors_info.items():
        # get the task vector
        finetuned_checkpoint = os.path.join(model_root, f"{dataset_name}/finetuned.pt")
        task_vector_curr = TaskVector(pretrained_checkpoint, finetuned_checkpoint)

        # add or minus the task vector
        if label == "add":
            task_vector = task_vector + task_vector_curr
        elif label == "minus":
            task_vector = task_vector + (-task_vector_curr)
        else:
            raise ValueError(f"Unknown label {label}.")
    
    # apply the task vector to the pretrained model
    return task_vector.apply_to(pretrained_checkpoint, scaling_coef)

    
def load_image_encoder_single_task(model_root: str, 
                                   dataset_name: str,
                                   scaling_coef: float) -> torch.nn.Module:
    """Loads the image encoder from a pretrained model.

    Args:
        model_root: Path to the folder containing the pretrained model.
        task_vectors_info: A dictionary containing the task vectors information.
            where keys are the task names, and values are the labels indicating "add" or "minus".
        scaling_coef: The scaling coefficient.
            
    Returns:
        The image encoder.
    """
    assert scaling_coef >= -1.0 and scaling_coef <= 1.0, \
            f"scaling_coef should be in [-1.0, 1.0], but got {scaling_coef}."
    
    # create an empty task vector
    pretrained_checkpoint = os.path.join(model_root, "zeroshot.pt")

    # get the task vector
    finetuned_checkpoint = os.path.join(model_root, f"{dataset_name}/finetuned.pt")
    task_vector = TaskVector(pretrained_checkpoint, finetuned_checkpoint)
    
    return task_vector.apply_to(pretrained_checkpoint, scaling_coef)
