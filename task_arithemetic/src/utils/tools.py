import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from collections import OrderedDict
from tqdm import tqdm
from functools import reduce

# import internal libs
from src.datasets.common import maybe_dictionarize
from src.utils import get_logits

def search_by_suffix(directory: str,
                     suffix: str) -> list:
    """find all the files with the suffix under the directory

    Args:
        directory (str): the directory to find the files
        suffix (str): the suffix of the files
    
    Returns:
        list: the list of the files
    """
    file_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(suffix):
                file_paths.append(os.path.join(root, file))
    return file_paths


def interpolate_weights(A: OrderedDict,
                        B: OrderedDict,
                        alpha: float,
                        beta: float,) -> OrderedDict:
    """interpolate the weights
    Args:
        A: the weights of model A
        B: the weights of model B
        alpha: the interpolation coefficient
        beta: the interpolation coefficient
    
    Returns:
        the interpolated weights
    """
    assert A.keys() == B.keys(), "the keys of A and B should be the same"
    C = OrderedDict()
    for k, v in A.items():
        if k.startswith("module."):
            k = k[7:]
        C[k] = alpha * v + beta * B[k]
    return C


def get_module(model: nn.Module, 
               module_name: str) -> nn.Module:
    """get the module from the model

    Args:
        model (nn.Module): the model to extract featuremaps.
        module_name (str): name of the module

    Returns:
        nn.Module: the module
    """
    return reduce(getattr, module_name.split('.'), model)


def evaluate(device: torch.device,
             model: nn.Module,
             dataloader: DataLoader,) -> tuple:
    """evaluate the model over the dataset

    Args:
        device (torch.device): the device to run the model.
        model (nn.Module): the model to be evaluated.
        dataloader (Dataloader): usually the test loader.
        
    Return: 
        (avg_acc, avg_loss, predictions)
    """
    # init the loss function
    loss_fn = nn.CrossEntropyLoss(reduction="none").to(device)
    # set the model to eval mode
    model.eval()
    # evaluate
    with torch.no_grad():
        predictions, loss_lst, corrects, n = [], [], 0., 0.
        for _, data in enumerate(tqdm(dataloader)):
            # put data to the device
            data = maybe_dictionarize(data)
            x = data['images'].to(device)
            y = data['labels'].to(device)

            # forward
            logits = get_logits(x, model)
            # get the loss
            losses = loss_fn(logits, y)
            # get the preds
            preds = logits.argmax(dim=1).to(device)

            # update
            predictions.extend(preds.tolist())
            loss_lst.extend(losses.tolist())
            corrects += preds.eq(y.view_as(preds)).sum().item()
            n += y.size(0)

        # get the average acc
        avg_acc = corrects / n
        avg_loss = sum(loss_lst) / len(loss_lst)
    
    return avg_acc, avg_loss, np.array(predictions)
