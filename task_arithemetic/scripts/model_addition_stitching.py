import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# cwd change to current file's dir
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import argparse
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from collections import OrderedDict
from open_clip import ResidualAttentionBlock

# import internal libs
from src.datasets.common import maybe_dictionarize
from src.modeling import ImageClassifier
from src.datasets.registry import get_dataset
from src.utils import get_datetime, set_logger, get_logger, set_seed, set_device, \
    log_settings, save_current_src, get_logits
from src.utils.avgmeter import MetricTracker
from src.utils.tools import get_module
from src.utils.featuremap import FeatureMap
from src.utils.load import load_image_encoder, decode_model_name



def get_featuremaps(device: torch.device,
                    weights: OrderedDict,
                    model: nn.Module,
                    dataloader: DataLoader,):
    """get the featuremaps of model A and model B

    Args:
        device (torch.device): the device to run the model.
        modelA_path (str): the path of model A.
        modelB_path (str): the path of model B.
        model (nn.Module): the model to extract featuremaps.
        dataloader (torch.utils.data.DataLoader): the dataloader.
        alpha (float): the interpolation coefficient.
        beta (float): the interpolation coefficient.

    Return:
        None
    """
    logger = get_logger(f"{__name__}.get_featuremaps")


    # set the layers
    layers = []
    for name, module in model.named_modules():
        if isinstance(module, (ResidualAttentionBlock)):
            layers.append(name)
    
    # get the featuremaps of model
    logger.info(f"get the featuremaps of model")
    model.image_encoder.load_state_dict(weights)
    
    # get the featuremaps
    fm = FeatureMap(device, model)
    featuremap = fm.get_featuremaps(dataloader, layer_names=layers)
    del fm
    return featuremap


def evaluate(device: torch.device,
             model: nn.Module,
             dataloader: DataLoader,) -> tuple:
    """evaluate the model over the dataset

    Args:
        device (torch.device): the device to run the model.
        model (nn.Module): the model to be evaluated.
        dataloader (Dataloader): usually the test loader.
        
    Return: 
        (avg_acc, predictions)
    """
    # set the model to eval mode
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="none").to(device)
    # evaluate
    with torch.no_grad():
        predictions, corrects, n = [], 0., 0.
        test_losses = []
        for _, data in enumerate(tqdm(dataloader)):
            # put data to the device
            data = maybe_dictionarize(data)
            x = data['images'].to(device)
            y = data['labels'].to(device)

            # forward
            logits = get_logits(x, model)
                
            # get losses
            losses = loss_fn(logits, y)
            
            # import ipdb; ipdb.set_trace()
            test_losses.extend(losses.cpu().detach().numpy())
            # get the preds
            preds = logits.argmax(dim=1).to(device)

            # update
            predictions.extend(preds.cpu().numpy())
            corrects += preds.eq(y.view_as(preds)).sum().item()
            n += y.size(0)

        # get the average acc, loss
        avg_acc = corrects / n
        avg_loss = np.mean(test_losses)
        
    return avg_acc, avg_loss, np.array(predictions)

def model_stitching(device: torch.device,
                    save_path: str,
                    model: nn.Module,
                    weights: OrderedDict,
                    featuremaps: OrderedDict,
                    dataloader: torch.utils.data.DataLoader,) -> None:
    """model_stitching

    Args:
        device (torch.device): _description_
        save_path (str): _description_
        model (nn.Module): _description_
        weights (OrderedDict): _description_
        featuremap (OrderedDict): _description_
        dataloader (torch.utils.data.DataLoader): _description_
    """
    logger = get_logger(f"{__name__}.model_stitching")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # initialize the tracker
    tracker = MetricTracker()
    # load the weights
    model.image_encoder.load_state_dict(weights)
    model.to(device)
    model.eval()
    featuremaps.pop("features")
    featuremaps.pop("logits")
    # for each layer, change the intermediate featuremaps
    for layer_name, X in featuremaps.items():
        logger.info(f"predict with Weight_B and Feature_A on {layer_name}")
        # get the module
        module = get_module(model, layer_name)
        # init the curr_idx
        curr_idx = 0
        def hook(module, input, output):
            from open_clip import ResidualAttentionBlock
            if isinstance(module, nn.MultiheadAttention):
                output = output[0]
            if isinstance(module, (nn.MultiheadAttention, nn.Sequential, ResidualAttentionBlock)):
                output = output.permute(1, 0, 2)
            nonlocal curr_idx
            output.data.copy_(X[curr_idx:curr_idx + len(output)])
            curr_idx += output.shape[0]
            if isinstance(module, (nn.MultiheadAttention, nn.Sequential, ResidualAttentionBlock)):
                output =  output.permute(1, 0, 2)

        handle = module.register_forward_hook(hook)
        
        avg_acc, avg_loss, predictions = evaluate(device, model, dataloader)
        if not os.path.exists(os.path.join(save_path, f"layer_{layer_name}")):
            os.makedirs(os.path.join(save_path, f"layer_{layer_name}"))
        np.save(os.path.join(os.path.join(save_path, f"layer_{layer_name}"), "predictions.npy"), predictions)
        handle.remove()
        # track
        tracker.track({
            "layer": layer_name, 
            "avg_acc": avg_acc,
            "avg_loss": avg_loss,
        })
        logger.info(f"layer: {layer_name}, avg_acc: {avg_acc}, avg_loss: {avg_loss}")
    # save the metric
    tracker.save_to_csv(os.path.join(save_path, "model_stitching.csv"))



def add_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="simple verification")
    ## the basic setting of exp
    parser.add_argument('--device', default=0, type=int,
                        help="set the device.")
    parser.add_argument("--seed", default=0, type=int,
                        help="set the seed.")
    parser.add_argument("--save_root", default="../outs/model_stitching/", type=str,
                        help='the path of saving results.')
    parser.add_argument("--model_root", default=None, type=str,
                        help='the path of loading models.')
    parser.add_argument("--data_root", default=None, type=str,
                        help="The root directory for the datasets.",)
    parser.add_argument("--modelA", default=None, type=str,
                        help='set the model A.')
    parser.add_argument("--modelB", default=None, type=str,
                        help='set the model B.')
    parser.add_argument("--dataset", default="ImageNet", type=str,
                        help='the dataset name.')
    parser.add_argument("--sample_num", default=10000, type=int,
                        help="set the sample number.")
    parser.add_argument("--batch_size", default=128, type=int,
                        help="set the batch size."),
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose",
                        help="enable debug info output.")
    args = parser.parse_args()

    if not os.path.exists(args.save_root):
        os.makedirs(args.save_root)

    # set the save_path
    exp_name = "-".join([get_datetime(),
                         f"seed{args.seed}",
                         f"{args.dataset}",
                         f"modelA_{args.modelA}",
                         f"modelB_{args.modelB}",
                         f"sample_num{args.sample_num}",
                         f"bs{args.batch_size}",])
    args.save_path = os.path.join(args.save_root, exp_name)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    return args


def main():
    # get the args.
    args = add_args()
    # set the logger
    set_logger(args.save_path)
    # get the logger
    logger = get_logger(__name__, args.verbose)
    # set the seed
    set_seed(args.seed)
    # set the device
    args.device = set_device(args.device)
    # save the current src
    # save_current_src(save_path = args.save_path)

    # show the args.
    logger.info("#########parameters settings....")
    log_settings(args)

    # prepare the model
    logger.info("#########prepare the model....")
    # find the checkpoint with specification
    image_encoder = torch.load(os.path.join(args.model_root, "zeroshot.pt"))
    # load the classification head
    classification_head = torch.load(os.path.join(args.model_root, f"head_{args.dataset}.pt"))
    # construct the model
    model = ImageClassifier(image_encoder, classification_head)
    logger.info(f"model: {model}")

    # prepare the dataset
    logger.info("#########prepare the dataset....")
    dataset_wrap = get_dataset(
        dataset_name = args.dataset,
        preprocess=model.val_preprocess,
        location=args.data_root,
        batch_size=args.batch_size,
    )
    dataset = dataset_wrap.test_loader.dataset
    indices = torch.randperm(len(dataset))[:args.sample_num]
    subset = Subset(dataset, indices)
    dataloader = DataLoader(subset, batch_size=args.batch_size, shuffle=False, num_workers=16)
    
    

    # get the featuremaps
    logger.info("#########get the featuremaps....")
    weights = dict()
    for key, model_name in [("A", args.modelA), ("B", args.modelB)]: # the model_name should be like "0.5+MNIST_CIFAR10"
        # extract the scaling coefficient and task vectors info
        if model_name is not None and model_name != "":
            scaling_coef, task_vectors_info = decode_model_name(model_name)
            logger.info(f"{key}: scaling_coef: {scaling_coef}, task_vectors_info: {task_vectors_info}")

            image_encoder_tmp = load_image_encoder(model_root=args.model_root,
                                                task_vectors_info=task_vectors_info,
                                                scaling_coef=scaling_coef,)                              
            weights[key] = image_encoder_tmp.state_dict()
        else:
            pretrain_image_encoder = torch.load(os.path.join(args.model_root, "zeroshot.pt"))
            weights[key] = pretrain_image_encoder.state_dict()

    featuremaps = get_featuremaps(device=args.device,
                                  weights=weights["A"],
                                  model=model,
                                  dataloader=dataloader,)
    
    # evalulate the model stitching
    logger.info("#########evalulate the model stitching....")
    model_stitching(device = args.device,
                    save_path = os.path.join(args.save_path, "model_stitching"),
                    model = model,
                    weights = weights["B"],
                    featuremaps = featuremaps,
                    dataloader=dataloader,)

    
if __name__ == "__main__":
    main()