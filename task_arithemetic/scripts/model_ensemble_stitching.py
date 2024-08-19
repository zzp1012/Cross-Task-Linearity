import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# cwd change to current file's dir
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from collections import OrderedDict

# import internal libs
from src.modeling import ImageClassifier
from src.datasets.registry import get_dataset
from src.utils import get_datetime, set_logger, get_logger, set_seed, set_device, \
    log_settings, save_current_src
from src.utils.avgmeter import MetricTracker
from src.utils.tools import interpolate_weights, get_module, evaluate
from src.utils.featuremap import FeatureMap
from src.utils.load import load_image_encoder, decode_model_name

def get_featuremaps(device: torch.device,
                    weights: OrderedDict,
                    model: nn.Module,
                    dataloader: DataLoader,):
    """get the featuremaps of model A and model B

    Args:
        device (torch.device): the device to run the model.
        weights (OrderedDict): the weights of the image_encoder.
        model (nn.Module): the model to extract featuremaps.
        dataloader (torch.utils.data.DataLoader): the dataloader.

    Return:
        None
    """
    logger = get_logger(f"{__name__}.get_featuremaps")

    # set the layers
    layers = []
    for name, module in model.named_modules():
        from open_clip import ResidualAttentionBlock
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


def ensemble_stitch(device: torch.device,
                    save_path: str,
                    model: nn.Module,
                    weights: dict,
                    featuremaps: dict,
                    dataloader: DataLoader,
                    alpha: float = 0.5,
                    beta: float = 0.5,
                    feat: str = "int") -> None:
    """ensemble the featuremaps and stitch them

    Args:
        device (torch.device): the device to run the model.
        save_path (str): the path to save the results.
        model (nn.Module): the model to extract featuremaps.
        weights (dict): the weights of the image_encoder.
        featuremaps (dict): the featuremaps of the model.
        dataloader (torch.utils.data.DataLoader): the dataloader.
        alpha (float, optional): the alpha to interpolate the weights. Defaults to 0.5.
        beta (float, optional): the beta to interpolate the weights. Defaults to 0.5.
        feat (str, optional): the featuremap to interpolate. Defaults to "int".
    
    Return:
        None
    """
    logger = get_logger(f"{__name__}.ensemble_stitch")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # init the metric tracker
    tracker = MetricTracker()

    # first interpolate the weights
    assert list(weights.keys()) == list(featuremaps.keys()) == ["A", "B"], \
        f"""the keys of weights and featuremaps should be the same, 
            but got {weights.keys()} and {featuremaps.keys()}
        """
        
    # use the pretrain model to compute
    # model.image_encoder.load_state_dict(
        # interpolate_weights(weights["A"], weights["B"], alpha=alpha, beta=beta)
    # )
    
    # set the model to eval mode
    model.to(device)
    model.eval()

    # for each layer, change the intermediate featuremap
    for layer_name in featuremaps["A"].keys():
        if layer_name == "features" or layer_name == "logits":
            continue
        logger.info(f"current layer {layer_name}")

        # get the H_A and H_B
        H_A = featuremaps["A"][layer_name]
        H_B = featuremaps["B"][layer_name]
        # get the interpolated H
        if feat == "int":
            H_int = alpha * H_A + beta * H_B
        elif feat == "A":
            H_int = H_A
        elif feat == "B":
            H_int = H_B
        else:
            raise NotImplementedError(f"feat: {feat} is not implemented.")

        # get the curr module
        module = get_module(model, layer_name)

        # hook part
        curr_idx = 0
        def hook(module, input, output):
            # preprocess
            from open_clip import ResidualAttentionBlock
            if isinstance(module, (ResidualAttentionBlock)):
                output = output.permute(1, 0, 2)
            
            # processing
            nonlocal curr_idx
            output.data.copy_(H_int[curr_idx:curr_idx + len(output)])
            curr_idx += output.shape[0]
            
            # transform back
            if isinstance(module, (ResidualAttentionBlock)):
                output =  output.permute(1, 0, 2)

        handle = module.register_forward_hook(hook)

        # evaluate
        avg_acc, avg_loss, _ = evaluate(device, model, dataloader)
        logger.info(f"avg_acc: {avg_acc}, avg_loss: {avg_loss}")

        # remove handle
        handle.remove()

        # track
        tracker.track({
            "layer": layer_name, 
            "avg_acc": avg_acc,
            "avg_loss": avg_loss,
        })

    # save the results
    tracker.save_to_csv(os.path.join(save_path, "ensemble_stitch.csv"))


def add_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="simple verification")
    ## the basic setting of exp
    parser.add_argument('--device', default=0, type=int,
                        help="set the device.")
    parser.add_argument("--seed", default=0, type=int,
                        help="set the seed.")
    parser.add_argument("--save_root", default="../outs/tmp/", type=str,
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
    parser.add_argument("--batch_size", default=256, type=int,
                        help="set the batch size."),
    parser.add_argument("--alpha", default=0.5, type=float,
                        help="set the alpha to interpolate the weights.")
    parser.add_argument("--beta", default=0.5, type=float,
                        help="set the beta to interpolate the weights.")
    parser.add_argument("--sample_num", default=5000, type=int,
                        help="the sample num to calculate the dissimilarity.")
    parser.add_argument("--metric", default="cosine", type=str,
                        help="the metric to calculate the dissimilarity.")
    parser.add_argument("--feat", default="int", type=str,
                        help="the featuremap to interpolate.")
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
                         f"alpha{args.alpha}",
                         f"beta{args.beta}",
                         f"bs{args.batch_size}",
                         f"sample_num{args.sample_num}",
                         f"{args.metric}",])
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
    save_current_src(save_path = args.save_path)

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
    weights, featuremaps = {}, {}
    for key, model_name in [("A", args.modelA), ("B", args.modelB)]: # the model_name should be like "0.5+MNIST_CIFAR10"
        # extract the scaling coefficient and task vectors info
        scaling_coef, task_vectors_info = decode_model_name(model_name)
        logger.info(f"{key}: scaling_coef: {scaling_coef}, task_vectors_info: {task_vectors_info}")

        # load the image_encoder
        image_encoder_tmp = load_image_encoder(model_root=args.model_root,
                                               task_vectors_info=task_vectors_info,
                                               scaling_coef=scaling_coef,)                              
        weights[key] = image_encoder_tmp.state_dict()

        # get the featuremaps
        featuremaps[key] = get_featuremaps(device=args.device,
                                           weights=weights[key],
                                           model=model,
                                           dataloader=dataloader,)
    
    # ensemble and stitch
    logger.info("#########ensemble and stitch...")
    ensemble_stitch(device=args.device,
                    save_path=os.path.join(args.save_path, "ensemble_stitch_exp"),
                    model=model,
                    weights=weights,
                    featuremaps=featuremaps,
                    dataloader=dataloader,
                    alpha=args.alpha,
                    beta=args.beta,
                    feat=args.feat,)
    
    
if __name__ == "__main__":
    main()