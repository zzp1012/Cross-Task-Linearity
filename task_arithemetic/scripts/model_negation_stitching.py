import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# cwd change to current file's dir
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from collections import OrderedDict, defaultdict

# import internal libs
from src.modeling import ImageClassifier
from src.datasets.registry import get_dataset
from src.utils import get_datetime, set_logger, get_logger, set_seed, set_device, \
    log_settings, save_current_src
from src.utils.avgmeter import MetricTracker
from src.utils.tools import get_module, evaluate
from src.utils.featuremap import FeatureMap
from src.utils.load import load_image_encoder

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


def model_stitch(device: torch.device,
                 model: nn.Module,
                 weights: dict,
                 featuremaps: dict,
                 dataloader: DataLoader,) -> defaultdict:
    """ensemble the featuremaps and stitch them
    Args:
        device (torch.device): the device to run the model.
        model (nn.Module): the model to extract featuremaps.
        weights (dict): the weights of the image_encoder.
        featuremaps (dict): the featuremaps of the model.
        dataloader (torch.utils.data.DataLoader): the dataloader.
    
    Return:
        None
    """
    logger = get_logger(f"{__name__}.model_stitch")
    # init the metric tracker
    tracker = MetricTracker()

    # set the model to eval mode
    model.image_encoder.load_state_dict(weights)
    model.to(device)
    model.eval()

    # for each layer, change the intermediate featuremap
    for layer_name in featuremaps.keys():
        if layer_name == "features" or layer_name == "logits":
            continue
        logger.info(f"current layer {layer_name}")

        # get the feature map in current layer
        H = featuremaps[layer_name]

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
            output.data.copy_(H[curr_idx:curr_idx + len(output)])
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
    return tracker.get_metrics()


def add_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="simple verification")
    ## the basic setting of exp
    parser.add_argument('--device', default=0, type=int,
                        help="set the device.")
    parser.add_argument("--seed", default=0, type=int,
                        help="set the seed.")
    parser.add_argument("--save_root", default="../outs/negation_stitching/", type=str,
                        help='the path of saving results.')
    parser.add_argument("--model_root", default=None, type=str,
                        help='the path of loading models.')
    parser.add_argument("--data_root", default=None, type=str,
                        help="The root directory for the datasets.",)
    parser.add_argument("--task", default=None, type=str,
                        help='set the task name.')
    parser.add_argument("--dataset", default="ImageNet", type=str,
                        help='the dataset name.')
    parser.add_argument("--sample_num", default=5000, type=int, 
                        help="the sample num to test")
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
                         f"{args.task}",
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
    save_current_src(save_path = args.save_path)

    # show the args.
    logger.info("#########parameters settings....")
    log_settings(args)

    # prepare the model
    logger.info("#########prepare the model....")
    # find the checkpoint with specification
    zeroshot_image_encoder = torch.load(os.path.join(args.model_root, "zeroshot.pt"))
    # load the classification head
    classification_head = torch.load(os.path.join(args.model_root, f"head_{args.dataset}.pt"))
    # construct the model
    model = ImageClassifier(zeroshot_image_encoder, classification_head)
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

    # get the zeroshot featuremaps
    logger.info("#########prepare the zeroshot featuremaps")
    zeroshot_weights = zeroshot_image_encoder.state_dict()
    zeroshot_featuremaps = get_featuremaps(device=args.device,
                                           weights=zeroshot_weights,
                                           model=model,
                                           dataloader=dataloader,)

    # iterate each lambda and layers
    logger.info("#########negation and stitch...")
    # init the tracker
    tracker = MetricTracker()

    lambdas = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, \
        0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
    for lambda_ in lambdas:
        logger.info(f"Evaluating the lambda {lambda_}")

        # first original minus featuremaps
        minus_image_encoder = load_image_encoder(model_root=args.model_root,
                                                 task_vectors_info={args.task: "minus"},
                                                 scaling_coef=lambda_,) 
        minus_origin_featuremaps = get_featuremaps(device=args.device,
                                                weights=minus_image_encoder.state_dict(),
                                                model=model,
                                                dataloader=dataloader,)
        minus_origin_metrics = model_stitch(device=args.device,
                                            model=model,
                                            weights=zeroshot_weights,
                                            featuremaps=minus_origin_featuremaps,
                                            dataloader=dataloader,)

        # then use the approximate minus featuremaps
        add_image_encoder = load_image_encoder(model_root=args.model_root,
                                               task_vectors_info={args.task: "add"},
                                               scaling_coef=lambda_,)
        add_featuremaps = get_featuremaps(device=args.device,
                                          weights=add_image_encoder.state_dict(),
                                          model=model,
                                          dataloader=dataloader,)
        # creae the approximate minus featuremaps
        minus_approx_featuremaps = {}
        for layer_name in minus_origin_featuremaps.keys():
            minus_approx_featuremaps[layer_name] = \
                2*zeroshot_featuremaps[layer_name] - add_featuremaps[layer_name]
        minus_approx_metrics = model_stitch(device=args.device,
                                            model=model,
                                            weights=zeroshot_weights,
                                            featuremaps=minus_approx_featuremaps,
                                            dataloader=dataloader,)

        for idx, layer in enumerate(minus_origin_metrics["layer"]):
            tracker.track({
                "layer": layer,
                "lambda": lambda_,
                "origin_acc": minus_origin_metrics["avg_acc"][idx],
                "origin_loss": minus_origin_metrics["avg_loss"][idx],
                "approx_acc": minus_approx_metrics["avg_acc"][idx],
                "approx_loss": minus_approx_metrics["avg_loss"][idx],
            })

    tracker.save_to_csv(os.path.join(args.save_path, "negation_stitch.csv"))


if __name__ == "__main__":
    main()