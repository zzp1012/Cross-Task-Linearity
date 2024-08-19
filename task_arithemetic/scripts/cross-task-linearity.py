import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# cwd change to current file's dir
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from collections import OrderedDict

# import internal libs
from src.modeling import ImageClassifier
from src.datasets.registry import get_dataset
from src.utils.avgmeter import MetricTracker
from src.utils.tools import interpolate_weights
from src.utils.dissimilarity import DissimilarityMetric, DissimilarityMetricOverSamples
from src.utils.featuremap import FeatureMap
from src.utils.load import load_image_encoder, decode_model_name
from src.utils import get_datetime, set_logger, get_logger, set_seed, set_device, \
    log_settings, save_current_src

def get_featuremaps(device: torch.device,
                    weightsA: OrderedDict,
                    weightsB: OrderedDict,
                    model: nn.Module,
                    dataloader: DataLoader,
                    alpha: float,
                    beta: float,):
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

    # prepare the weights of model A and B
    weights_alpha = interpolate_weights(weightsA, weightsB, alpha, beta)

    # set the layers
    layers = []
    for name, module in model.named_modules():
        from open_clip import ResidualAttentionBlock
        if isinstance(module, (ResidualAttentionBlock)):
            layers.append(name)
    
    # get the featuremaps of interpolated model
    logger.info(f"get the featuremaps of interpolated model")
    featuremaps = dict()
    for key, weights in [("A", weightsA), ("B", weightsB), ("alpha", weights_alpha)]:
        logger.info(f"key: {key}")
        model.image_encoder.load_state_dict(weights)
        
        # get the featuremaps
        fm = FeatureMap(device, model)
        featuremap = fm.get_featuremaps(dataloader, layer_names=layers)
        del fm
        
        featuremaps[key] = featuremap
    return featuremaps


def eval_linearity(save_path: str,
                   featuremaps: dict,
                   alpha: float, 
                   beta: float,
                   metric: str,) -> None:
    """evaluate the linearity over different layers and alpha and beta.

    Args:
        save_path (str): the path to save the dissimilarity.
        featuremaps (dict): the featuremaps of model A and model B.
        alpha (float): the interpolation coefficient.
        beta (float): the interpolation coefficient.
        metric (str): the metric to calculate the dissimilarity.

    Return:
        None
    """
    logger = get_logger(f"{__name__}.eval_linearity")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # init the dissimilarity metric
    distance_fn = DissimilarityMetric(metric=metric)
    distance_fn_over_samples = DissimilarityMetricOverSamples(metric=metric)
    
    # initialize the MetricTracker
    tracker = MetricTracker()

    # get the layer_names
    layer_names = list(featuremaps["A"].keys())
    # calculate the dissimilarity
    for layer_name in layer_names:
        logger.info(f"layer: {layer_name}")

        # get the featuremap of model A and model B
        featuremapA = featuremaps["A"][layer_name]
        featuremapB = featuremaps["B"][layer_name]
        featuremap_alpha = featuremaps["alpha"][layer_name]
        featuremap_int = alpha * featuremapA + beta * featuremapB

        # calculate the dissimilarity
        if metric == "cosine":
            dist_A_B, coef_A_B = \
                distance_fn(featuremapA.cpu(), featuremapB.cpu(), get_coef=True)
            dist_alpha_A, coef_alpha_A = \
                distance_fn(featuremap_alpha.cpu(), featuremapA.cpu(), get_coef=True)
            dist_alpha_B, coef_alpha_B = \
                distance_fn(featuremap_alpha.cpu(), featuremapB.cpu(), get_coef=True)
            dist_alpha_int, coef_alpha_int = \
                distance_fn(featuremap_alpha.cpu(), featuremap_int.cpu(), get_coef=True)
            
            # over samples
            dist_A_B_over_samples, coef_A_B_over_samples = \
                distance_fn_over_samples(featuremapA.cpu(), featuremapB.cpu(), get_coef=True)
            dist_alpha_A_over_samples, coef_alpha_A_over_samples = \
                distance_fn_over_samples(featuremap_alpha.cpu(), featuremapA.cpu(), get_coef=True)
            dist_alpha_B_over_samples, coef_alpha_B_over_samples = \
                distance_fn_over_samples(featuremap_alpha.cpu(), featuremapB.cpu(), get_coef=True)
            dist_alpha_int_over_samples, coef_alpha_int_over_samples = \
                distance_fn_over_samples(featuremap_alpha.cpu(), featuremap_int.cpu(), get_coef=True)
             
        else:
            dist_A_B = distance_fn(featuremapA.cpu(), featuremapB.cpu())
            dist_alpha_A = distance_fn(featuremap_alpha.cpu(), featuremapA.cpu())
            dist_alpha_B = distance_fn(featuremap_alpha.cpu(), featuremapB.cpu())
            dist_alpha_int = distance_fn(featuremap_alpha.cpu(), featuremap_int.cpu())
            # over samples
            dist_A_B_over_samples = distance_fn_over_samples(featuremapA.cpu(), featuremapB.cpu())
            dist_alpha_A_over_samples = distance_fn_over_samples(featuremap_alpha.cpu(), featuremapA.cpu())
            dist_alpha_B_over_samples = distance_fn_over_samples(featuremap_alpha.cpu(), featuremapB.cpu())
            dist_alpha_int_over_samples = distance_fn_over_samples(featuremap_alpha.cpu(), featuremap_int.cpu())

        layer_save_path = os.path.join(save_path, layer_name)
        if not os.path.exists(layer_save_path):
            os.makedirs(layer_save_path)
        torch.save(dist_A_B_over_samples, os.path.join(layer_save_path, f"dist_A_B.pt"))
        torch.save(dist_alpha_A_over_samples, os.path.join(layer_save_path, f"dist_alpha_A.pt"))
        torch.save(dist_alpha_B_over_samples, os.path.join(layer_save_path, f"dist_alpha_B.pt"))
        torch.save(dist_alpha_int_over_samples, os.path.join(layer_save_path, f"dist_alpha_int.pt"))
        
        if metric == "cosine":
            torch.save(coef_A_B_over_samples, os.path.join(layer_save_path, f"coef_A_B.pt"))
            torch.save(coef_alpha_int_over_samples, os.path.join(layer_save_path, f"coef_alpha_int.pt"))
            torch.save(coef_alpha_A_over_samples, os.path.join(layer_save_path, f"coef_alpha_A.pt"))
            torch.save(coef_alpha_B_over_samples, os.path.join(layer_save_path, f"coef_alpha_B.pt"))
        
        tracker.track({
            "layer": layer_name,
            "dist_A_B": dist_A_B.item(),
            **({"coef_A_B": coef_A_B.item()} if metric == "cosine" else {}),
            "dist_alpha_A": dist_alpha_A.item(),
            **({"coef_alpha_A": coef_alpha_A.item()} if metric == "cosine" else {}),
            "dist_alpha_B": dist_alpha_B.item(),
            **({"coef_alpha_B": coef_alpha_B.item()} if metric == "cosine" else {}),
            "dist_alpha_int": dist_alpha_int.item(),
            **({"coef_alpha_int": coef_alpha_int.item()} if metric == "cosine" else {}),
            # over samples
            "dist_A_B_over_samples_mean": dist_A_B_over_samples.mean().item(),
            "dist_A_B_over_samples_std": dist_A_B_over_samples.std().item(),            
            **{"coef_A_B_over_samples_mean": coef_A_B_over_samples.mean().item() if metric == "cosine" else {}},
            **{"coef_A_B_over_samples_std": coef_A_B_over_samples.std().item() if metric == "cosine" else {}},
            
            "dist_alpha_A_over_samples_mean": dist_alpha_A_over_samples.mean().item(),
            "dist_alpha_A_over_samples_std": dist_alpha_A_over_samples.std().item(),
            **{"coef_alpha_A_over_samples_mean": coef_alpha_A_over_samples.mean().item() if metric == "cosine" else {}},
            **{"coef_alpha_A_over_samples_std": coef_alpha_A_over_samples.std().item() if metric == "cosine" else {}},
            
            "dist_alpha_B_over_samples_mean": dist_alpha_B_over_samples.mean().item(),
            "dist_alpha_B_over_samples_std": dist_alpha_B_over_samples.std().item(),
            **{"coef_alpha_B_over_samples_mean": coef_alpha_B_over_samples.mean().item() if metric == "cosine" else {}},
            **{"coef_alpha_B_over_samples_std": coef_alpha_B_over_samples.std().item() if metric == "cosine" else {}},
            
            
            "dist_alpha_int_over_samples_mean": dist_alpha_int_over_samples.mean().item(),
            "dist_alpha_int_over_samples_std": dist_alpha_int_over_samples.std().item(),
            **{"coef_alpha_int_over_samples_mean": coef_alpha_int_over_samples.mean().item() if metric == "cosine" else {}},
            **{"coef_alpha_int_over_samples_std": coef_alpha_int_over_samples.std().item() if metric == "cosine" else {}},
            
        })

    tracker.save_to_csv(os.path.join(save_path, "sub_linearity.csv"))


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
    parser.add_argument("--sample_num", default=20000, type=int,
                        help="set the sample number.")
    
    parser.add_argument("--metric", default="cosine", type=str,
                        help="the metric to calculate the dissimilarity.")
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
        scaling_coef, task_vectors_info = decode_model_name(model_name)
        logger.info(f"{key}: scaling_coef: {scaling_coef}, task_vectors_info: {task_vectors_info}")

        image_encoder_tmp = load_image_encoder(model_root=args.model_root,
                                               task_vectors_info=task_vectors_info,
                                               scaling_coef=scaling_coef,)                              
        weights[key] = image_encoder_tmp.state_dict()

    featuremaps = get_featuremaps(device=args.device,
                                  weightsA=weights["A"],
                                  weightsB=weights["B"],
                                  model=model,
                                  dataloader=dataloader,
                                  alpha=args.alpha,
                                  beta=args.beta,)
    
    # calculate the dissimilarity
    logger.info("#########calculate the dissimilarity....")
    eval_linearity(save_path=os.path.join(args.save_path, "exp"),
                   featuremaps=featuremaps,
                   alpha=args.alpha,
                   beta=args.beta,
                   metric=args.metric,)
    
    
if __name__ == "__main__":
    main()