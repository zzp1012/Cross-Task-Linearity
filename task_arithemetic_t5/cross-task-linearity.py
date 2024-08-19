import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import argparse
import torch
import torch.nn as nn
import copy
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from utils import get_datetime, get_logger, set_logger, set_device, set_seed, \
    log_settings, interpolate_weights, save_current_src
from data_utils import load_hf_dataset, get_evaluate_fn
from avgmeter import MetricTracker
from dissimilarity import DissimilarityMetricOverSamples, DissimilarityMetric
from task_vectors import TaskVector

def eval_linearity(save_path: str,
                   featuremaps: dict,
                   alpha: float, 
                   beta: float,
                   metric: str) -> None:
    """evaluate the linearity over different layers and alpha and beta.

    Args:
        save_path (str): the path to save the dissimilarity.
        featuremaps (dict): the featuremaps of model A and model B.
        alpha (float): the interpolation coefficient.
        beta (float): the interpolation coefficient.

    Return:
        None
    """
    logger = get_logger(f"{__name__}.eval_linearity")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
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
    

def get_featuremaps(device: torch.device,
                    modelA,
                    modelB,
                    base_model: nn.Module,
                    tokenizer,
                    dataset,
                    dataset_name: str,
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
    weightA, weightB = modelA.state_dict(), modelB.state_dict()
    weight_alpha = interpolate_weights(weightA, weightB, alpha, beta)
    # base_model.load_state_dict(weight_alpha)
    model_alpha = copy.deepcopy(base_model)
    model_alpha.load_state_dict(weight_alpha)
    
    # set the layers
    model_name = base_model.__class__.__name__
    logger.info(f"model_name: {model_name}")
    if model_name.startswith("T5"):
        from transformers.models.t5.modeling_t5 import T5Block
        layers = [name for name, module in base_model.named_modules() if isinstance(module, (T5Block))]
        logger.info(f"feature layers: {layers}")
    else:
        raise NotImplementedError(f"the model - {model_name} is not implemented")

    # get the featuremaps of interpolated model
    logger.info(f"get the featuremaps of interpolated model")
    featuremaps = dict()
    for key, model in [("A", modelA), ("B", modelB), ("alpha", model_alpha)]:
        logger.info(f"key: {key}")
        
        eval_fn = get_evaluate_fn(dataset_name)  
        model.eval()
        model.to(device)      
        encoder_featuremap = eval_fn(model=model,
                         tokenizer=tokenizer,
                         dataset=dataset,
                         device=device)
        
        featuremaps[key] = encoder_featuremap
    return featuremaps

def add_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="simple verification")
    ## the basic setting of exp

    parser.add_argument('--device', default=0, type=int,
                        help="set the device.")
    parser.add_argument("--seed", default=0, type=int,
                        help="set the seed.")
    parser.add_argument("--save_root", default="./outs/llfc/", type=str,
                        help='the path of saving results.')
    parser.add_argument("--base_model", default="./t5-base", type=str,
                        help='the base model name.')
    parser.add_argument("--modelA_path", default=None, type=str,
                        help='the path of pretrained model A.')
    parser.add_argument("--modelA_coef", default=1.0, type=float,
                        help='the coefficient of model A.')
    parser.add_argument("--modelB_path", default=None, type=str,
                        help='the path of pretrained model B.')
    parser.add_argument("--modelB_coef", default=1.0, type=float,
                        help='the coefficient of model B.')
    parser.add_argument("--dataset", default="imdb", type=str, choices=["imdb", "race", "qasc", "multi_news", "squad", "common_gen"],
                        help='the dataset name.')
    parser.add_argument("--model", default="T5", type=str,
                        help='the model name.')
    parser.add_argument("--alpha", default=0.5, type=float,
                        help="set the alpha to interpolate the weights.")
    parser.add_argument("--beta", default=0.5, type=float,
                        help="set the beta to interpolate the weights.")
    parser.add_argument("--sample_num", default=5000, type=int,
                        help="set the sample number.")
    parser.add_argument("--batch-size", default=128, type=int,
                        help="set the batch size.")
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose",
                        help="enable debug info output.")
    args = parser.parse_args()

    if not os.path.exists(args.save_root):
        os.makedirs(args.save_root)

    # set the save_path
    modelA_name = args.modelA_path.split("/")[-1]
    modelB_name = args.modelB_path.split("/")[-1]
    exp_name = "-".join([get_datetime(),
                         f"seed{args.seed}",
                         f"{args.dataset}",
                         f"{args.model}",
                         f"modelA_{modelA_name}",
                         f"modelB_{modelB_name}",
                         f"modelA_coef{args.modelA_coef}",
                         f"modelB_coef{args.modelB_coef}",
                         f"alpha{args.alpha}",
                         f"beta{args.beta}",
                         f"sample_num{args.sample_num}",
                         f"bs{args.batch_size}",])
    args.save_path = os.path.join(args.save_root, exp_name)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    return args


def main():
    args = add_args()
    
    set_logger(args.save_path)
    logger = get_logger(__name__, args.verbose)
    set_seed(args.seed)
    args.device = set_device(args.device)
    logger.info("#########parameters settings....")
    log_settings(args)
    
    # prepare datasets
    assert args.model == "T5", "only support T5 now."
    
    # prepare dataset
    logger.info("#########prepare dataset....")
    dataset = load_hf_dataset(args.dataset)
    indices = torch.randperm(len(dataset))[:args.sample_num].numpy().tolist()
    subset = Subset(dataset, indices)
    # dataloader = DataLoader(subset, batch_size=args.batch_size, shuffle=False)
    
    
    # prepare model
    logger.info("#########prepare model....")
    base_model, base_tokenizer = AutoModelForSeq2SeqLM.from_pretrained(args.base_model), \
        AutoTokenizer.from_pretrained(args.base_model)
    modelA_origin, tokenizerA = AutoModelForSeq2SeqLM.from_pretrained(args.modelA_path), \
        AutoTokenizer.from_pretrained(args.modelA_path)
    modelB_origin, tokenizerB = AutoModelForSeq2SeqLM.from_pretrained(args.modelB_path), \
        AutoTokenizer.from_pretrained(args.modelB_path)
    
    
    task_vector_A = TaskVector(pretrained_checkpoint=base_model, finetuned_checkpoint=modelA_origin)
    modelA = task_vector_A.apply_to(copy.deepcopy(base_model), scaling_coef=args.modelA_coef)
    task_vector_B = TaskVector(pretrained_checkpoint=base_model, finetuned_checkpoint=modelB_origin)
    modelB = task_vector_B.apply_to(copy.deepcopy(base_model), scaling_coef=args.modelB_coef)
    
    
    # modelA = modelA_origin
    # modelB = modelB_origin
    
    logger.info("#########get the featuremaps....")
    featuremaps = get_featuremaps(device=args.device,
                                  modelA=modelA,
                                  modelB=modelB,
                                  base_model = base_model,
                                  tokenizer=base_tokenizer,
                                  dataset=subset,
                                  dataset_name=args.dataset,
                                  alpha=args.alpha,
                                  beta=args.beta,)
    
    eval_linearity(save_path=os.path.join(args.save_path, "exp"),
              featuremaps=featuremaps,
              alpha=args.alpha,
              beta=args.beta,
              metric="cosine")

if __name__ == "__main__":
    main()
        