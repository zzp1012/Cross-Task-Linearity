import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# cwd change to current file's dir
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import argparse
import numpy as np
import torch

# import internal libs
from src.modeling import ImageClassifier
from src.datasets.registry import get_dataset
from src.utils.load import load_image_encoder, decode_model_name
from src.utils.tools import evaluate
from src.utils import get_datetime, set_logger, get_logger, set_seed, set_device, \
    log_settings, save_current_src

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
    parser.add_argument("--model", default=None, type=str,
                        help="specify the model name",)
    parser.add_argument("--dataset", default="ImageNet", type=str,
                        help='the dataset name.')
    parser.add_argument("--batch_size", default=256, type=int,
                        help="set the batch size."),
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose",
                        help="enable debug info output.")
    args = parser.parse_args()

    if not os.path.exists(args.save_root):
        os.makedirs(args.save_root)

    # set the save_path
    exp_name = "-".join([get_datetime(),
                         f"seed{args.seed}",
                         f"{args.model}",
                         f"{args.dataset}",
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
    # load the image encoder
    image_encoder = torch.load(os.path.join(args.model_root, "zeroshot.pt"))
    # load the classification head
    classification_head = torch.load(os.path.join(args.model_root, f"head_{args.dataset}.pt"))
    # construct the model
    model = ImageClassifier(image_encoder, classification_head)
    logger.info(f"model: {model}")

    # prepare the dataset
    logger.info("#########prepare the dataset....")
    dataset = get_dataset(
        dataset_name = args.dataset,
        preprocess=model.val_preprocess,
        location=args.data_root,
        batch_size=args.batch_size,
    )
    dataloader = dataset.test_loader
    
    # eval the model over the dataset
    logger.info("#########eval the model over the dataset....")
    scaling_coef, task_vectors_info = decode_model_name(args.model)

    # get the image encoder to be evaluated
    model.image_encoder = load_image_encoder(
        model_root = args.model_root,
        task_vectors_info = task_vectors_info,
        scaling_coef = scaling_coef,
    )

    avg_acc, avg_loss, predictions = evaluate(args.device, model, dataloader)
    logger.info(f"avg_acc: {avg_acc}")
    logger.info(f"avg_loss: {avg_loss}")

    # save the predictions
    np.save(os.path.join(args.save_path, "predictions.npy"), predictions)


if __name__ == "__main__":
    main()