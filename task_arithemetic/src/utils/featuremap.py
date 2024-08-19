import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from functools import reduce
from collections import defaultdict
from tqdm import tqdm

# import internal libs
from src.utils import get_logger, get_logits
from src.datasets.common import maybe_dictionarize

class FeatureMap:
    """class used to extract feature map from a given model
    """
    def __init__(self,
                 device: torch.device,
                 model: nn.Module,) -> None:
        """initialize the feature map extractor

        Args:
            device (torch.device): device to run the model
            model (nn.Module): model to extract feature map
        """
        self.__device = device
        self.__model = model.to(device)
        self.__model_name = model.__class__.__name__
        
        # make sure the model is in eval mode
        self.__model.eval()
        # initialize the hooks
        self.__hooks = {}
        self.__featuremaps = defaultdict(list)
        
    def __get_module(self, module_name: str) -> nn.Module:
        """get the module from the model

        Args:
            module_name (str): name of the module

        Returns:
            nn.Module: the module
        """
        return reduce(getattr, module_name.split('.'), self.__model)

    def __get_conv_fc_layer_names(self) -> list:
        """get the names of the conv layers and the fc layers

        Returns:
            list: names of the conv layers
        """
        layer_names = []
        for name, module in self.__model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                layer_names.append(name)
        return layer_names

    def __get_conv_fc_modules(self, layer_names: list = None) -> dict:
        """get the conv and fc modules

        Args:
            layer_names (list, optional): names of the layers. Defaults to None.

        Returns:
            dict: conv modules
        """
        if layer_names is None:
            layer_names = self.__get_conv_fc_layer_names()
        modules = dict()
        for name in layer_names:
            modules[name] = self.__get_module(name)
        return modules
    
    def __register_single_hook(self, 
                               layer_name: str,
                               module: nn.Module,
                               get_input: bool) -> None:
        """register the hook for a single module

        Args:
            layer_name (str): name of the layer
            module (nn.Module): module to register the hook
            get_input (bool): whether to get the input of the model
        """
        def forward_hook(module, input, output):
            if get_input:
                assert len(input) == 1, "only support single input"
                self.__featuremaps[layer_name].append(input[0].detach().cpu())
            else:
                if isinstance(output, tuple):
                    self.__featuremaps[layer_name].append(output[0].detach().cpu())
                else:
                    self.__featuremaps[layer_name].append(output.detach().cpu())
        self.__hooks[layer_name] = module.register_forward_hook(forward_hook)

    def __register_hooks(self, 
                         layer_names: list = None,
                         get_input: bool = False) -> None:
        """register the hooks

        Args:
            layer_names (list, optional): names of the layers to register the hooks. Defaults to None.
            get_input (bool, optional): whether to get the input of the model. Defaults to False.
        """
        conv_modules = self.__get_conv_fc_modules(layer_names)
        for name, module in conv_modules.items():
            self.__register_single_hook(name, module, get_input)

    def __remove_hooks(self) -> None:
        """remove the hooks
        """
        for hook in self.__hooks.values():
            hook.remove()
        self.__hooks = {}

    def __remove_featuremaps(self) -> None:
        """remove the feature maps
        """
        self.__featuremaps = defaultdict(list)

    def __convert_featuremaps_to_tensor(self) -> dict:
        """convert the feature maps to tensor

        Returns:
            dict: feature maps
        """
        featuremaps = dict()
        for key, value in self.__featuremaps.items():
            if "visual.transformer.resblocks" in key and len(value[0].shape) == 3:
                value = [value[i].permute(1, 0, 2) for i in range(len(value))]
            featuremaps[key] = torch.cat(value, dim=0)
        return featuremaps

    def get_featuremaps(self,
                        dataloader: DataLoader,
                        layer_names: list = None,
                        get_input: bool = False) -> tuple:
        """get the feature maps from the conv layers
            tips: use the hook to get the feature maps

        Args:
            dataloader (DataLoader): data loader to load the data
            layer_names (list): names of the layers to extract feature map. Defaults to None.
            get_input (bool): whether to get the input of the model. Defaults to False.

        Returns:
            (feature maps, preds): feature maps and preds
        """
        logger = get_logger(
            f"{__name__}.{self.__class__.__name__}.get_featuremaps"
        )

        # clear the feature maps
        self.__remove_featuremaps()
        
        # register the hooks
        self.__register_hooks(layer_names, get_input)

        # extract the feature maps
        with torch.no_grad():
            features_lst, logits_lst = [], []
            for _, data in enumerate(tqdm(dataloader)):
                # move the data to the device
                data = maybe_dictionarize(data)
                x = data['images'].to(self.__device)
                # forward
                features = self.__model.image_encoder(x)
                logits = self.__model.classification_head(features)
                # get the logits
                logits_lst.append(logits.detach().cpu())
                # get the features
                features_lst.append(features.detach().cpu())

        # concat the features & logits
        features = torch.cat(features_lst, dim=0)
        logits = torch.cat(logits_lst, dim=0)

        # convert the feature maps to tensor
        featuremaps = self.__convert_featuremaps_to_tensor()
        featuremaps = {**featuremaps, "features": features, "logits": logits}
        # clear the feature_maps
        self.__remove_featuremaps()
        # remove the hooks
        self.__remove_hooks()
        
        logger.info(f"feature maps: {featuremaps.keys()}")       
        return featuremaps