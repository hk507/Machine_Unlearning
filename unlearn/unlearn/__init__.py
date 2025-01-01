from .FT import FT, FT_l1
from .GA import GA, GA_l1
from .impl import load_unlearn_checkpoint, save_unlearn_checkpoint
from .retrain import retrain


def raw(data_loaders, model, criterion, args):
    pass


def get_unlearn_method(name):
    """method usage:

    function(data_loaders, model, criterion, args)"""
    if name == "raw":
        return raw
    elif name == "GA":
        return GA
    elif name == "FT":
        return FT
    elif name == "FT_l1":
        return FT_l1
    elif name == "retrain":
        return retrain
    elif name == "GA_l1":
        return GA_l1
    else:
        raise NotImplementedError(f"Unlearn method {name} not implemented!")
