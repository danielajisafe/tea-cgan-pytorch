import torch
import wandb
import random
import numpy as np


def seed_everything(seed=0, harsh=False):
    """
    Seeds all important random functions
    Args:
        seed (int, optional): seed value. Defaults to 0.
        harsh (bool, optional): torch backend deterministic. Defaults to False.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    if harsh:
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def init_wandb(cfg: dict) -> None:
    """Initialize project on Weights & Biases
    Args:
        cfg (dict): Configuration dictionary
    """
    for key in cfg:
        cfg[key] = cfg[key].__dict__

    wandb.init(project="tea-cgan", name=cfg["exp_cfg"]["run_name"], 
        notes=cfg["exp_cfg"]["description"], config=cfg)

def dict2device(x, device):
    for key in x:
        if isinstance(x[key], torch.Tensor):
            x[key] = x[key].to(device)

    return x
