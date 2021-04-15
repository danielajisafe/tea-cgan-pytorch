import torch
import wandb
import random
import numpy as np
import torchvision.utils as vutils


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

def aggregate(losses):
    for key in losses:
        losses[key] = np.mean(losses[key])

    return losses

def wandb_log(epochID, losses, mode):
    logs = {}
    losses = aggregate(losses)
    for key in losses.keys():
        logs.update({"{}/{}".format(mode, key): losses[key]})

    wandb.log(logs, step=epochID)

def log_images(epochID:int, mode:str, x, x_hat, name='reconstruction'):
    x = (x.detach().cpu().numpy() * 0.5) + 0.5
    x_hat = (x_hat.detach().cpu().numpy() * 0.5) + 0.5
    grid = np.zeros((12, x.shape[1], x.shape[2], x.shape[3]))

    for i in range(grid.shape[0]//2):
        grid[i] = x[i]
        grid[i+6] = x_hat[i]

    grid = vutils.make_grid(torch.from_numpy(grid), nrow=6, normalize=True, scale_each=True)

    wandb.log({"{}_{}".format(mode, name): wandb.Image(grid)}, step=epochID)
