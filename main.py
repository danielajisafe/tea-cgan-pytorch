import os
import wandb
import argparse
import numpy as np
from os.path import join, splitext

from trainer import TEACGANTrainer
from config.config import cfg_parser
from utils import seed_everything, init_wandb


if __name__=="__main__":
    seed_everything(seed=1234, harsh=False)
    parser = argparse.ArgumentParser(
        description="TEA-CGAN for CelebA", allow_abbrev=False
    )
    parser.add_argument(
        "-v",
        "--version",
        type=str,
        default="tea_cgan.yml",
        help="name of the config file to use"
        )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU ID"
        )
    parser.add_argument(
        "-w", "--wandb", action="store_true", help="Log to wandb"
    )
    parser.add_argument(
        "-s", "--sweep", action="store_true", help="Use wandb sweep or not"
    )
    (args, unknown_args) = parser.parse_known_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    cfg = cfg_parser(join("config", args.version), args.sweep)
    cfg["exp_cfg"].version = splitext(args.version)[0]
    cfg["exp_cfg"].run_name = cfg["exp_cfg"].version
    cfg["exp_cfg"].wandb = args.wandb

    for unknown_arg in unknown_args:
        t = unknown_arg.split("--")[1]
        x = t.split("=")
        exec("{} = {}".format(x[0], type(eval(x[0]))(x[1])))
    
    if args.wandb:
        init_wandb(cfg.copy())
    pipeline = TEACGANTrainer(**cfg)
    pipeline.train()
