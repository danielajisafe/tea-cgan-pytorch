import wandb
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F

from utils import dict2device
from model.tea_cgan import TEACGAN
from data.dataloader import get_loader


class TEACGANTrainer(object):
    def __init__(self, data_cfg, model_cfg, exp_cfg):
        self.data_cfg = data_cfg
        self.model_cfg = model_cfg
        self.exp_cfg = exp_cfg

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.dataloader = get_loader(data_cfg)

        self.model = TEACGAN(self.model_cfg).to(self.device)
        if exp_cfg.wandb:
            wandb.watch(self.model)

        self._setup_optimizer()

    def _setup_optimizer(self):
        gen_params = list(self.model.image_encoder.parameters()) + \
                           list(self.model.image_decoder.parameters()) + \
                           list(self.model.projection_layer.parameters()) + \
                           list(self.model.text_encoder.parameters()) + \
                           list(self.model.res_block1.parameters()) + \
                           list(self.model.res_block2.parameters())

        # disc_params = list(self.model.discriminator.parameters())

        gen_opt_cfg = self.model_cfg.optimizer['generator']
        self.gen_opt = eval("torch.optim.{}(gen_params, **{})".format([*gen_opt_cfg.keys()][0], [*gen_opt_cfg.values()][0]))

    def _backprop(self, loss, opt):
        opt.zero_grad()
        loss.backward()
        opt.step()

    def _get_iterator(self):
        data_iter = iter(self.dataloader)
        tqdm_iter = tqdm(range(len(self.dataloader)))

        return data_iter, tqdm_iter

    def _epoch(self, epochID:int, mode="train"):
        train = mode == 'train'
        if train:
            self.model.train()
        else:
            self.model.eval()

        data_iter, tqdm_iter = self._get_iterator()

        for i in tqdm_iter:
            batch = next(data_iter)
            batch = dict2device(batch, self.device)

            image_hat = self.model(batch['image'], batch['caption'])

            recon_loss = F.l1_loss(image_hat, batch['image'])
            gen_loss = recon_loss

            if train:
                self._backprop(gen_loss, self.gen_opt)

            tqdm_iter.set_description("{} | Epoch {} | {} | loss:{:.4f}".format(self.exp_cfg.version, epochID, mode, gen_loss.item()))

    def train(self):
        for epochID in range(self.model_cfg.epochs):
            for mode in self.model_cfg.modes:
                self._epoch(epochID, mode)
