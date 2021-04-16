import wandb
import numpy as np
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.nn.functional as F

from model.tea_cgan import TEACGAN
from data.dataloader import get_loader
from utils import dict2device, log_images, wandb_log, aggregate, save_model


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

        self.D = self.model.discriminator_forward

        self.vis_idx = np.random.randint(0, self.data_cfg.batch_size, 6)
        self.gen_loss = np.inf

    def _setup_optimizer(self):
        gen_params = list(self.model.image_encoder.parameters()) + \
                           list(self.model.image_decoder.parameters()) + \
                           list(self.model.projection_layer.parameters()) + \
                           list(self.model.text_encoder.parameters()) + \
                           list(self.model.res_block1.parameters()) + \
                           list(self.model.res_block2.parameters())

        gen_opt_cfg = self.model_cfg.optimizer['generator']
        self.gen_opt = eval("torch.optim.{}(gen_params, **{})".format([*gen_opt_cfg.keys()][0], [*gen_opt_cfg.values()][0]))

        disc_params = list(self.model.discriminator.parameters()) + \
                      list(self.model.joint_conv.parameters()) + \
                      list(self.model.logit_conv.parameters()) + \
                      list(self.model.uncond_logit_conv.parameters())

        disc_opt_cfg = self.model_cfg.optimizer['discriminator']
        self.disc_opt = eval("torch.optim.{}(disc_params, **{})".format([*disc_opt_cfg.keys()][0], [*disc_opt_cfg.values()][0]))

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

        metrics = defaultdict(list)
        N = self.data_cfg.batch_size
        ones = torch.ones((N)).to(self.device)
        zeros = torch.zeros((N)).to(self.device)

        for i in tqdm_iter:
            batch = next(data_iter)
            batch = dict2device(batch, self.device)

            g_I_T, T = self.model(batch['image'], batch['caption'])
            g_I_T_hat, T_hat = self.model(batch['image'], batch['mismatch'])

            gen_loss = 0
            gen_loss += F.mse_loss(self.D(g_I_T), ones)
            gen_loss += F.mse_loss(self.D(g_I_T_hat, T_hat)[0], ones)
            gen_loss += self.model_cfg.gamma2 * F.l1_loss(g_I_T, batch['image']) #recon_loss

            disc_loss = 0
            disc_loss += F.mse_loss(self.D(batch['image']), ones)
            disc_loss += F.mse_loss(self.D(g_I_T.detach()), zeros)
            disc_loss += self.model_cfg.gamma1 * F.mse_loss(self.D(batch['image'], T.detach())[0], ones)
            disc_loss += self.model_cfg.gamma1 * F.mse_loss(self.D(g_I_T_hat.detach(), T_hat.detach())[0], zeros)

            # gen_loss = 0
            # gen_loss += torch.log(self.D(batch['image'])).mean()
            # gen_loss += self.model_cfg.gamma1 * torch.log(self.D(g_I_T_hat, T_hat)[0]).mean()
            # gen_loss += self.model_cfg.gamma2 * F.mse_loss(g_I_T, batch['image']) #recon_loss

            # disc_obj = 0
            # disc_obj += torch.log(self.D(batch['image'])).mean()
            # disc_obj += torch.log(1 - self.D(g_I_T.detach())).mean()
            # disc_obj += self.model_cfg.gamma1 * torch.log(self.D(batch['image'], T.detach())[0]).mean()
            # disc_obj += self.model_cfg.gamma1 * torch.log(1 - self.D(g_I_T_hat.detach(), T_hat.detach())[0]).mean()

            if train:
                self._backprop(gen_loss, self.gen_opt)
                self._backprop(disc_loss, self.disc_opt)

            tqdm_iter.set_description("{} | Epoch {} | {} | loss:{:.4f}".format(self.exp_cfg.version, epochID, mode, gen_loss.item()))

            if self.exp_cfg.wandb and i==0:
                log_images(epochID, mode, batch['image'][self.vis_idx], g_I_T[self.vis_idx], name='reconstruction')
                log_images(epochID, mode, batch['image'][self.vis_idx], g_I_T_hat[self.vis_idx], name='mismatch')

            metrics['gen_loss'].append(gen_loss.item())
            metrics['disc_loss'].append(disc_loss.item())

        metrics = aggregate(metrics)
        if self.exp_cfg.wandb:
            wandb_log(epochID, metrics, mode)

        if metrics['gen_loss'] < self.gen_loss:
            self.gen_loss = metrics['gen_loss']
            save_model(self.model.eval(), epochID, self.gen_loss, self.exp_cfg.output_loc)
        elif epochID % 5 == 0:
            save_model(self.model.eval(), epochID, metrics['gen_loss'], self.exp_cfg.output_loc)


    def train(self):
        for epochID in range(self.model_cfg.epochs):
            for mode in self.model_cfg.modes:
                self._epoch(epochID, mode)
