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


	def _get_iterator(self):
		data_iter = iter(self.dataloader)
		tqdm_iter = tqdm(range(len(self.dataloader)))

		return data_iter, tqdm_iter

	def _epoch(self, epochID:int, mode='train'):
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

			#TODO

	def train(self):
		for epochID in range(self.model_cfg.epochs):
			for mode in self.model_cfg.modes:
				self._epoch(epochID, mode)