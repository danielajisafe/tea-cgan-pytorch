import os
import glob
import pandas as pd
import numpy as np
from PIL import Image

import torch
import math
from torch.utils.data import Dataset
from torchvision import transforms as T


class CelebA(Dataset):
	def __init__(self, datadir, captions, word_vectors, transform, sample_size=None):
		self.captions = pd.read_csv(captions, delimiter='\t', header=None, index_col=0).to_dict()[1]
		nan_keys = [key for key in self.captions if isinstance(self.captions[key], float)]
		for key in nan_keys:
			del self.captions[key]
		self.fnames = [os.path.join(datadir, fname) for fname in self.captions][:sample_size]
		self.word_vectors = pd.read_csv(word_vectors, header=None, delimiter=' ', skiprows=1, index_col=0)
		self.word_vectors.drop(columns=101, inplace=True)
		self.transform = transform
		self.max_len = max([len(str(x).split()) for x in [*self.captions.values()]])

	def word2vec(self, sentence):
		words = sentence.split()
		sentence_vec = np.zeros((self.max_len, 100))
		for i, word in enumerate(words):
			vec = np.array(self.word_vectors.loc[str(word)])
			sentence_vec[i] = vec

		return torch.FloatTensor(sentence_vec)
		
	def __len__(self):
		return len(self.fnames)

	def __getitem__(self, index):
		fname = self.fnames[index]
		image = Image.open(fname).convert('RGB')
		image = self.transform(image)
		
		key = os.path.basename(fname)
		caption = self.captions[key]
		caption_vec = self.word2vec(caption)

		alt = index
		while alt == index:
			alt = np.random.randint(self.__len__())
		mismatch = self.captions[os.path.basename(self.fnames[alt])]
		mismatch_vec = self.word2vec(mismatch)

		return {'image': image,
				'caption_txt': caption,
				'caption': caption_vec,
				'mismatch_txt': mismatch,
				'mismatch': mismatch_vec}


if __name__=='__main__':
	transform = []
	transform.append(T.Resize((128, 128)))
	transform.append(T.ToTensor())
	transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
	transform = T.Compose(transform)
	dataset = CelebA('/data/namrata/celebA/img_align_celeba', './celeba/captions.txt', './celeba/celeba.vec', transform)
	from tqdm import tqdm
	for i in tqdm(range(len(dataset))):
		dataset[i]
