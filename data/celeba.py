import os
import glob
import pandas as pd
import numpy as np
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms as T


class CelebA(Dataset):
	def __init__(self, datadir, captions, transform):
		self.fnames = glob.glob(os.path.join(datadir, '*.jpg'))
		self.captions = pd.read_csv(captions, delimiter='\t', header=None, index_col=0).to_dict()[1]
		self.transform = transform

	def __len__(self):
		return len(self.fnames)

	def __getitem__(self, index):
		fname = self.fnames[index]
		image = Image.open(fname).convert('RGB')
		image = self.transform(image)
		
		key = os.path.basename(fname)
		caption = self.captions[key]

		alt = index
		while alt == index:
			alt = np.random.randint(self.__len__())
		mismatch = self.captions[os.path.basename(self.fnames[alt])]

		return {'image': image,
				'caption': caption,
				'mismatch': mismatch}


if __name__=='__main__':
	transform = []
	transform.append(T.Resize((128, 128)))
	transform.append(T.ToTensor())
	transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
	transform = T.Compose(transform)
	dataset = CelebA('/data/namrata/celebA/img_align_celeba', './celeba/captions.txt', transform)
	dataset[0]
