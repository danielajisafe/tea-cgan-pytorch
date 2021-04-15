from torchvision import transforms as T
from torch.utils.data import DataLoader

from data.celeba import CelebA


def get_loader(data_cfg, mode='train'):
	transform = []
	transform.append(T.RandomHorizontalFlip())
	transform.append(T.Resize((128, 128)))
	transform.append(T.ToTensor())
	transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
	transform = T.Compose(transform)

	dataset = CelebA(data_cfg.datadir, data_cfg.captions, data_cfg.word_vectors, transform, data_cfg.sample_size)

	loader = DataLoader(dataset, batch_size=data_cfg.batch_size,
						shuffle=True, drop_last=True,
						num_workers=data_cfg.num_workers)

	return loader
