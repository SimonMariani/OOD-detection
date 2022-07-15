import os
import numpy as np
from torch.utils.data.dataset import Dataset
from imread import imread


class DatasetFromSubset(Dataset):
    
	def __init__(self, subset, transform=None, target_transform=None):
		r"""Essentially this class allows for a subset object with a transform. One thing that needs to be taken into account
		is that if the original dataset from which the subset is taken already has transforms that these transforms are added 
		on top of those."""

		self.subset = subset
		self.transform = transform
		self.target_transform = target_transform

	def __getitem__(self, index):
		x, y = self.subset[index]

		if self.transform:
			x = self.transform(x)
		
		if self.target_transform:
			y = self.target_transform(y)

		return x, y

	def __len__(self):
		return len(self.subset)


class notMNIST(Dataset):

	# The init method is called when this class will be instantiated.
	def __init__(self, root, transform=None, target_transform=None, class_samples=1000):
		r"""The notMNIST class found online"""

		images = []
		Y = []
		folders = os.listdir(root)

		self.transform = transform
		self.target_transform = target_transform

		for folder in folders:
			folder_path = os.path.join(root, folder)
			for i, ims in enumerate(os.listdir(folder_path)):
				try:
					img_path = os.path.join(folder_path, ims)
					images.append(np.array(imread(img_path)))
					Y.append(ord(folder) - 65)  # Folders are A-J so labels will be 0-9
				except:
					# Some images in the dataset are damaged
					print("File {}/{} is broken".format(folder, ims))

				if class_samples is not None:
					if i > class_samples - 1:
						break

			print(f'loaded {folder}')

		data = [(x, y) for x, y in zip(images, Y)]
		self.data = data

	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		img = self.data[index][0]
		target = self.data[index][1]

		if self.transform is not None:
			img =self.transform(img)

		if self.target_transform is not None:
			target = self.target_transform(target)

		return (img, target)
