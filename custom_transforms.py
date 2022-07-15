import torch
import torch.nn as nn
import torch.nn.functional as F


class UniformLabels(nn.Module):

	def __init__(self, num_classes=10, size=1):
		super().__init__()
		self.size = size
		self.num_classes = num_classes

	def forward(self, labels):

		return (torch.ones((self.num_classes)) * self.size).float()

	def __repr__(self):
		return self.__class__.__name__ + f"(size={self.size})"


class AdditionalClassLabel(nn.Module):
	def __init__(self, num_classes=10):
		super().__init__()
		self.num_classes = num_classes

	def forward(self, labels):
		return torch.tensor(self.num_classes)

	def __repr__(self):
		return self.__class__.__name__ + f"(num_classes={self.num_classes})"


class OneHotEncoding(nn.Module):

	def __init__(self, num_classes=10):
		super().__init__()
		self.num_classes = num_classes

	def forward(self, labels):
		return F.one_hot(torch.tensor(labels), self.num_classes).float()

	def __repr__(self):
		return self.__class__.__name__ + f"(num_classes={self.num_classes})"


class SmoothOneHotEncoding(nn.Module):

	def __init__(self, num_classes=10, precision=100):
		super().__init__()
		self.num_classes = num_classes
		self.precision = precision

	def forward(self, labels):

		if self.precision > self.num_classes:
			labels = F.one_hot(torch.tensor(labels), self.num_classes)
			return (labels *(self.precision - self.num_classes) + 1).float()
		else:
			raise ValueError("The precision value must be larger than the number of classes")

	def __repr__(self):
		return self.__class__.__name__ + f"(num_classes={self.num_classes}, precision={self.precision})"
