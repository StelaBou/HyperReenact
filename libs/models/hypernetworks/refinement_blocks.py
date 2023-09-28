import numpy as np
from torch import nn
from torch.nn import Conv2d, Sequential, Module
import torch

from libs.models.encoders.helpers import SeparableBlock
from libs.models.gan.StyleGAN2.model import EqualLinear


# layer_idx: [kernel_size, in_channels, out_channels]
PARAMETERS_256_voxceleb = {
	0: [3, 512, 512],
	1: [1, 512, 3],
	2: [3, 512, 512],
	3: [3, 512, 512],
	4: [1, 512, 3],
	5: [3, 512, 512],
	6: [3, 512, 512],
	7: [1, 512, 3],
	8: [3, 512, 512],
	9: [3, 512, 512],
	10: [1, 512, 3],
	11: [3, 512, 256],
	12: [3, 256, 256],
	13: [1, 256, 3],
	14: [3, 256, 128],
	15: [3, 128, 128],
	16: [1, 128, 3],
	17: [3, 128, 64],
	18: [3, 64, 64],
	19: [1, 64, 3],
}
TO_RGB_LAYERS = [1, 4, 7, 10, 13, 16, 19, 22, 25]


class RefinementBlock(Module):

	def __init__(self, layer_idx, n_channels=512, inner_c=256):
		super(RefinementBlock, self).__init__()
		self.layer_idx = layer_idx		
		self.kernel_size, self.in_channels, self.out_channels = PARAMETERS_256_voxceleb[self.layer_idx]
	
		self.n_channels = n_channels
		self.inner_c = inner_c
		self.out_c = 512
		self.modules = []
		self.modules = [Conv2d(self.n_channels, self.inner_c, kernel_size=3, stride=1, padding=1), nn.LeakyReLU()]
		self.modules += [Conv2d(self.inner_c, self.inner_c, kernel_size=3, stride=1, padding=0), nn.LeakyReLU()]
		self.modules += [Conv2d(self.inner_c, self.inner_c, kernel_size=3, stride=1, padding=0), nn.LeakyReLU()]
		self.modules += [Conv2d(self.inner_c, self.out_c, kernel_size=3, stride=1, padding=0), nn.LeakyReLU()]
		self.convs = nn.Sequential(*self.modules)
		self.output = Conv2d(self.out_c, self.in_channels * self.out_channels, kernel_size=1, stride=1, padding=0)

	def forward(self, x):
		
		x = self.convs(x)
		x = self.output(x)	
		if self.layer_idx in TO_RGB_LAYERS:
			x = x.view(-1, self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
		else:
			x = x.view(-1, self.out_channels, self.in_channels)
			x = x.unsqueeze(3).repeat(1, 1, 1, self.kernel_size).unsqueeze(4).repeat(1, 1, 1, 1, self.kernel_size)
		
		return x

class HyperRefinementBlock(Module):
	def __init__(self, hypernet, n_channels=512, inner_c=128):
		super(HyperRefinementBlock, self).__init__()
		
		self.n_channels = n_channels
		self.inner_c = inner_c
		self.out_c = 512

		modules = [Conv2d(self.n_channels, self.inner_c, kernel_size=3, stride=1, padding=1), nn.LeakyReLU()]
		modules += [Conv2d(self.inner_c, self.inner_c, kernel_size=3, stride=1, padding=0), nn.LeakyReLU()]
		modules += [Conv2d(self.inner_c, self.inner_c, kernel_size=3, stride=1, padding=0), nn.LeakyReLU()]
		modules += [Conv2d(self.inner_c, self.out_c, kernel_size=3, stride=1, padding=0), nn.LeakyReLU()]

		self.convs = nn.Sequential(*modules)
		
		self.linear = nn.Linear(self.out_c, self.out_c)
		nn.init.normal_(self.linear.weight, mean=0.0, std=0.01)
		self.hypernet = hypernet

	def forward(self, features):

		code = self.convs(features)
		code = code.view(-1, self.out_c)
		code = self.linear(code)
		weight_delta = self.hypernet(code)
		return weight_delta


