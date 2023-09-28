import math
import torch.nn as nn
import torch


class SPADE(nn.Module):
	def __init__(self, input_channels, kernel_size = 3, padding=1):
		super().__init__()

		self.conv_gamma = nn.Conv2d(input_channels, input_channels, kernel_size=(kernel_size, kernel_size), stride = 1, padding=padding)
		self.conv_beta = nn.Conv2d(input_channels, input_channels, kernel_size=(kernel_size, kernel_size), stride = 1, padding=padding)

		torch.nn.init.normal_(self.conv_gamma.weight, mean=0.0, std=0.01)
		torch.nn.init.normal_(self.conv_beta.weight, mean=0.0, std=0.01)

	def forward(self, x):

		gamma = self.conv_gamma(x)
		beta = self.conv_beta(x)

		return gamma, beta

class Reenactment_module(nn.Module):
	def __init__(self, input_channels, kernel_size = 3, padding=1):
		super(Reenactment_module, self).__init__()
		
		self.spade_pose = SPADE(input_channels, kernel_size = kernel_size, padding = padding)
		self.spade_app = SPADE(input_channels, kernel_size = kernel_size, padding = padding)
		
	def forward(self, f_p, f_app):
		"""
			Input:
				f_a: appearance feature maps
				f_p: pose feature maps
		"""
		
		gamma_p, beta_p = self.spade_pose(f_p)
		gamma_a, beta_a = self.spade_app(f_app)

		f_combined = torch.matmul(gamma_p, f_p) + torch.matmul(gamma_a, f_app) + beta_p + beta_a
		
		return f_combined

