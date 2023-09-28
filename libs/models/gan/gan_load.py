import json
import numpy as np
import torch
from torch import nn

from .StyleGAN2.model import Generator as StyleGAN2Generator
from .StyleGAN2.model import Discriminator as StyleGAN2Discriminator

class StyleGAN2Wrapper(nn.Module):
	def __init__(self, g, shift_in_w):
		super(StyleGAN2Wrapper, self).__init__()
		self.style_gan2 = g
		self.shift_in_w = shift_in_w
		self.dim_z = 512
		self.dim_shift = self.style_gan2.style_dim if shift_in_w else self.dim_z
		self.mean_latent = self.style_gan2.mean_latent
		self.get_latent = self.style_gan2.get_latent
		self.n_latent = self.style_gan2.n_latent

	def get_w(self, z):
		w = self.style_gan2.get_latent(z)
		inject_index = self.style_gan2.n_latent
		latent = w.unsqueeze(1).repeat(1, inject_index, 1)
		return latent

	def forward(self, input, input_is_latent=False, return_latents = True, truncation=1, truncation_latent=None):
		# input = input.squeeze(0) # GMACs
		# print('forward', input.shape)
		if return_latents:
			return self.style_gan2([input], return_latents = True, input_is_latent = input_is_latent, truncation = truncation, truncation_latent = truncation_latent)
		else:
			return self.style_gan2([input], return_latents = False, input_is_latent = input_is_latent, truncation = truncation, truncation_latent = truncation_latent)[0]

	def gen_shifted(self, z, shift, return_latents = False, input_is_latent = False, truncation=1, truncation_latent = None, w_plus = False, num_layers = None):
		# print(w_plus, input_is_latent, shift.shape, z.shape, self.shift_in_w, num_layers)
		# quit()
		if not w_plus:
			if self.shift_in_w and not input_is_latent:
				w = self.style_gan2.get_latent(z)
				inject_index = self.style_gan2.n_latent
				latent = w.unsqueeze(1).repeat(1, inject_index, 1)
				if num_layers is None: # add shift in all layers
					shift_rep = shift.unsqueeze(1)
					shift_rep = shift_rep.repeat(1, inject_index, 1)
					latent += shift_rep
				else:
					for i in range(num_layers):
						latent[:, i,:] += shift
				
				return self.forward(latent , input_is_latent=True,  return_latents = return_latents, truncation=truncation, truncation_latent=truncation_latent)

			if not self.shift_in_w:
				print('check not self.shift_in_w' )
				quit()
				return self.forward(z + shift, input_is_latent=False, return_latents = False)

			if self.shift_in_w and input_is_latent:
				inject_index = self.style_gan2.n_latent
				latent = z.clone()
				if latent.shape == (1, 512):
					latent = latent.repeat(1, inject_index, 1)
				if num_layers is None: # add shift in all layers
					shift_rep = shift.unsqueeze(1)
					shift_rep = shift_rep.repeat(1, inject_index, 1)
					latent += shift
				else:
					for i in range(num_layers):
						latent[:, i,:] += shift
				
				return self.forward(latent , input_is_latent=True, return_latents = return_latents, truncation=truncation, truncation_latent=truncation_latent)
		else:
			if self.shift_in_w and not input_is_latent:
				w = self.style_gan2.get_latent(z)
				inject_index = self.style_gan2.n_latent
				latent = w.unsqueeze(1).repeat(1, inject_index, 1)
				latent[:,:shift.shape[1],:] += shift
				return self.forward(latent, input_is_latent = True, return_latents = return_latents, truncation=truncation, truncation_latent=truncation_latent)
			
			if self.shift_in_w and input_is_latent:
				latent = z.clone()
				if latent.ndim == 2:
					inject_index = self.style_gan2.n_latent
					latent = latent.unsqueeze(1).repeat(1, inject_index, 1)     
				latent[:,:shift.shape[1],:] += shift

				# for i in range(num_layers):
				# 	latent[:, i, :] += shift[:, i]
				
				return self.forward(latent , input_is_latent = True, return_latents = return_latents, truncation=0.7, truncation_latent=truncation_latent)


def load_generator_styleGAN(size, weights, shift_in_w = True, strict = True, channel_multiplier = 2):
	G = StyleGAN2Generator(size, 512, 8, channel_multiplier=channel_multiplier)
	if size == 256:
		G.load_state_dict(torch.load(weights)['g_ema'], strict = False)
	else:
		G.load_state_dict(torch.load(weights)['g_ema'], strict = True)
	G.cuda().eval()

	return StyleGAN2Wrapper(G, shift_in_w=shift_in_w)

def load_discriminator_styleGAN(size, weights):

	D = StyleGAN2Discriminator(size, channel_multiplier=1)
	D.load_state_dict(torch.load(weights)['d'], strict=False)

	return D