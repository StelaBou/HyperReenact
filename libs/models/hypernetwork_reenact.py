import math
import torch
from torch import nn
import os

from libs.models.hypernetworks.hypernetwork_blocks import HypernetworkBlocks_edit
from libs.models.gan.StyleGAN2.model import Generator

class Hypernetwork_reenact(nn.Module):

	def __init__(self, opts):
		super(Hypernetwork_reenact, self).__init__()
		self.opts = opts
		self.n_styles = int(math.log(self.opts.output_size, 2)) * 2 - 2
		self.opts.n_hypernet_outputs = 20 # 256 image resolution
		# Define architecture
		self.hypernet = HypernetworkBlocks_edit(opts = self.opts)	
		self.decoder = Generator(self.opts.output_size, 512, 8, channel_multiplier=self.opts.channel_multiplier)
		self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
		# Load weights if needed
		self.load_weights()
		
	def load_weights(self):
		
		if self.opts.checkpoint_path is not None:
			ckpt = torch.load(self.opts.checkpoint_path, map_location='cpu')
			self.hypernet.load_state_dict(self.__get_keys(ckpt, 'hypernet'), strict=True)
			ckpt = torch.load(self.opts.stylegan_weights)
			self.decoder.load_state_dict(ckpt['g_ema'], strict=False)
			self.__load_latent_avg(ckpt)	
		else:
			# print('---- Loading Generator from pretrained path {} ----'.format(self.opts.stylegan_weights))
			ckpt = torch.load(self.opts.stylegan_weights)
			self.decoder.load_state_dict(ckpt['g_ema'], strict=False)
			self.__load_latent_avg(ckpt, repeat=self.n_styles)

	def forward(self, f_pose, f_app, codes=None, weights_deltas=None, truncation = 1.0, trunc = None, input_is_latent = True,
					return_latents = True, return_weight_deltas_and_codes=False, randomize_noise = False):
	
		
		hypernet_outputs = self.hypernet(f_pose, f_app)

		if weights_deltas is None:
			weights_deltas = hypernet_outputs
		else:
			weights_deltas = [weights_deltas[i] + hypernet_outputs[i] if weights_deltas[i] is not None else None
								for i in range(len(hypernet_outputs))]

		images, result_latent = self.decoder([codes],
											 weights_deltas=weights_deltas, truncation = truncation, truncation_latent = trunc,
											 input_is_latent=input_is_latent,
											 randomize_noise=randomize_noise,
											 return_latents=return_latents)

		if return_latents and return_weight_deltas_and_codes:
			return images, codes, weights_deltas
		elif return_latents:
			return images, result_latent
		else:
			return images

	def __load_latent_avg(self, ckpt, repeat=None):
		if 'latent_avg' in ckpt:
			self.latent_avg = ckpt['latent_avg'].to(self.opts.device)
			if repeat is not None:
				self.latent_avg = self.latent_avg.repeat(repeat, 1)
		else:
			self.latent_avg = None

	@staticmethod
	def __get_keys(d, name):
		if 'state_dict' in d:
			d = d['state_dict']
		d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
		
		return d_filt

