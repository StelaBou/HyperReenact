import torch
from torch import nn
from torch.nn import Module, ReLU

from libs.models.hypernetworks.refinement_blocks import HyperRefinementBlock, RefinementBlock
from libs.models.hypernetworks.shared_weights_hypernet import SharedWeightsHypernet
from libs.models.reenactment_module import Reenactment_module

stylegan_voxceleb_256_dict = {
	'stylespace_dim':		4928,
	'split_sections':		[512, 512, 512, 512, 512, 512, 512, 512, 256, 256, 128, 128, 64,]
}


class HypernetworkBlocks_edit(Module):

	def __init__(self, opts):
		super(HypernetworkBlocks_edit, self).__init__()

		self.opts = opts
		feature_channels_blend = 512
		if len(opts.layers_to_tune) == 0:
			self.layers_to_tune = list(range(opts.n_hypernet_outputs))
		else:
			self.layers_to_tune = [int(l) for l in opts.layers_to_tune.split(',')]
		self.shared_layers = [0, 2, 3, 5, 6, 8, 9] # '0,2,3,5,6,8,9,11,12,14,15,17,18'
		self.n_outputs = opts.n_hypernet_outputs
		spatial = 7
		batch_norm = False

		# Feature map from pose encoder when DECA is 2048 x 7 x 7 use a conv layer to make it 512 x 7 x 7
		self.conv_transf = nn.Conv2d(2048, 512, kernel_size=(1, 1), stride = 1, padding=0)
		nn.init.normal_(self.conv_transf.weight, mean=0.0, std=0.01)
		
		self.blend_module = Reenactment_module(feature_channels_blend, kernel_size = 1, padding=0)

		self.shared_weight_hypernet = SharedWeightsHypernet(in_size=512, out_size=512, mode=opts.mode)
		self.refinement_blocks = nn.ModuleList()
		
		for layer_idx in range(self.n_outputs):
			if layer_idx in self.layers_to_tune:
				if layer_idx in self.shared_layers:
					refinement_block = HyperRefinementBlock(self.shared_weight_hypernet, n_channels=512, inner_c=128)
				else:
					refinement_block = RefinementBlock(layer_idx, n_channels=512, inner_c=256)
			else:
				refinement_block = None
			self.refinement_blocks.append(refinement_block)



	def forward(self, f_pose, f_app):

		f_pose = self.conv_transf(f_pose)		
		f_combined = self.blend_module(f_pose, f_app)

		weight_deltas = []
		for j in range(self.n_outputs):
			if self.refinement_blocks[j] is not None:
				delta = self.refinement_blocks[j](f_combined)
			else:
				delta = None
			weight_deltas.append(delta)
		
		return weight_deltas

