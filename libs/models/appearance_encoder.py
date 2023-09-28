import torch
from torch.nn import Module

from libs.criteria.helpers import l2_norm
from libs.criteria.csim import Backbone as Backbone_landmarks
from libs.criteria.csim import find_affine_transformation


class ArcFaceEncoder(Module):
	def __init__(self, num_layer = 23):
		super(ArcFaceEncoder, self).__init__()
		self.num_layer = num_layer

		self.facenet = Backbone_landmarks(num_layers=50, drop_ratio=0.6, mode='ir_se')
	
	def forward(self, x, real_poses):

		b, c, h, w = x.shape
		affine_matrices = find_affine_transformation(real_poses, h, w)   
		batch_size = b
		grid = torch.nn.functional.affine_grid(affine_matrices, torch.Size((batch_size, 3, 112, 112)))
		warped_image = torch.nn.functional.grid_sample(x, grid)
		warped_image_keep = warped_image.clone()
		warped_image = self.facenet.input_layer(warped_image)
		count = 0
		for module in self.facenet.body.children():
			warped_image = module(warped_image)
			if count == self.num_layer:
				id_feat_map = warped_image
			count += 1
		
		warped_image = self.facenet.output_layer(warped_image)

		return l2_norm(warped_image), id_feat_map 
