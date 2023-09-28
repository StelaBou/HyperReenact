import torch
import torch.nn as nn
from torch.nn import Sequential, Module

from libs.DECA.decalib.models import resnet


class DECAEncoder(Module):
	def __init__(self, layer = 'layer4'):
		super(DECAEncoder, self).__init__()
		
		feature_size = 2048
		outsize = 236
		self.layer = layer
		self.encoder = resnet.load_ResNet50Model() 
		
		### regressor ###
		self.layers = Sequential(
			nn.Linear(feature_size, 1024),
			nn.ReLU(),
			nn.Linear(1024, outsize)
		)

	def forward(self, x):
		# torch.Size([1, 3, 224, 224])
		# torch.Size([1, 64, 112, 112])
		# torch.Size([1, 64, 56, 56])
		# torch.Size([1, 256, 56, 56])
		# torch.Size([1, 512, 28, 28])
		# torch.Size([1, 1024, 14, 14])
		# torch.Size([1, 2048, 7, 7])
		x = self.encoder.conv1(x)
		x = self.encoder.bn1(x)
		x = self.encoder.relu(x)
		x = self.encoder.maxpool(x)
		x = self.encoder.layer1(x)
		x = self.encoder.layer2(x) #  512, 28, 28
		x = self.encoder.layer3(x)
		x = self.encoder.layer4(x)
		# if self.layer == 'layer4':
		pose_feat_map = x
		x = self.encoder.avgpool(x)
		x = x.view(x.size(0), -1)

		parameters = self.layers(x)
		
		return parameters, pose_feat_map