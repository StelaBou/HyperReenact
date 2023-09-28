from __future__ import absolute_import
import os
import torch
import torch.nn as nn
import numpy as np
import scipy.misc
import cv2
from matplotlib import pyplot as plt
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
from PIL import Image

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
	# initialize the dimensions of the image to be resized and
	# grab the image size
	dim = None
	(h, w) = image.shape[:2]

	# if both the width and height are None, then return the
	# original image
	if width is None and height is None:
		return image

	# check to see if the width is None
	if width is None:
		# calculate the ratio of the height and construct the
		# dimensions
		r = height / float(h)
		dim = (int(w * r), height)
		scale = r

	# otherwise, the height is None
	elif height is None:
		# calculate the ratio of the width and construct the
		# dimensions
		r = width / float(w)
		dim = (width, int(h * r))
		scale = r
	else:
		dim = (width, height)
		scale = width / float(w)
	
	# resize the image
	resized = cv2.resize(image, dim, interpolation = inter)

	# return the resized image
	
	
	return resized,scale
	
" Load image from path and transform it into the GAN range [-1,1]"
def load_img(image_path):
	transforms_dict = transforms.Compose([
				transforms.Resize((256, 256)),
				transforms.ToTensor(),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

	image = Image.open(image_path)
	image = image.convert('RGB')
	image = transforms_dict(image)
	image = image.unsqueeze(0).cuda()

	return image

" Trasnform torch tensor to numpy images from range [-1,1] to [0,255]"
def tensor_to_image(image_tensor):
	if image_tensor.ndim == 4:
		image_tensor = image_tensor.squeeze(0)

	min_val = -1
	max_val = 1
	image_tensor.clamp_(min=min_val, max=max_val)
	image_tensor.add_(-min_val).div_(max_val - min_val + 1e-5)
	image_tensor = image_tensor.mul(255.0)
	image_tensor = image_tensor.detach().cpu().numpy()
	image_tensor = np.transpose(image_tensor, (1, 2, 0))

	return image_tensor

' image numpy array - > image tensor to generators space [-1,1]'
def image_to_tensor(image):

	max_val = 1
	min_val = -1
	image_tensor = torch.tensor(np.transpose(image,(2,0,1))).float().div(255.0)	
	image_tensor = image_tensor * (max_val - min_val) + min_val
	
	return image_tensor

" Trasnform torch tensor images from range [-1,1] to [0,255]"
def torch_range_1_to_255(image): 
	img_tmp = image.clone()
	min_val = -1
	max_val = 1
	img_tmp.clamp_(min=min_val, max=max_val)
	img_tmp.add_(-min_val).div_(max_val - min_val + 1e-5)
	img_tmp = img_tmp.mul(255.0)
	return img_tmp