"""
Crop images using facial landmarks
"""
import numpy as np
import cv2
import os
import collections
import PIL.Image
import PIL.ImageFile
from PIL import Image
import scipy.ndimage

def pad_img_to_fit_bbox(img, x1, x2, y1, y2, crop_box):
	img_or = img.copy()
	img = cv2.copyMakeBorder(img,
		-min(0, y1), max(y2 - img.shape[0], 0),
		-min(0, x1), max(x2 - img.shape[1], 0), cv2.BORDER_REFLECT)
	
	y2 += -min(0, y1)
	y1 += -min(0, y1)
	x2 += -min(0, x1)
	x1 += -min(0, x1)	

	pad = crop_box
	pad = (max(-pad[0], 0), max(-pad[1], 0), max(pad[2] - img_or.shape[1] , 0), max(pad[3] - img_or.shape[0] , 0))
	
	h, w, _ = img.shape
	y, x, _ = np.ogrid[:h, :w, :1]
	pad = np.array(pad, dtype=np.float32)
	pad[pad == 0] = 1e-10
	mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w-1-x) / pad[2]), 1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h-1-y) / pad[3]))
	img = np.array(img, dtype=np.float32)
	blur = 5.0
	img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
	img += (np.median(img, axis=(0,1)) - img) * np.clip(mask, 0.0, 1.0)

	return img, x1, x2, y1, y2

def crop_from_bbox(img, bbox):
	"""
		bbox: tuple, (x1, y1, x2, y2)
			x: horizontal, y: vertical, exclusive
	"""
	x1, y1, x2, y2 = bbox
	if x1 < 0 or y1 < 0 or x2 > img.shape[1] or y2 > img.shape[0]:
		img, x1, x2, y1, y2 = pad_img_to_fit_bbox(img, x1, x2, y1, y2, bbox)
	return img[y1:y2, x1:x2]

def crop_using_landmarks(image, landmarks):
	image_size = 256
	center = ((landmarks.min(0) + landmarks.max(0)) / 2).round().astype(int)
	size = int(max(landmarks[:, 0].max() - landmarks[:, 0].min(), landmarks[:, 1].max() - landmarks[:, 1].min()))
	try:
		center[1] -= size // 6
	except:
		return None
	
	# Crop images and poses
	h, w, _ = image.shape
	img = Image.fromarray(image)
	crop_box = (center[0]-size, center[1]-size, center[0]+size, center[1]+size)
	image = crop_from_bbox(image, crop_box)
	try:
		img = Image.fromarray(image.astype(np.uint8))
		img = img.resize((image_size, image_size), Image.BICUBIC)
		pix = np.array(img)
		return pix
	except:
		return None
