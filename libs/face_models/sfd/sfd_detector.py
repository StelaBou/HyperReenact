import os
import cv2
from torch.utils.model_zoo import load_url
import sys
import matplotlib.pyplot as plt
from .core import FaceDetector

from .net_s3fd import s3fd
from .bbox import *
from .detect import *
import torch.backends.cudnn as cudnn


models_urls = {
	's3fd': 'https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth',
}


class SFDDetector(FaceDetector):
	def __init__(self, device, path_to_detector=None, verbose=False):
		super(SFDDetector, self).__init__(device, verbose)
		
		self.device = device
		model_weights = torch.load(path_to_detector)

		self.face_detector = s3fd()
		self.face_detector.load_state_dict(model_weights)
		self.face_detector.to(self.device)
		self.face_detector.eval()

	def detect_from_batch(self, tensor):
		
		bboxlists = batch_detect(self.face_detector, tensor, device=self.device)
		
		new_bboxlists = []
		for i in range(bboxlists.shape[0]):
			bboxlist = bboxlists[i]
			keep = nms(bboxlist, 0.3)
			# print(keep)
			if len(keep)>0:
				bboxlist = bboxlist[keep, :]
				bboxlist = [x for x in bboxlist if x[-1] > 0.5]			
				new_bboxlists.append(bboxlist)            
			else: 
				new_bboxlists.append([])

		return new_bboxlists

	@property
	def reference_scale(self):
		return 195

	@property
	def reference_x_shift(self):
		return 0

	@property
	def reference_y_shift(self):
		return 0
