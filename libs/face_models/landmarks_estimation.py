"""
Calculate euler angles yaw pitch roll using deep network HopeNet
https://github.com/natanielruiz/deep-head-pose

The face detector used is SFD (taken from face-alignment FAN) https://github.com/1adrianb/face-alignment

"""
import os 
import numpy as np
import cv2
from enum import Enum
import torch
from torch.utils.model_zoo import load_url


from libs.face_models.sfd.sfd_detector import SFDDetector as FaceDetector
from libs.face_models.fan_model.models import FAN, ResNetDepth
from libs.face_models.fan_model.utils import *

models_urls = {
	'2DFAN-4': 'https://www.adrianbulat.com/downloads/python-fan/2DFAN4-11f355bf06.pth.tar',
	'3DFAN-4': 'https://www.adrianbulat.com/downloads/python-fan/3DFAN4-7835d9f11d.pth.tar',
	'depth': 'https://www.adrianbulat.com/downloads/python-fan/depth-2a464da4ea.pth.tar',
}

class LandmarksType(Enum):
	"""Enum class defining the type of landmarks to detect.

	``_2D`` - the detected points ``(x,y)`` are detected in a 2D space and follow the visible contour of the face
	``_2halfD`` - this points represent the projection of the 3D points into 3D
	``_3D`` - detect the points ``(x,y,z)``` in a 3D space

	"""
	_2D = 1
	_2halfD = 2
	_3D = 3

class NetworkSize(Enum):
	# TINY = 1
	# SMALL = 2
	# MEDIUM = 3
	LARGE = 4

	def __new__(cls, value):
		member = object.__new__(cls)
		member._value_ = value
		return member

	def __int__(self):
		return self.value


def get_preds_fromhm(hm, center=None, scale=None):
	"""Obtain (x,y) coordinates given a set of N heatmaps. If the center
	and the scale is provided the function will return the points also in
	the original coordinate frame.

	Arguments:
		hm {torch.tensor} -- the predicted heatmaps, of shape [B, N, W, H]

	Keyword Arguments:
		center {torch.tensor} -- the center of the bounding box (default: {None})
		scale {float} -- face scale (default: {None})
	"""
	max, idx = torch.max(
		hm.view(hm.size(0), hm.size(1), hm.size(2) * hm.size(3)), 2)
	idx = idx + 1
	preds = idx.view(idx.size(0), idx.size(1), 1).repeat(1, 1, 2).float()
	preds[..., 0].apply_(lambda x: (x - 1) % hm.size(3) + 1)
	preds[..., 1].add_(-1).div_(hm.size(2)).floor_().add_(1)

	for i in range(preds.size(0)):
		for j in range(preds.size(1)):
			hm_ = hm[i, j, :]
			pX, pY = int(preds[i, j, 0]) - 1, int(preds[i, j, 1]) - 1
			if pX > 0 and pX < 63 and pY > 0 and pY < 63:
				diff = torch.FloatTensor(
					[hm_[pY, pX + 1] - hm_[pY, pX - 1],
					 hm_[pY + 1, pX] - hm_[pY - 1, pX]])
				preds[i, j].add_(diff.sign_().mul_(.25))

	preds.add_(-.5)

	preds_orig = torch.zeros(preds.size())
	if center is not None and scale is not None:
		for i in range(hm.size(0)):
			for j in range(hm.size(1)):
				preds_orig[i, j] = transform(
					preds[i, j], center, scale, hm.size(2), True)

	return preds, preds_orig



class LandmarksEstimation():
	def __init__(self, type = '3D', path_to_detector = './pretrained_models/s3fd-619a316812.pth'):
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		# Load all needed models - Face detector and Pose detector
		network_size = NetworkSize.LARGE
		network_size = int(network_size)
		if type == '3D':
			self.landmarks_type = LandmarksType._3D
		else:
			self.landmarks_type = LandmarksType._2D
		self.flip_input = False

		#################### SFD face detection ###################
		if not os.path.exists(path_to_detector):
			print('Pretrained model of SFD face detector does not exist in {}'.format(path_to_detector))
			exit()
		self.face_detector = FaceDetector(device=self.device, verbose=False, path_to_detector = path_to_detector)
		###########################################################

		################### Initialise the face alignemnt networks ###################
		self.face_alignment_net = FAN(network_size)
		if self.landmarks_type == LandmarksType._2D: #
			network_name = '2DFAN-' + str(network_size)
		else:
			network_name = '3DFAN-' + str(network_size)
		fan_weights = load_url(models_urls[network_name], map_location=lambda storage, loc: storage)
		self.face_alignment_net.load_state_dict(fan_weights)
		self.face_alignment_net.to(self.device)
		self.face_alignment_net.eval()
		##############################################################################

		# Initialiase the depth prediciton network if 3D landmarks
		if self.landmarks_type  == LandmarksType._3D:
			self.depth_prediciton_net = ResNetDepth()
			depth_weights = load_url(models_urls['depth'], map_location=lambda storage, loc: storage)
			depth_dict = {
				k.replace('module.', ''): v for k,
				v in depth_weights['state_dict'].items()}
			self.depth_prediciton_net.load_state_dict(depth_dict)
			self.depth_prediciton_net.to(self.device)
			self.depth_prediciton_net.eval()

	def get_landmarks(self, face, image):

		center = torch.FloatTensor(
			[(face[2] + face[0]) / 2.0,
				(face[3] + face[1]) / 2.0])

		center[1] = center[1] - (face[3] - face[1]) * 0.12
		scale = (face[2] - face[0] + face[3] - face[1]) / self.face_detector.reference_scale
		
		inp = crop_torch(image, center, scale).float().cuda()
		inp = inp.div(255.0)
		
		out = self.face_alignment_net(inp)[-1]

		if self.flip_input:
			out = out + flip(self.face_alignment_net(flip(inp))
						[-1],  is_label=True)  # patched inp_batch undefined variable error
		out = out.cpu()

		pts, pts_img = get_preds_fromhm(out, center, scale)
		out = out.cuda()
		# Added 3D landmark support
		if self.landmarks_type == LandmarksType._3D:
			pts, pts_img = pts.view(68, 2) * 4, pts_img.view(68, 2)
			heatmaps = torch.zeros((68,256,256), dtype=torch.float32)
			for i in range(68):
				if pts[i, 0] > 0:
					heatmaps[i] = draw_gaussian(
						heatmaps[i], pts[i], 2)
		
			heatmaps = heatmaps.unsqueeze(0)
			
			heatmaps = heatmaps.to(self.device)
			depth_pred = self.depth_prediciton_net(
				torch.cat((inp, heatmaps), 1)).view(68, 1)  #.data.cpu().view(68, 1)
			# print(depth_pred.view(68, 1).shape)
			pts_img = pts_img.cuda()
			pts_img = torch.cat(
				(pts_img, depth_pred * (1.0 / (256.0 / (200.0 * scale)))), 1)	
		else:
			pts, pts_img = pts.view(-1, 68, 2) * 4, pts_img.view(-1, 68, 2)
		
		return pts_img, out

	def detect_landmarks(self, image):

		if len(image.shape) == 3:
			image = image.unsqueeze(0)
		
		if self.device == 'cuda':
			image = image.cuda()

		with torch.no_grad():
			detected_faces = self.face_detector.detect_from_batch(image)
			if self.landmarks_type == LandmarksType._3D:
				landmarks = torch.empty((1, 68, 3))
			else:
				landmarks = torch.empty((1, 68, 2))
			
			for face in detected_faces[0]:
				conf = face[4]	
				if conf > 0.80:	
					pts_img, heatmaps = self.get_landmarks(face, image)
					landmarks[0] = pts_img
		return landmarks

	def get_landmarks_from_batch(self, image_batch, return_faces = False):
		"""Predict the landmarks for each face present in the image.

		This function predicts a set of 68 2D or 3D images, one for each image in a batch in parallel.
		If detect_faces is None the method will also run a face detector.

		 Arguments:
			image_batch {torch.tensor} -- The input images batch

		Keyword Arguments:
			detected_faces {list of numpy.array} -- list of bounding boxes, one for each face found
			in the image (default: {None})
			return_bboxes {boolean} -- If True, return the face bounding boxes in addition to the keypoints.
			return_landmark_score {boolean} -- If True, return the keypoint scores along with the keypoints.

		Return:
			result:
				1. if both return_bboxes and return_landmark_score are False, result will be:
					landmarks
				2. Otherwise, result will be one of the following, depending on the actual value of return_* arguments.
					(landmark, landmark_score, detected_face)
					(landmark, None,           detected_face)
					(landmark, landmark_score, None         )
		"""

		detected_faces = self.face_detector.detect_from_batch(image_batch)
		
		if self.landmarks_type == LandmarksType._3D:
			landmarks = torch.empty((image_batch.shape[0], 68, 3))
		else:
			landmarks = torch.empty((image_batch.shape[0], 68, 2))

		if len(detected_faces) == 0:
			warnings.warn("No faces were detected.")
			return None

		# A batch for each frame
		for i, faces in enumerate(detected_faces):
			
			if len(faces) > 1:
				confs = [sublist[-1] for sublist in faces]
				max_conf = np.max(confs)
				max_index = confs.index(max_conf)
				bbox = faces[max_index]
				pts_img, heatmaps = self.get_landmarks(bbox, image_batch[i].unsqueeze(0))
				landmarks[i] = pts_img[0]

			elif len(faces) == 1:
				pts_img, heatmaps = self.get_landmarks(faces[0], image_batch[i].unsqueeze(0))
				landmarks[i] = pts_img[0]

		if return_faces:
			confs = []
			for i, faces in enumerate(detected_faces):
				conf = [sublist[-1] for sublist in faces]
				confs.append(conf)
			return landmarks, confs 
		else:
			return landmarks