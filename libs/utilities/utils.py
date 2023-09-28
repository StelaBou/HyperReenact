import os
import numpy as np
import cv2
import torch
from torchvision import utils as torch_utils
import glob

from libs.utilities.image_utils import *
from libs.utilities.ffhq_cropping import crop_using_landmarks

def make_path(path):
	if not os.path.exists(path):
		os.makedirs(path, exist_ok = True)
		
def get_image_files(path):
	types = ('*.png', '*.jpg') # the tuple of file types
	files_grabbed = []
	for files in types:
		files_grabbed.extend(glob.glob(os.path.join(path, files)))
	files_grabbed.sort()
	return files_grabbed

def extract_frames(input_video, max_frames = None):
	frames = []
	cap = cv2.VideoCapture(input_video)
	fps = cap.get(cv2.CAP_PROP_FPS)
	
	counter = 0
	fcount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	flag_stop = False
	while cap.isOpened():
		ret, frame = cap.read()		
		if not ret:
			break
		(h, w) = frame.shape[:2]	
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		frames.append(frame)
		if max_frames is not None:
			if counter == max_frames:
				break
		counter += 1
		
	cap.release()
	return np.asarray(frames), fps

def get_landmarks(image, landmarks_est):
	frame_255 = torch_range_1_to_255(image)
	with torch.no_grad():
		landmarks = landmarks_est.get_landmarks_from_batch(frame_255) # torch tensor batch x 68 x 2
	return landmarks

" Crop images using facial landmarks "
def preprocess_image(image_path, landmarks_est, save_filename = None, landmarks = None, return_landmarks = False):
	if os.path.isfile(image_path):
		image = Image.open(image_path)
		image = image.convert('RGB')
		image = np.array(image)
	else:
		image = image_path
	
	if image.shape[1] > 1000:
		image, scale = image_resize(image, width = 1000)
	image_tensor = torch.tensor(np.transpose(image, (2,0,1))).float().cuda()		
	if landmarks is None:
		with torch.no_grad():
			landmarks = landmarks_est.get_landmarks_from_batch(image_tensor.unsqueeze(0))
			landmarks = landmarks[0].detach().cpu().numpy()
			landmarks = np.asarray(landmarks)
		
	img = crop_using_landmarks(image, landmarks)
	if img is not None and save_filename is not None:
		cv2.imwrite(save_filename,  cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)) #cv2.COLOR_RGB2BGR))
	if img is not None and return_landmarks:
		return img, landmarks
	elif img is not None and not return_landmarks:
		return img
	else:
		print('Error with image preprocessing')
		exit()

def save_image(image, save_dir):
	grid = torch_utils.save_image(
		image,
		save_dir,
		normalize=True,
		range=(-1, 1),
	)

def generate_grid_image(source, target, reenacted):
	num_images = source.shape[0] # batch size
	width = 256; height = 256
	grid_image = torch.zeros((3, num_images*height, 3*width))
	for i in range(num_images):
		s = i*height
		e = s + height
		grid_image[:, s:e, :width] = source[i, :, :, :]
		grid_image[:, s:e, width:2*width] = target[i, :, :, :]	
		grid_image[:, s:e, 2*width:] = reenacted[i, :, :, :]
	
	return grid_image

def generate_video(images, video_path, fps = 25):
	
	dim = (images[0].shape[1], images[0].shape[0])
	com_video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'MP4V') , fps, dim)
	
	for image in images:
		image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
		com_video.write(np.uint8(image))
		
	com_video.release()