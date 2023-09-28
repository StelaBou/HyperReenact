import os
import numpy as np
from PIL import Image
import torch
import warnings
import sys
from tqdm import tqdm
warnings.filterwarnings("ignore")
sys.dont_write_bytecode = True
import argparse 
from argparse import Namespace
import random
import sys
sys.path.append(".")
sys.path.append("..")

from libs.face_models.landmarks_estimation import LandmarksEstimation
from libs.models.pose_encoder import DECAEncoder
from libs.models.appearance_encoder import ArcFaceEncoder
from libs.models.encoders.psp_encoders import Encoder4Editing
from libs.models.hypernetwork_reenact import Hypernetwork_reenact
from libs.DECA.decalib.datasets import datasets 


from libs.utilities.image_utils import *
from libs.configs.config_models import *
from libs.utilities.utils import *

root_path =  os.getcwd()
random.seed(0)

class Inference():

	def __init__(self, args):
		self.args = args
		self.device = 'cuda'

		self.source_path = args['source_path']
		self.target_path = args['target_path']
		self.output_path = args['output_path']
		make_path(self.output_path)

		self.model_path = args['model_path']

		self.save_grids = args['save_grids']
		self.save_images = args['save_images']
		self.save_video = args['save_video']
		
		####################################

		self.image_resolution = model_arguments['image_resolution']
		self.deca_layer = model_arguments['deca_layer']
		self.arcface_layer = model_arguments['arcface_layer']
		self.pose_encoder_path = model_arguments['pose_encoder_path']
		self.app_encoder_path = model_arguments['app_encoder_path']
		self.e4e_path = model_arguments['e4e_path']
		self.sfd_detector_path = model_arguments['sfd_detector_path']

	def load_auxiliary_models(self):
		self.landmarks_est =  LandmarksEstimation(type = '2D', path_to_detector = self.sfd_detector_path)
		
		################ Pose encoder ################
		print('********* Upload pose encoder *********')
		self.pose_encoder = DECAEncoder(layer = self.deca_layer).to(self.device) # resnet50 pretrained for DECA eval mode
		self.posedata = datasets.TestData()
		ckpt = torch.load(self.pose_encoder_path, map_location='cpu')
		d = ckpt['E_flame']					
		self.pose_encoder.load_state_dict(d)
		self.pose_encoder.eval()
		##############################################
		
		############# Appearance encoder #############
		print('********* Upload appearance encoder *********')
		self.appearance_encoder = ArcFaceEncoder(num_layer = self.arcface_layer).to(self.device) # ArcFace model
		ckpt = torch.load(self.app_encoder_path, map_location='cpu')
		d_filt = {'facenet.{}'.format(k) : v for k, v in ckpt.items() }
		self.appearance_encoder.load_state_dict(d_filt)
		self.appearance_encoder.eval()
		#############################################

		print('********* Upload Encoder4Editing *********')
		self.encoder = Encoder4Editing(50, 'ir_se', self.image_resolution).to(self.device)
		ckpt = torch.load(self.e4e_path)
		self.encoder.load_state_dict(ckpt['e']) 
		self.encoder.eval()
		
	def load_models(self):

		self.load_auxiliary_models()

		print('********* Upload HyperReenact *********')
		opts = {}
		opts['root_path'] = root_path
		opts['device'] = self.device
		opts['deca_layer'] = self.deca_layer
		opts['arcface_layer'] = self.arcface_layer
		opts['checkpoint_path'] = self.model_path
		opts['output_size'] = self.image_resolution
		opts['channel_multiplier'] =model_arguments['channel_multiplier']
		opts['layers_to_tune'] = model_arguments['layers_to_tune']
		opts['mode'] = model_arguments['mode']
		opts['stylegan_weights'] = model_arguments['generator_weights']
		

		opts = Namespace(**opts)
		self.net = Hypernetwork_reenact(opts).to(self.device)
		self.net.eval()
		

		if self.net.latent_avg is None:
			self.net.latent_avg = self.net.decoder.mean_latent(int(1e5))[0].detach()
	
		self.truncation = 0.7
		self.trunc = self.net.decoder.mean_latent(4096).detach().clone()
					
	def get_identity_embeddings(self, image):
		
		landmarks = get_landmarks(image, self.landmarks_est)
		id_hat, f_app = self.appearance_encoder(image, landmarks) # f_app 256 x 14 x 14 and id_hat 512
			
		return 	id_hat, f_app

	def get_pose_embeddings(self, image):
		# Preprocess like DECA the input image for pose encoder
		image_pose = image.clone()
		image_prepro = torch.zeros(image_pose.shape[0], 3, 224, 224).cuda()
		for k in range(image_pose.shape[0]):
			min_val = -1
			max_val = 1
			image_pose[k].clamp_(min=min_val, max=max_val)
			image_pose[k].add_(-min_val).div_(max_val - min_val + 1e-5)
			image_pose[k] = image_pose[k].mul(255.0).add(0.0) 
			image_prepro_, error_flag = self.posedata.get_image_tensor(image_pose[k])
			image_prepro[k] = image_prepro_		
		pose_hat, f_pose = self.pose_encoder(image_prepro) #  512, 28, 28

		return pose_hat, f_pose

	def forward_model(self, source_img, target_img, shifted_codes):
		with torch.no_grad():	
							
			# Get identity embeddings
			id_hat, f_app = self.get_identity_embeddings(source_img)
			# Get pose embeddings
			pose_hat, f_pose = self.get_pose_embeddings(target_img)
			
			reenacted_image, shifted_codes, shifted_weights_deltas = self.net.forward(f_pose = f_pose,
														f_app = f_app, 
														codes = shifted_codes,
														truncation = self.truncation, trunc = self.trunc,
														return_latents=True,
														return_weight_deltas_and_codes=True
														)

		return reenacted_image
	
	def load_source_image(self):

		head, tail = os.path.split(self.source_path)
		ext = tail.split('.')[-1]
		if ext != 'png' and ext != 'jpg':
			print('Wrong source path. Expected file image (.png, .jpg)')
			exit()
		self.input_is_latent = True
		# Step 1: Preprocess image using landmarks
		cropped_image = preprocess_image(self.source_path, self.landmarks_est, save_filename = None)	
		# Step 2: Invert image into the latent space of StyleGAN2 using e4e encoder
		source_image = image_to_tensor(cropped_image).unsqueeze(0).cuda()
		source_code = self.encoder(source_image)
		return source_image, source_code

	def load_image(self, image):
		
		image = preprocess_image(image, self.landmarks_est)
		image = image_to_tensor(image).unsqueeze(0).cuda()
		return image

	def load_target_data(self):
		image_files = None
		# Check if target path is a directory
		if os.path.isdir(self.target_path):
			image_files = get_image_files(self.target_path)
			if len(image_files) == 0:
				print('There are no target images in {}'.format(self.target_path))
				exit()
			
		elif os.path.isfile(self.target_path):
			head, tail = os.path.split(self.target_path)
			ext = tail.split('.')[-1]
			# Check if file is image
			if ext == 'png' or ext == 'jpg':
				image_files = [self.target_path]
			# Check if file is image
			elif ext == 'mp4' or ext == 'avi':
				# Change FPS if needed
				image_files, fps = extract_frames(self.target_path, max_frames = 100) # Add a max number of frames to extract if needed
			else:
				print('Please specify correct target path. Extension should be .png, .jpg or .mp4')
				exit()
		else:
			print('Please specify correct target path: directory with images or image file (.png, .jpg) or video (.mp4)')
			exit()

		return image_files

	def run_reenactment(self):

		if self.save_grids:
			output_path_grids = os.path.join(self.output_path, 'grids')
			make_path(output_path_grids)

		self.load_models()
		source_image, source_code = self.load_source_image()
		
		target_images = self.load_target_data()
		print('Run reenactment for {} images. Save results into {}'.format(len(target_images), self.output_path))
		
		reenacted_images = []
		for i, target_image in enumerate(tqdm(target_images)):
			## Load target image
			target_image = self.load_image(target_image)	
			reenacted_image = self.forward_model(source_image, target_image, source_code)
			
			if self.save_images:
				target_index = '{:06d}.png'.format(i) 
				save_image(reenacted_image, os.path.join(self.output_path, target_index))
			if self.save_grids:
				target_index = '{:06d}.png'.format(i) 
				grid = generate_grid_image(source_image, target_image, reenacted_image)
				save_image(grid, os.path.join(output_path_grids, target_index))

			if self.save_video:
				reenacted_image = tensor_to_image(reenacted_image)
				reenacted_images.append(reenacted_image)

		if self.save_video:
			video_id = 'reenacted_video'
			fps = 25
			video_path = os.path.join(self.output_path, '{}.mp4'.format(video_id))
			generate_video(reenacted_images, video_path, fps = fps) 
	
def main():
	"""
	Inference script. Generate reenactment resuls.
	Input: 
		--source image: 			Reenact the source image 
		--target images/video:  	Driving frames 
	Output:
		--reenacted images			

	Options:
		######### General ###########
		--source_path						: Path to source frame. Type: image (.png or .jpg)
		--target_path						: Path to target frames. Type: image (.png or .jpg), video (.mp4) or folder path with images
		--output_path						: Path to save the results.

		######### Visualization #########
		--save_grids						: save source-target-reenacted grid
		--save_images						: save only reenacted images
		--save_video						: save results on video

	Example:

	python run_inference.py --source_path ./inference_examples/source.png \
							--target_path ./inference_examples/target_video.mp4 \
							--output_path ./results --save_video

	"""
	parser = argparse.ArgumentParser(description="inference script")

	######### General ###########
	parser.add_argument('--source_path', type=str, required = True, help="path to source identity")
	parser.add_argument('--target_path', type=str, required = True, help="path to target pose")
	parser.add_argument('--output_path', type=str, required = True, help="path to save the results")

	######### Generator #########
	parser.add_argument('--model_path', type=str, default='./pretrained_models/hypernetwork.pt', help="set pre-trained e4e model path")

	# Default is False. Use argument for True
	parser.add_argument('--save_grids', dest='save_grids', action='store_true', help="save results on a grid (source, target, reenacted)")
	parser.set_defaults(save_grids=False)
	parser.add_argument('--save_images', dest='save_images', action='store_true', help="save reenacted images")
	parser.set_defaults(save_images=False)
	parser.add_argument('--save_video', dest='save_video', action='store_true', help="save results on video (source, target, reenacted)")
	parser.set_defaults(save_video=False)
	

	# Parse given arguments
	args = parser.parse_args()	
	args = vars(args) # convert to dictionary

	inference = Inference(args)
	inference.run_reenactment()
	


if __name__ == '__main__':
	main()

	
	






