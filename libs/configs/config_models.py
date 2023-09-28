import os
import numpy as np

model_arguments = {
	'image_resolution':				256,
	'channel_multiplier':			1,
	'generator_weights':		    './pretrained_models/stylegan-voxceleb.pt',

	'deca_layer':					'layer4',
	'arcface_layer':				23,
	'hypernet_type':				'SharedWeightsHyperNetResNet',
	'use_truncation':				True,
	'layers_to_tune':				'0,2,3,5,6,8,9,11,12,14,15,17,18',
	'kernel_size':					1,
	'pretrained_pose_encoder':		'Deca',
	'pretrained_app_encoder':		'ArcFace',
	'mode':							'delta_per_channel',

	'pose_encoder_path':			'./pretrained_models/data/deca_model.tar', 
	'app_encoder_path':				'./pretrained_models/insight_face.pth', 
	'e4e_path':						'./pretrained_models/e4e-voxceleb.pt',
	'sfd_detector_path':			'./pretrained_models/s3fd-619a316812.pth',
}


