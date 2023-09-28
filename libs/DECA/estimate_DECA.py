"""
The pipeline of 3DDFA prediction: given one image, predict the 3d face vertices, 68 landmarks and visualization.
"""

import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import os
import scipy.io as sio
import argparse
import torch.backends.cudnn as cudnn
from torchvision import utils as torch_utils

from .decalib.deca import DECA
from .decalib.datasets import datasets 
from .decalib.utils import util
from .decalib.utils.config import cfg as deca_cfg
from .decalib.utils.rotation_converter import *
import pdb

class DECA_model():
    def __init__(self, device):

        dir_path = os.path.dirname(os.path.realpath(__file__))
        models_path =  os.path.join(dir_path, 'data')
        if not os.path.exists(models_path):
            print('Download DECA model from google drive')
            exit()

        deca_cfg.model.use_tex = False
        self.deca = DECA(config = deca_cfg, device=device)
        self.testdata = datasets.TestData()

    'Batch torch tensor'
    def extract_DECA_params(self, images, visualize = False, save_path = None, index = None, prefix = None):
       
        p_tensor = torch.zeros(images.shape[0], 6).cuda()
        alpha_shp_tensor = torch.zeros(images.shape[0], 100).cuda()
        alpha_exp_tensor = torch.zeros(images.shape[0], 50).cuda()
        angles = torch.zeros(images.shape[0], 3).cuda()
        cam = torch.zeros(images.shape[0], 3).cuda()
        images_deca = torch.zeros(images.shape[0], 3, 224, 224).cuda()
        codedicts = []
        for batch in range(images.shape[0]):   
            
            image_prepro, error_flag = self.testdata.get_image_tensor(images[batch].clone())
            if not error_flag:
                codedict = self.deca.encode(image_prepro.unsqueeze(0).cuda())
                codedicts.append(codedict)
                pose = codedict['pose'][:,:3]
                pose = rad2deg(batch_axis2euler(pose))
                
                p_tensor[batch] = codedict['pose'][0]
                alpha_shp_tensor[batch] = codedict['shape'][0]
                alpha_exp_tensor[batch] = codedict['exp'][0]
                cam[batch] = codedict['cam'][0]
                angles[batch] = pose
                images_deca[batch] = codedict['images'][0]
                if visualize:
                    id_opdict, id_visdict = self.deca.decode(codedict)
                    id_visdict_ = id_visdict.copy()
                    id_visdict = {x:id_visdict[x] for x in ['inputs', 'shape_detail_images']}   
                    name = 'test'
                    cv2.imwrite(os.path.join(save_path, name + '_vis_{}.jpg'.format(prefix)), self.deca.visualize(id_visdict))

                    image = util.tensor2image(id_visdict_['rendered_images'][0])
                    cv2.imwrite(os.path.join(save_path, name + '_render_{}.jpg'.format(prefix)), image)
            else:
                
                angles[batch][0] = -180
                angles[batch][1] = -180
                angles[batch][2] = -180

        return p_tensor, alpha_shp_tensor, alpha_exp_tensor, angles, cam, images_deca, codedicts

    def calculate_shape(self, coefficients, image = None, save_path = None, prefix = None):
        
        landmarks2d, landmarks3d, points = self.deca.decode_stella(coefficients)
        if image is not None and save_path is not None:
            landmarks2d_vis = landmarks2d.clone()
            for i in range(landmarks2d.shape[0]):
                landmarks2d_vis_ = landmarks2d_vis[i].detach().cpu().numpy()
                image_ = util.tensor2image(image[i])
                image_ = util.plot_kpts(image_, landmarks2d_vis_)  #  plot_verts
                save_path_ = os.path.join(save_path, '{}_{:2d}.png'.format(prefix, i))
                cv2.imwrite(save_path_, image_)
        return landmarks2d, landmarks3d, points

    