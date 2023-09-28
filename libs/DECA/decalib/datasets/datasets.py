# -*- coding: utf-8 -*-
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms 
# in the LICENSE file included with this software distribution. 
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at deca@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de

import os, sys
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import cv2
import scipy
from skimage.io import imread, imsave
from skimage.transform import estimate_transform, warp, resize, rescale
from glob import glob
import scipy.io
import torch
from . import detectors
import kornia

def video2sequence(video_path):
    videofolder = video_path.split('.')[0]
    util.check_mkdir(videofolder)
    video_name = video_path.split('/')[-1].split('.')[0]
    vidcap = cv2.VideoCapture(video_path)
    success,image = vidcap.read()
    count = 0
    imagepath_list = []
    while success:
        imagepath = '{}/{}_frame{:04d}.jpg'.format(videofolder, video_name, count)
        cv2.imwrite(imagepath, image)     # save frame as JPEG file
        success,image = vidcap.read()
        count += 1
        imagepath_list.append(imagepath)
    print('video frames are stored in {}'.format(videofolder))
    return imagepath_list

class TestData(Dataset):
    def __init__(self, iscrop=True, crop_size=224, scale=1.25):
        '''
            testpath: folder, imagepath_list, image path, video path
        '''

        self.crop_size = crop_size
        self.scale = scale
        self.iscrop = iscrop
        self.resolution_inp = crop_size
        self.face_detector = detectors.FAN() # CHANGE 
        

    def bbox2point(self, left, right, top, bottom, type='bbox'):
        ''' bbox from detector and landmarks are different
        '''
        if type=='kpt68':
            old_size = (right - left + bottom - top)/2*1.1
            center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 ])
        elif type=='bbox':
            old_size = (right - left + bottom - top)/2
            center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0  + old_size*0.12])
        else:
            raise NotImplementedError
        return old_size, center

    def get_image_tensor(self, image):
        " image: tensor 3x256x256"
        img_tmp = image.clone()
        img_tmp = img_tmp.permute(1,2,0)
        # print(img_tmp.shape)
        bbox, bbox_type = self.face_detector.run(img_tmp)

        if bbox_type != 'error':
            if len(bbox) < 4:
                print('no face detected! run original image')
                left = 0; right = h-1; top=0; bottom=w-1
            else:
                left = bbox[0]; right=bbox[2]
                top = bbox[1]; bottom=bbox[3]
            old_size, center = self.bbox2point(left, right, top, bottom, type=bbox_type)
            size = int(old_size*self.scale)
            src_pts = np.array([[center[0]-size/2, center[1]-size/2], [center[0] - size/2, center[1]+size/2], [center[0]+size/2, center[1]-size/2]])

            DST_PTS = np.array([[0,0], [0,self.resolution_inp - 1], [self.resolution_inp - 1, 0]])
            tform = estimate_transform('similarity', src_pts, DST_PTS)
            theta =  torch.tensor(tform.params, dtype=torch.float32).unsqueeze(0).cuda()

            image_tensor = image.clone()
            image_tensor = image_tensor.unsqueeze(0)
            dst_image = kornia.warp_affine(image_tensor, theta[:,:2,:], dsize=(224, 224))
            dst_image = dst_image.div(255.)
        
            return dst_image.squeeze(0), False

        else:

            return image, True
 

    def get_image(self, image_path):
        
        imagepath = image_path
        image = np.array(imread(imagepath))
        if len(image.shape) == 2:
            image = image[:,:,None].repeat(1,1,3)
        if len(image.shape) == 3 and image.shape[2] > 3:
            image = image[:,:,:3]

        h, w, _ = image.shape
        if self.iscrop:
            # provide kpt as txt file, or mat file (for AFLW2000)
            kpt_matpath = imagepath.replace('.jpg', '.mat').replace('.png', '.mat')
            kpt_txtpath = imagepath.replace('.jpg', '.txt').replace('.png', '.txt')
            if os.path.exists(kpt_matpath):
                kpt = scipy.io.loadmat(kpt_matpath)['pt3d_68'].T        
                left = np.min(kpt[:,0]); right = np.max(kpt[:,0]); 
                top = np.min(kpt[:,1]); bottom = np.max(kpt[:,1])
                old_size, center = self.bbox2point(left, right, top, bottom, type='kpt68')
            elif os.path.exists(kpt_txtpath):
                kpt = np.loadtxt(kptpath)
                left = np.min(kpt[:,0]); right = np.max(kpt[:,0]); 
                top = np.min(kpt[:,1]); bottom = np.max(kpt[:,1])
                old_size, center = self.bbox2point(left, right, top, bottom, type='kpt68')
            else:
                bbox, bbox_type = self.face_detector.run(image)
                if len(bbox) < 4:
                    print('no face detected! run original image')
                    left = 0; right = h-1; top=0; bottom=w-1
                else:
                    left = bbox[0]; right=bbox[2]
                    top = bbox[1]; bottom=bbox[3]
                old_size, center = self.bbox2point(left, right, top, bottom, type=bbox_type)
            size = int(old_size*self.scale)
            src_pts = np.array([[center[0]-size/2, center[1]-size/2], [center[0] - size/2, center[1]+size/2], [center[0]+size/2, center[1]-size/2]])
        else:
            src_pts = np.array([[0, 0], [0, h-1], [w-1, 0]])
        
        DST_PTS = np.array([[0,0], [0,self.resolution_inp - 1], [self.resolution_inp - 1, 0]])
        tform = estimate_transform('similarity', src_pts, DST_PTS)
        
        image = image/255.

        dst_image = warp(image, tform.inverse, output_shape=(self.resolution_inp, self.resolution_inp))
        dst_image = dst_image.transpose(2,0,1)
        return torch.tensor(dst_image).float()