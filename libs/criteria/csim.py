import torch
import torch.nn as nn
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, ReLU, Sigmoid, Dropout2d, Dropout, AvgPool2d, MaxPool2d, AdaptiveAvgPool2d, Sequential, Module, Parameter
import torch.nn.functional as F
import numpy as np
from collections import namedtuple
import math
import pathlib
from skimage import transform

from torchvision import utils as torch_utils


class ID_LossWrapper(nn.Module):
    def __init__(self, csm_model, device = 'cuda'):
        super(ID_LossWrapper, self).__init__()

        model_path = csm_model

        self.model = InsightFaceWrapper(model_path)
        self.device = device
        if self.device == 'cuda':
            self.model.cuda().eval()
        self.criterion = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, real_imgs, fake_imgs, real_poses, scale = False, shifted_poses = None, prefix = ''):
        b, c, h, w = real_imgs.shape
        
        affine_matrices = find_affine_transformation(real_poses, h, w, scale = scale)
        if shifted_poses is not None:
            affine_matrices_sh = find_affine_transformation(shifted_poses, h, w, scale = scale)

        with torch.no_grad():
            real_embeds, real_warped_image = self.model(affine_matrices, real_imgs, '{}_real'.format(prefix))
        if shifted_poses is not None: 
            fake_embeds, fake_warped_image = self.model(affine_matrices_sh, fake_imgs,'{}_fake'.format(prefix))
        else:
            fake_embeds, fake_warped_image = self.model(affine_matrices, fake_imgs, '{}_fake'.format(prefix))
        
        # Calc cosine similarity
        # cosine_sim = (real_embeds * fake_embeds).sum(1) / (real_embeds**2).sum(1)**0.5 / (fake_embeds**2).sum(1)**0.5 

        cosine_sim = self.criterion(fake_embeds, real_embeds)
        loss = 1 - cosine_sim
        loss = torch.mean(loss)
        return loss, real_warped_image, fake_warped_image

    def __repr__(self):
        return '(cdist): LossWrapper()'

def find_affine_transformation(poses, h_img=256, w_img=256, scale = False):
    """
    Function return matrix of affine transformation to use in torch.nn.functional.affine_grid

    Input:
    facial_keypoints: np.array of size (5, 2) - coordinates of key facial points in pixel coords
                      right eye, left eye, nose, right mouse, left mouse
    h_img: int, height of input image
    w_img: int, width of input image
    returns: np.array of size (2, 3) - affine matrix
    """
    # poses = (poses.detach() + 1) / 2

    right_eye = list(range(36 , 42))
    left_eye = list(range(42, 48))
    nose = [30]
    right_mouth = [48]
    left_mouth = [54]
    
    if torch.is_tensor(poses):
        keypoints = poses.cpu().numpy().reshape(-1, 68, 2)
    else:
        keypoints = poses.reshape(-1, 68, 2)

    facial_keypoints = np.concatenate([
        keypoints[:, right_eye].astype('float32').mean(1, keepdims=True), # right eye
        keypoints[:, left_eye].astype('float32').mean(1, keepdims=True), # left eye
        keypoints[:, nose].astype('float32'), # nose
        keypoints[:, right_mouth].astype('float32'), # right mouth
        keypoints[:, left_mouth].astype('float32'), # left mouth
    ], 1)
    
    if scale:
        facial_keypoints[:, 0, 0] -= 20 # right eye (left opws to vlepw sthn eikona )
        facial_keypoints[:, 1, 0] += 20 # left eye (right opws to vlepw sthn eikona )
        facial_keypoints[:, 3, 1] += 20
        facial_keypoints[:, 4, 1] += 20
    

    #affine_matrix = torch.from_numpy(find_affine_transformation(facial_keypoints, 
    #    h_img=h_img, w_img=w_img)).float()

    h_grid = 112
    w_grid = 112

    src = np.array([
        [35.343697, 51.6963] ,
        [76.453766, 51.5014],
        [56.029396, 71.7366],
        [39.14085 , 92.3655],
        [73.18488 , 92.2041]], dtype=np.float32)

    

    affine_matrices = []

    for facial_keypoints_i in facial_keypoints:
        tform = transform.estimate_transform('similarity', src, facial_keypoints_i)
        affine_matrix = tform.params[:2, :]

        affine_matrices.append(affine_matrix)

    affine_matrices = np.stack(affine_matrices, axis=0)
   
    # do transformation for grid in [-1, 1]
    affine_matrices[:, 0, 0] = affine_matrices[:, 0, 0]*w_grid/w_img
    affine_matrices[:, 0, 1] = affine_matrices[:, 0, 1]*h_grid/w_img
    affine_matrices[:, 0, 2] = (affine_matrices[:, 0, 2])*2/w_img + affine_matrices[:, 0, 1] + affine_matrices[:, 0, 0] - 1
    affine_matrices[:, 1, 0] = affine_matrices[:, 1, 0]*w_grid/h_img
    affine_matrices[:, 1, 1] = affine_matrices[:, 1, 1]*h_grid/h_img
    affine_matrices[:, 1, 2] = (affine_matrices[:, 1, 2])*2/h_img + affine_matrices[:, 1, 0] + affine_matrices[:, 1, 1] - 1
    
    affine_matrices = torch.from_numpy(affine_matrices).float().cuda()
    
    return affine_matrices


class InsightFaceWrapper(nn.Module):
    """
    Wrapper of InsightFaceModel
    """
    def __init__(self, path_weights, num_layers=50, drop_ratio=0.6, mode='ir_se'):
        super(InsightFaceWrapper, self).__init__()
        self.model = Backbone(num_layers, drop_ratio, mode)
        self.model.load_state_dict(torch.load(path_weights))
        self.model.train(False)

    def forward(self, affine_matrix, image, prefix):
        batch_size = image.shape[0]
        grid = nn.functional.affine_grid(affine_matrix, torch.Size((batch_size, 3, 112, 112)))
        
        warped_image = nn.functional.grid_sample(image, grid)
        
        return self.model(warped_image), warped_image

##################################  Original Arcface Model #############################################################

class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

def l2_norm(input,axis=1):
    norm = torch.norm(input,2,axis,True)
    output = torch.div(input, norm)
    return output

class SEModule(Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = AdaptiveAvgPool2d(1)
        self.fc1 = Conv2d(
            channels, channels // reduction, kernel_size=1, padding=0 ,bias=False)
        self.relu = ReLU(inplace=True)
        self.fc2 = Conv2d(
            channels // reduction, channels, kernel_size=1, padding=0 ,bias=False)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x

class bottleneck_IR(Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride ,bias=False), BatchNorm2d(depth))
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1 ,bias=False), PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1 ,bias=False), BatchNorm2d(depth))

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut

class bottleneck_IR_SE(Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR_SE, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride ,bias=False), 
                BatchNorm2d(depth))
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3,3), (1,1),1 ,bias=False),
            PReLU(depth),
            Conv2d(depth, depth, (3,3), stride, 1 ,bias=False),
            BatchNorm2d(depth),
            SEModule(depth,16)
            )
    def forward(self,x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut

class Bottleneck(namedtuple('Block', ['in_channel', 'depth', 'stride'])):
    '''A named tuple describing a ResNet block.'''
    
def get_block(in_channel, depth, num_units, stride = 2):
  return [Bottleneck(in_channel, depth, stride)] + [Bottleneck(depth, depth, 1) for i in range(num_units-1)]

def get_blocks(num_layers):
    if num_layers == 50:
        blocks = [
            get_block(in_channel=64, depth=64, num_units = 3),
            get_block(in_channel=64, depth=128, num_units=4),
            get_block(in_channel=128, depth=256, num_units=14),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 100:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=13),
            get_block(in_channel=128, depth=256, num_units=30),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 152:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=8),
            get_block(in_channel=128, depth=256, num_units=36),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    return blocks

class Backbone(Module):
    def __init__(self, num_layers, drop_ratio, mode='ir'):
        super(Backbone, self).__init__()
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1 ,bias=False), 
                                      BatchNorm2d(64), 
                                      PReLU(64))
        self.output_layer = Sequential(BatchNorm2d(512), 
                                       Dropout(drop_ratio),
                                       Flatten(),
                                       Linear(512 * 7 * 7, 512),
                                       BatchNorm1d(512))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    unit_module(bottleneck.in_channel,
                                bottleneck.depth,
                                bottleneck.stride))
        self.body = Sequential(*modules)
    
    def forward(self,x):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_layer(x)
        return l2_norm(x)
        # return l2_norm(torch.nn.ReLU(inplace=True)(x))

##################################  MobileFaceNet #############################################################
    
class Conv_block(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Conv_block, self).__init__()
        self.conv = Conv2d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding, bias=False)
        self.bn = BatchNorm2d(out_c)
        self.prelu = PReLU(out_c)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.prelu(x)
        return x

class Linear_block(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Linear_block, self).__init__()
        self.conv = Conv2d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding, bias=False)
        self.bn = BatchNorm2d(out_c)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class Depth_Wise(Module):
     def __init__(self, in_c, out_c, residual = False, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1):
        super(Depth_Wise, self).__init__()
        self.conv = Conv_block(in_c, out_c=groups, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.conv_dw = Conv_block(groups, groups, groups=groups, kernel=kernel, padding=padding, stride=stride)
        self.project = Linear_block(groups, out_c, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.residual = residual
     def forward(self, x):
        if self.residual:
            short_cut = x
        x = self.conv(x)
        x = self.conv_dw(x)
        x = self.project(x)
        if self.residual:
            output = short_cut + x
        else:
            output = x
        return output

class Residual(Module):
    def __init__(self, c, num_block, groups, kernel=(3, 3), stride=(1, 1), padding=(1, 1)):
        super(Residual, self).__init__()
        modules = []
        for _ in range(num_block):
            modules.append(Depth_Wise(c, c, residual=True, kernel=kernel, padding=padding, stride=stride, groups=groups))
        self.model = Sequential(*modules)
    def forward(self, x):
        return self.model(x)

class MobileFaceNet(Module):
    def __init__(self, embedding_size):
        super(MobileFaceNet, self).__init__()
        self.conv1 = Conv_block(3, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv2_dw = Conv_block(64, 64, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=64)
        self.conv_23 = Depth_Wise(64, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=128)
        self.conv_3 = Residual(64, num_block=4, groups=128, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_34 = Depth_Wise(64, 128, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=256)
        self.conv_4 = Residual(128, num_block=6, groups=256, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_45 = Depth_Wise(128, 128, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=512)
        self.conv_5 = Residual(128, num_block=2, groups=256, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_6_sep = Conv_block(128, 512, kernel=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_6_dw = Linear_block(512, 512, groups=512, kernel=(7,7), stride=(1, 1), padding=(0, 0))
        self.conv_6_flatten = Flatten()
        self.linear = Linear(512, embedding_size, bias=False)
        self.bn = BatchNorm1d(embedding_size)
    
    def forward(self, x):
        out = self.conv1(x)

        out = self.conv2_dw(out)

        out = self.conv_23(out)

        out = self.conv_3(out)
        
        out = self.conv_34(out)

        out = self.conv_4(out)

        out = self.conv_45(out)

        out = self.conv_5(out)

        out = self.conv_6_sep(out)

        out = self.conv_6_dw(out)

        out = self.conv_6_flatten(out)

        out = self.linear(out)

        out = self.bn(out)
        return l2_norm(out)

##################################  Arcface head #############################################################

class Arcface(Module):
    # implementation of additive margin softmax loss in https://arxiv.org/abs/1801.05599    
    def __init__(self, embedding_size=512, classnum=51332,  s=64., m=0.5):
        super(Arcface, self).__init__()
        self.classnum = classnum
        self.kernel = Parameter(torch.Tensor(embedding_size,classnum))
        # initial kernel
        self.kernel.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        self.m = m # the margin value, default is 0.5
        self.s = s # scalar value default is 64, see normface https://arxiv.org/abs/1704.06369
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.mm = self.sin_m * m  # issue 1
        self.threshold = math.cos(math.pi - m)
    def forward(self, embbedings, label):
        # weights norm
        nB = len(embbedings)
        kernel_norm = l2_norm(self.kernel,axis=0)
        # cos(theta+m)
        cos_theta = torch.mm(embbedings,kernel_norm)
#         output = torch.mm(embbedings,kernel_norm)
        cos_theta = cos_theta.clamp(-1,1) # for numerical stability
        cos_theta_2 = torch.pow(cos_theta, 2)
        sin_theta_2 = 1 - cos_theta_2
        sin_theta = torch.sqrt(sin_theta_2)
        cos_theta_m = (cos_theta * self.cos_m - sin_theta * self.sin_m)
        # this condition controls the theta+m should in range [0, pi]
        #      0<=theta+m<=pi
        #     -m<=theta<=pi-m
        cond_v = cos_theta - self.threshold
        cond_mask = cond_v <= 0
        keep_val = (cos_theta - self.mm) # when theta not in [0,pi], use cosface instead
        cos_theta_m[cond_mask] = keep_val[cond_mask]
        output = cos_theta * 1.0 # a little bit hacky way to prevent in_place operation on cos_theta
        idx_ = torch.arange(0, nB, dtype=torch.long)
        output[idx_, label] = cos_theta_m[idx_, label]
        output *= self.s # scale up in order to make softmax work, first introduced in normface
        return output

##################################  Cosface head #############################################################    
    
class Am_softmax(Module):
    # implementation of additive margin softmax loss in https://arxiv.org/abs/1801.05599    
    def __init__(self,embedding_size=512,classnum=51332):
        super(Am_softmax, self).__init__()
        self.classnum = classnum
        self.kernel = Parameter(torch.Tensor(embedding_size,classnum))
        # initial kernel
        self.kernel.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        self.m = 0.35 # additive margin recommended by the paper
        self.s = 30. # see normface https://arxiv.org/abs/1704.06369
    def forward(self,embbedings,label):
        kernel_norm = l2_norm(self.kernel,axis=0)
        cos_theta = torch.mm(embbedings,kernel_norm)
        cos_theta = cos_theta.clamp(-1,1) # for numerical stability
        phi = cos_theta - self.m
        label = label.view(-1,1) #size=(B,1)
        index = cos_theta.data * 0.0 #size=(B,Classnum)
        index.scatter_(1,label.data.view(-1,1),1)
        index = index.byte()
        output = cos_theta * 1.0
        output[index] = phi[index] #only change the correct predicted output
        output *= self.s # scale up in order to make softmax work, first introduced in normface
        return output