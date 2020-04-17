from __future__ import print_function
import argparse
import os
import sys
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
cudnn.fastest = True
import torch.optim as optim
import torchvision.utils as vutils
from torch.autograd import Variable

from misc import *
import models.mopnet as net
from models.vgg16 import Vgg16
from myutils import utils
from visualizer import Visualizer
import time
import torch.nn.functional as F
import scipy.stats as st
import datetime

from PIL import Image
import math
import numpy as np
import cv2
from collections import OrderedDict

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=False,
  default='my_loader',  help='')
parser.add_argument('--dataroot', required=False,
  default='', help='path to trn dataset')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netE', default="EdgePredictWeight/netG_epoch_33.pth", help="path to netE (to continue training)")
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--originalSize', type=int,
  default=532, help='the height / width of the original input image')
parser.add_argument('--imageSize', type=int,
  default=512, help='the height / width of the cropped input image to network')
parser.add_argument('--pre', type=str, default='', help='prefix of different dataset')
parser.add_argument('--image_path', type=str, default='', help='path to save the generated vali image')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--inputChannelSize', type=int,
  default=3, help='size of the input channels')
parser.add_argument('--outputChannelSize', type=int,
  default=3, help='size of the output channels')
parser.add_argument('--record', type=str, default='default.txt', help='prefix of different dataset')
parser.add_argument('--number', type=int, default=10)
parser.add_argument('--write', type=int, default=0, help='if write the results?')
opt = parser.parse_args()
print(opt)

path_class_color = "./classifier/color_epoch_95.pth"
path_class_geo = "./classifier/geo_epoch_95.pth"

device = torch.device("cuda:0")

opt.manualSeed = random.randint(1, 10000)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
torch.cuda.manual_seed_all(opt.manualSeed)
print("Random Seed: ", opt.manualSeed)

val_dataloader = getLoader(opt.dataset,
                       opt.dataroot,
                       opt.originalSize,
                       opt.imageSize,
                       opt.batchSize,
                       opt.workers,
                       mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
                       split='test',
                       shuffle=False,
                       seed=opt.manualSeed,
                       pre=opt.pre)

inputChannelSize = opt.inputChannelSize
outputChannelSize= opt.outputChannelSize

netG=net.Single()
netG.load_state_dict(torch.load(opt.netG))
netG.eval()
netG.to(device)
netEdge = net.EdgePredict()
netEdge.load_state_dict(torch.load(opt.netE))
netEdge.eval()
netEdge.to(device)
print(netG)

target = torch.FloatTensor(opt.batchSize, outputChannelSize, opt.imageSize, opt.imageSize)
input = torch.FloatTensor(opt.batchSize, inputChannelSize, opt.imageSize, opt.imageSize)
target, input = target.to(device), input.to(device)

# Classifiers
net_label_color=net.vgg19ca()
net_label_color.load_state_dict(torch.load(path_class_color))
net_label_color=net_label_color.to(device)

net_label_geo = net.vgg19ca_2()
net_label_geo.load_state_dict(torch.load(path_class_geo))
net_label_geo=net_label_geo.to(device)

vcnt = 0

# Sobel kernel Conv
a = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]], dtype=np.float32)
a = a.reshape(1, 1, 3, 3)
a = np.repeat(a, 3, axis=0)
conv1=nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False, groups=3)
conv1.weight.data.copy_(torch.from_numpy(a))
conv1.weight.requires_grad = False
conv1.cuda()

b = np.array([[-1, -2, -1],[0, 0, 0],[1, 2, 1]], dtype=np.float32)
b = b.reshape(1, 1, 3, 3)
b = np.repeat(b, 3, axis=0)
conv2=nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False, groups=3)
conv2.weight.data.copy_(torch.from_numpy(b))
conv2.weight.requires_grad = False
conv2.cuda()

def gauss_kernel(kernlen=21, nsig=3, channels=1):
    interval = (2 * nsig + 1.) / (kernlen)
    x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kernlen + 1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw / kernel_raw.sum()
    out_filter = np.array(kernel, dtype=np.float32)
    out_filter = out_filter.reshape((kernlen, kernlen, 1, 1))
    out_filter = np.repeat(out_filter, channels, axis=2)
    return out_filter

# Gaussian blur
g_kernel = gauss_kernel(3, 5, 1).transpose((3, 2, 1, 0))
gauss_conv = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=3/2, bias=False)
gauss_conv.weight.data.copy_(torch.from_numpy(g_kernel))
gauss_conv.weight.requires_grad = False
gauss_conv.cuda()


for i, data in enumerate(val_dataloader, 0):

    input_cpu, target_cpu = data
    batch_size = target_cpu.size(0)

    # get paired data
    target_cpu, input_cpu = target_cpu.float().to(device), input_cpu.float().to(device)
    target.data.resize_as_(target_cpu).copy_(target_cpu)
    input.data.resize_as_(input_cpu).copy_(input_cpu)

    i_G_x = conv1(input)
    i_G_y = conv2(input)
    iG = torch.tanh(torch.abs(i_G_x)+torch.abs(i_G_y))

    res1 = gauss_conv(input[:, 0, :, :].unsqueeze(1))
    res2 = gauss_conv(input[:, 1, :, :].unsqueeze(1))
    res3 = gauss_conv(input[:, 2, :, :].unsqueeze(1))
    input_ = torch.cat((res1, res2, res3), dim=1)

    # predict color labels
    _, label_color = torch.max(net_label_color(input), 1)
    label_curve, label_thick = net_label_geo(iG)
    _, label_curve = torch.max(label_curve, 1)
    _, label_thick = torch.max(label_thick, 1)
    label_curve = label_curve.float()
    label_color = label_color.float()
    label_thick = label_thick.float()
    labels = [label_curve, label_color, label_thick]

    # Get input edges
    i_G_x_ = conv1(input)
    i_G_y_ = conv2(input)
    input_edge = torch.tanh(torch.abs(i_G_x_)+torch.abs(i_G_y_))

    # Get predicted edges
    edge1 = netEdge(torch.cat([input, input_edge], 1))
    _, edge = edge1

    # Moire removal
    x_hat1 = netG(input, edge, labels)
    residual, x_hat = x_hat1

    # Save results
    for j in range(x_hat.shape[0]):
        vcnt += 1
        b, c, w, h = x_hat.shape
        ti1 = x_hat[j, :,:,: ]
        tt1 = target[j, :,:,: ]
        ori = input[j, :, :, :]
        mi1 = cv2.cvtColor(utils.my_tensor2im(ti1), cv2.COLOR_BGR2RGB)
        mt1 = cv2.cvtColor(utils.my_tensor2im(tt1), cv2.COLOR_BGR2RGB)
        ori = cv2.cvtColor(utils.my_tensor2im(ori), cv2.COLOR_BGR2RGB)

        if opt.write==1:
            cv2.imwrite(opt.image_path + os.sep + 'd'+os.sep+'d'+str(i)+'_'+str(j) +'_.png', mi1)
            cv2.imwrite(opt.image_path + os.sep+ 'o'+os.sep + 'o' + str(i)+'_'+str(j) + "_.png", ori)
            cv2.imwrite(opt.image_path + os.sep + 'g' + os.sep + 'g' + str(i) + '_' + str(j) + "_.png", mt1)

    print(50*'-')
    print(vcnt)
    print(50*'-')

