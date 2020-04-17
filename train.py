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
from torch.autograd import Variable
from misc import *
import models.mopnet as net
from models.vgg16 import Vgg16
from myutils import utils
from visualizer import Visualizer
import time
import torch.nn.functional as F
from PIL import Image
import math
import numpy as np
from skimage.measure import compare_psnr as Psnr
import cv2
from collections import OrderedDict
from lib.NLEDN import NLEDN
import itertools

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=False,
  default='my_loader',  help='')
parser.add_argument('--dataroot', required=False,
  default='', help='path to trn dataset')
parser.add_argument('--valDataroot', required=False,
  default='', help='path to val dataset')
parser.add_argument('--pre', type=str, default='', help='prefix of different dataset')
parser.add_argument('--label_file', type=str, default='', help='file for labels')
parser.add_argument('--val_label_file', type=str, default='', help='file for val labels')
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--epoch_count', type=int, default=0, help='input batch size')
parser.add_argument('--valBatchSize', type=int, default=1, help='input batch size')
parser.add_argument('--originalSize', type=int,
  default=532, help='the height / width of the original input image')
parser.add_argument('--imageSize', type=int,
  default=512, help='the height / width of the cropped input image to network')
parser.add_argument('--inputChannelSize', type=int,
  default=3, help='size of the input channels')
parser.add_argument('--outputChannelSize', type=int,
  default=3, help='size of the output channels')
parser.add_argument('--niter', type=int, default=500, help='number of epochs to train for')
parser.add_argument('--lrG', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--annealStart', type=int, default=0, help='annealing learning rate start to')
parser.add_argument('--annealEvery', type=int, default=150, help='epoch to reaching at learning rate of 0')
parser.add_argument('--lambdaIMG', type=float, default=1, help='lambdaIMG')
parser.add_argument('--lambdaEDGE', type=float, default=0.1, help='lambdaIMG')
parser.add_argument('--factor', type=float, default=1.8, help='lambdaIMG')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netE', default="edge/netG_epoch_50.pth", help="path to netE (to continue training)")
parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
parser.add_argument('--exp', default='sample', help='folder to output images and model checkpoints')
parser.add_argument('--display', type=int, default=5, help='interval for displaying train-logs')
parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
parser.add_argument('--name', type=str, default='experiment_name',
                         help='name of the experiment. It decides where to store samples and models')
opt = parser.parse_args()
print(opt)
opt.manualSeed = random.randint(1, 10000)
create_exp_dir(opt.exp)
device = torch.device("cuda:0")

# get dataloader
dataloader = getLoader(opt.dataset,
                       opt.dataroot,
                       opt.originalSize,
                       opt.imageSize,
                       opt.batchSize,
                       opt.workers,
                       mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
                       split='train',
                       shuffle=True,
                       seed=opt.manualSeed,
                       pre=opt.pre,
                       label_file=opt.label_file)
print(len(dataloader))
val_dataloader = getLoader(opt.dataset,
                       opt.valDataroot,
                       opt.originalSize,
                       opt.imageSize,
                       opt.valBatchSize,
                       1,
                       mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
                       split='test',
                       shuffle=False,
                       seed=opt.manualSeed,
                       pre=opt.pre,
                       label_file=opt.val_label_file)


val_iterator = enumerate(val_dataloader, 0)
val_cnt = 0

# get logger
trainLogger = open('%s/train.log' % opt.exp, 'w')

inputChannelSize = opt.inputChannelSize
outputChannelSize= opt.outputChannelSize
lambdaIMG = opt.lambdaIMG
lambdaEDGE = opt.lambdaEDGE

# moire removal framework
netG = net.Single()
if opt.netG != '':
  print("load pre-trained model!!!!!!!!!!!!!!!!!")
  netG.load_state_dict(torch.load(opt.netG))
print(netG)
netG.train()
netG.to(device)

# channel-wise edge predictor
netEdge = net.EdgePredict()
netEdge.load_state_dict(torch.load(opt.netE))
netEdge.to(device)

# Initialize VGG-16
vgg = Vgg16()
utils.init_vgg16('./models/')
vgg.load_state_dict(torch.load(os.path.join('./models/', "vgg16.weight")))
vgg.to(device)

visualizer = Visualizer(opt.display_port, opt.name)

target = torch.FloatTensor(opt.batchSize, outputChannelSize, opt.imageSize, opt.imageSize)
input = torch.FloatTensor(opt.batchSize, inputChannelSize, opt.imageSize, opt.imageSize)
val_target = torch.FloatTensor(1, outputChannelSize, opt.imageSize, opt.imageSize)
val_input = torch.FloatTensor(1, inputChannelSize, opt.imageSize, opt.imageSize)
target, input, val_input, val_target = target.to(device), input.to(device), val_input.to(device), val_target.to(device)

criterionCAE = nn.L1Loss()
criterionCAE.to(device)

# get optimizer
my_lrG = opt.lrG - opt.epoch_count * (opt.lrG/opt.annealEvery)
optimizerG = optim.Adam(itertools.chain(netG.parameters(), netEdge.parameters()), lr = my_lrG, betas = (opt.beta1, 0.999), weight_decay=0.00005)

# NOTE training loop
total_steps = 0
dataset_size = len(dataloader)
my_psnr = 0

# Sobel kernel conv with 2 directions
a = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]], dtype=np.float32)
a = a.reshape(1, 1, 3, 3) # out_c/3, in_c, w, h
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

for epoch in range(opt.epoch_count, opt.niter):
  trainLogger = open('%s/train.log' % opt.exp, 'a+')
  netG.train()
  netEdge.train()
  # update lr
  if epoch >= opt.annealStart:
    adjust_learning_rate(optimizerG, opt.lrG, epoch, None, opt.annealEvery)
    print(50*'-'+'lr'+50*'-')
    print(str(optimizerG.param_groups[0]['lr']))
    print(50 * '-' + 'lr' + 50 * '-')
  epoch_iter = 0
  epoch_start_time = time.time()
  iter_data_time = time.time()
  my_psnr = 0
  ccnt = 0
  for i, data in enumerate(dataloader, 0):
    iter_start_time = time.time()
    if total_steps % 100 == 0:
        t_data = iter_start_time - iter_data_time
    visualizer.reset()
    total_steps += opt.batchSize
    epoch_iter += opt.batchSize

    # get paired data and labels
    input_cpu, target_cpu, label_0_cpu, label_1_cpu, label_2_cpu = data
    batch_size = target_cpu.size(0)
    target_cpu, input_cpu = target_cpu.float().to(device), input_cpu.float().to(device)
    label_0_cpu, label_1_cpu, label_2_cpu = label_0_cpu.float().to(device), label_1_cpu.float().to(device), label_2_cpu.float().to(device)
    labels = [label_0_cpu, label_1_cpu, label_2_cpu]
    target.data.resize_as_(target_cpu).copy_(target_cpu)
    input.data.resize_as_(input_cpu).copy_(input_cpu)

    # get edges of input image and GT
    i_G_x = conv1(input)
    i_G_y = conv2(input)
    input_edge = torch.tanh(torch.abs(i_G_x)+torch.abs(i_G_y))
    t_G_x = conv1(target)
    t_G_y = conv2(target)
    target_edge = torch.tanh(torch.abs(t_G_x)+torch.abs(t_G_y))

    # get predicted edge
    edge1 = netEdge(torch.cat([input, input_edge], 1))
    _, edge = edge1

    # combine multi-sclae, edge and class info
    x_hat1 = netG(input, edge, labels)
    residual, x_hat = x_hat1

    # calculate psnr for current removal results
    for i in range(x_hat.shape[0]):
        ccnt+=1
        ti1 = x_hat[i,:,:,:]
        tt1 = target[i,:,:,:]
        mi1 = utils.my_tensor2im(ti1)
        mt1 = utils.my_tensor2im(tt1)
        g_mi1 = cv2.cvtColor(mi1, cv2.COLOR_BGR2RGB)
        g_mt1 = cv2.cvtColor(mt1, cv2.COLOR_BGR2RGB)
        my_psnr += Psnr(g_mt1, g_mi1)

    # start to update netG and netEdge
    netG.zero_grad()
    netEdge.zero_grad()

    L_img_ = criterionCAE(x_hat, target)
    L_img = lambdaIMG * L_img_
    L_edge = lambdaEDGE * criterionCAE(edge, target_edge)

    # Perceptual Loss 1
    features_content = vgg(target)
    f_xc_c = features_content[1].detach()
    features_y = vgg(x_hat)
    content_loss0 =  opt.factor*lambdaIMG* criterionCAE(features_y[1], f_xc_c)

    # Perceptual Loss 2
    features_content2 = vgg(target)
    f_xc_c2 = features_content2[0].detach()
    features_y2 = vgg(x_hat)
    content_loss1 =  opt.factor*lambdaIMG* criterionCAE(features_y2[0], f_xc_c2)

    # Perceptual Loss 1
    features_content_edge = vgg(target_edge)
    f_xc_c_edge = features_content_edge[1].detach()

    features_y_edge = vgg(edge)
    content_loss0_edge =  opt.factor*lambdaEDGE* criterionCAE(features_y_edge[1], f_xc_c_edge)
    # content_loss0.backward(retain_variables=True)

    # Perceptual Loss 2
    features_content2_edge = vgg(target_edge)
    f_xc_c2_edge = features_content2_edge[0].detach()

    features_y2_edge = vgg(edge)
    content_loss1_edge =  opt.factor*lambdaEDGE* criterionCAE(features_y2_edge[0], f_xc_c2_edge)
    L = L_img + L_edge + content_loss0 + content_loss1 + content_loss0_edge + content_loss1_edge
    L.backward()
    optimizerG.step()

    # show visual results from validation set
    if total_steps % 100 == 0:
        netG.eval()
        netEdge.eval()
        val_data = val_iterator.next()
        val_cnt += 1
        if val_cnt == len(val_dataloader):
            val_cnt = 0
            val_iterator = enumerate(val_dataloader)
        _, l = val_data

        val_input_cpu = l[0]
        val_target_cpu = l[1]
        val_label_0_cpu = l[2]
        val_label_1_cpu = l[3]
        val_label_2_cpu = l[4]
        val_target_cpu, val_input_cpu = val_target_cpu.float().to(device), val_input_cpu.float().to(device)
        val_label_0_cpu, val_label_1_cpu, val_label_2_cpu = val_label_0_cpu.float().to(device), val_label_1_cpu.float().to(device), val_label_2_cpu.float().to(device)

        # get paired data and labels
        val_target.data.resize_as_(val_target_cpu).copy_(val_target_cpu)
        val_input.data.resize_as_(val_input_cpu).copy_(val_input_cpu)
        val_labels = [val_label_0_cpu, val_label_1_cpu, val_label_2_cpu]

        val_i_G_x = conv1(val_input)
        val_i_G_y = conv2(val_input)
        val_input_edge = torch.tanh(torch.abs(val_i_G_x) + torch.abs(val_i_G_y))
        # print(input_edge)
        val_t_G_x = conv1(val_target)
        val_t_G_y = conv2(val_target)
        val_target_edge = torch.tanh(torch.abs(val_t_G_x) + torch.abs(val_t_G_y))

        val_edge1 = netEdge(torch.cat([val_input, val_input_edge], 1))
        _, val_edge = val_edge1

        val_x_hat1 = netG(val_input, val_edge, val_labels)
        val_residual, val_x_hat = val_x_hat1

        current_visuals = OrderedDict([('val_input', val_input), ('val_output', val_x_hat), ('val_GT', val_target),
                                       ('val_pred_edge', val_edge), ('val_GT_edge', val_target_edge)])

        losses = OrderedDict([('L_img', L_img.detach().cpu().float().numpy()),
                              ('content_loss0', content_loss0.detach().cpu().float().numpy()),
                              ('content_loss1', content_loss1.detach().cpu().float().numpy()),
                              ('content_loss0_edge', content_loss0_edge.detach().cpu().float().numpy()),
                              ('content_loss1_edge', content_loss1_edge.detach().cpu().float().numpy()),
                              ('L_img_edge', L_edge.detach().cpu().float().numpy()),
                              ('my_psnr', my_psnr / (ccnt))])
        t = (time.time() - iter_start_time) / opt.batchSize
        trainLogger.write(visualizer.print_current_losses(epoch, epoch_iter, losses, t, t_data) + '\n')
        visualizer.print_current_losses(epoch, epoch_iter, losses, t, t_data)
        r = float(epoch_iter) / (dataset_size*opt.batchSize)
        if opt.display_port!=-1:
            visualizer.display_current_results(current_visuals, epoch, False)
            visualizer.plot_current_losses(epoch, r, opt, losses)

        netG.train()
        netEdge.train()

  # save model and test performance in validation set
  if epoch % 1 == 0:

        print('hit')
        my_file = open("./" + opt.name + "_" + "evaluation.txt", 'a+')
        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.exp, epoch))
        torch.save(netEdge.state_dict(), '%s/netEdge_epoch_%d.pth' % (opt.exp, epoch))
        vcnt = 0
        vpsnr = 0
        netG.eval()
        netEdge.eval()
        for i, data in enumerate(val_dataloader, 0):
          input_cpu, target_cpu, label_0_cpu, label_1_cpu, label_2_cpu = data
          batch_size = target_cpu.size(0)

          target_cpu, input_cpu = target_cpu.float().to(device), input_cpu.float().to(device)
          label_0_cpu, label_1_cpu, label_2_cpu = label_0_cpu.float().to(device), label_1_cpu.float().to(device), label_2_cpu.float().to(device)

          labels = [label_0_cpu, label_1_cpu, label_2_cpu]
          # get paired data
          target.data.resize_as_(target_cpu).copy_(target_cpu)
          input.data.resize_as_(input_cpu).copy_(input_cpu)

          i_G_x = conv1(input)
          i_G_y = conv2(input)
          input_edge = torch.tanh(torch.abs(i_G_x) + torch.abs(i_G_y))
          # print(input_edge)
          t_G_x = conv1(target)
          t_G_y = conv2(target)
          target_edge = torch.tanh(torch.abs(t_G_x) + torch.abs(t_G_y))

          edge1 = netEdge(torch.cat([input, input_edge], 1))
          _, edge = edge1

          x_hat1 = netG(input, edge, labels)
          residual, x_hat = x_hat1

          for i in range(x_hat.shape[0]):
              vcnt += 1
              ti1 = x_hat[i, :, :, :]
              tt1 = target[i, :, :, :]
              mi1 = utils.my_tensor2im(ti1)
              mt1 = utils.my_tensor2im(tt1)
              g_mi1 = cv2.cvtColor(mi1, cv2.COLOR_BGR2RGB)
              g_mt1 = cv2.cvtColor(mt1, cv2.COLOR_BGR2RGB)
              vpsnr += Psnr(g_mt1, g_mi1)

        my_file.write(str(epoch) + str('-') + str(total_steps) + '\n')
        my_file.write(str(float(vpsnr) / vcnt) + '\n')
        print("val:")
        print(float(vpsnr) / vcnt)
        trainLogger.close()
        netG.train()


my_file.close()
trainLogger.close()