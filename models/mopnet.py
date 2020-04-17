import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
import torchvision.models as models
from torch.autograd import Variable
from NEDB import NEDB
from RNEDB import RNEDB
import numpy as np
import cv2
import os
import sys

class SEBlock(nn.Module):
    def __init__(self, input_dim, reduction):
        super(SEBlock, self).__init__()
        mid = int(input_dim / reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(input_dim, reduction),
            nn.ReLU(inplace=True),
            nn.Linear(reduction, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckBlock, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)

class BottleneckBlock1(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckBlock1, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=5, stride=1,
                               padding=2, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)

class BottleneckBlock2(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckBlock2, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=7, stride=1,
                               padding=3, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)

class TransitionBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return F.upsample_nearest(out, scale_factor=2)

class TransitionBlock1(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlock1, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return F.avg_pool2d(out, 2)

class TransitionBlock3(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlock3, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return out

class vgg19ca_2(nn.Module):
    def __init__(self):
        super(vgg19ca_2, self).__init__()
        moire_class = models.vgg19_bn(pretrained=True)
        self.feature = nn.Sequential(moire_class.features[0])

        for i in range(1,3):
            self.feature.add_module(str(i),moire_class.features[i])

        self.conv16=nn.Conv2d(64, 24, kernel_size=3,stride=1,padding=1)  # 1mm
        self.dense_classifier=nn.Linear(31104, 512)
        self.dense_classifier1=nn.Linear(512, 2)

        self.conv16_=nn.Conv2d(64, 24, kernel_size=3,stride=1,padding=1)  # 1mm
        self.dense_classifier_=nn.Linear(31104, 512)
        self.dense_classifier1_=nn.Linear(512, 2)
    def forward(self, x):
        feature1=self.feature(x)
        feature=self.conv16(feature1)
        out = F.relu(feature, inplace=True)
        out = F.avg_pool2d(out, kernel_size=7).view(out.size(0), -1)
        out = F.relu(self.dense_classifier(out))
        out = (self.dense_classifier1(out))
        feature_=self.conv16_(feature1)
        out_ = F.relu(feature_, inplace=True)
        out_ = F.avg_pool2d(out_, kernel_size=7).view(out.size(0), -1)
        out_ = F.relu(self.dense_classifier_(out_))
        out_ = (self.dense_classifier1_(out_))
        return out, out_

class vgg19ca(nn.Module):
    def __init__(self):
        super(vgg19ca, self).__init__()
        moire_class = models.vgg19_bn(pretrained=True)
        self.feature = nn.Sequential(moire_class.features[0])
        for i in range(1,3):
            self.feature.add_module(str(i),moire_class.features[i])
        self.conv16=nn.Conv2d(64, 24, kernel_size=3,stride=1,padding=1)  # 1mm
        self.dense_classifier=nn.Linear(31104, 512)
        self.dense_classifier1=nn.Linear(512, 2)
    def forward(self, x):
        feature=self.feature(x)
        feature=self.conv16(feature)
        out = F.relu(feature, inplace=True)
        out = F.avg_pool2d(out, kernel_size=7).view(out.size(0), -1)
        out = F.relu(self.dense_classifier(out))
        out = (self.dense_classifier1(out))
        return out

class EdgePredict(nn.Module):
    def __init__(self):
        super(EdgePredict, self).__init__()
        self.dense2=Dense_base_down2_()

    def forward(self, x):
        x3=self.dense2(x)
        return torch.Tensor([1,1]), x3

class Dense_base_down2_(nn.Module):
    def __init__(self):
        super(Dense_base_down2_, self).__init__()

        self.conv1 = nn.Conv2d(3 + 3, 64, 3, 1, 1)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.se = RNEDB(block_num=4, inter_channel=32, channel=64, grid=[8, 8])

        self.dense_block1=BottleneckBlock2(64,64)
        self.trans_block1=TransitionBlock1(128,64)
        # self.se0 = SEBlock(8, 4)
        self.se0 = RNEDB(block_num=4, inter_channel=32, channel=64, grid=[4, 4])
        ############# Block2-down 32-32  ##############
        self.dense_block2=BottleneckBlock2(64,64)
        self.trans_block2=TransitionBlock1(128,64)
        #self.se1 = SEBlock(16, 8)
        self.se1 = RNEDB(block_num=4, inter_channel=32, channel=64, grid=[2, 2])
        ############# Block3-down  16-16 ##############
        self.dense_block3=BottleneckBlock2(64,64)
        self.trans_block3=TransitionBlock1(128,64)
        # self.se2 = SEBlock(16, 8)
        self.se2 = NEDB(block_num=4, inter_channel=32, channel=64)
        ############# Block4-up  8-8  ##############
        self.dense_block4=BottleneckBlock2(64,64)
        self.trans_block4=TransitionBlock(128,64)
       #  self.se3 = SEBlock(32, 16)
        self.se3 = RNEDB(block_num=4, inter_channel=64, channel=128, grid=[2, 2])
        ############# Block5-up  16-16 ##############
        self.dense_block5=BottleneckBlock2(128,64)
        self.trans_block5=TransitionBlock(192,64)
        #self.se4 = SEBlock(16, 8)
        self.se4 = RNEDB(block_num=4, inter_channel=64, channel=128, grid=[4, 4])
        self.dense_block6=BottleneckBlock2(128,64)
        self.trans_block6=TransitionBlock(192,64)


        # self.conv_refin=nn.Conv2d(11,20,3,1,1)
        self.tanh=nn.Tanh()


        self.conv11 = nn.Conv2d(64, 64, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv21 = nn.Conv2d(64, 64, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv31 = nn.Conv2d(64, 64, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv41 = nn.Conv2d(128, 64, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv51 = nn.Conv2d(128, 64, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv61 = nn.Conv2d(64, 64, kernel_size=1,stride=1,padding=0)  # 1mm

        #self.refine3= nn.Conv2d(20+4, 3, kernel_size=3,stride=1,padding=1)
        # self.refine3= nn.Conv2d(20+4, 3, kernel_size=7,stride=1,padding=3)

        self.upsample = F.upsample_nearest

        self.relu=nn.LeakyReLU(0.2, inplace=True)


        #self.batchnorm20=nn.BatchNorm2d(20)
        #self.batchnorm1=nn.BatchNorm2d(1)

        self.fusion = nn.Sequential(
            nn.Conv2d(64 * 8, 64, 1, 1, 0),
            nn.Conv2d(64, 64, 3, 1, 1),
        )

        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 3, 3, 1, 1),
            nn.Tanh(),
        )


    def forward(self, x):
        ## 256x256
        f0 = self.conv1(x)
        f1 = self.conv2(f0)
        #
        # print("f1")
        # print(f1.shape)
        f1_ = self.se(f1)
        x1=self.dense_block1(f1_)
        # print("x1_")
        # print(x1_.shape)
        x1=self.trans_block1(x1)

        x1 = self.se0(x1)
        ###  32x32
        x2=(self.dense_block2(x1))
        x2=self.trans_block2(x2)


        x2 = self.se1(x2)
        x3=(self.dense_block3(x2))
        x3=self.trans_block3(x3)


        x3 = self.se2(x3)
        ## Classifier  ##
        x4=(self.dense_block4(x3))
        x4=self.trans_block4(x4)

        # x4=x4+x2
        x4=torch.cat([x4,x2],1)
        x4 = self.se3(x4)
        x5=(self.dense_block5(x4))
        x5=self.trans_block5(x5)

        # x5=x5+x1
        x5=torch.cat([x5,x1],1)
        x5 = self.se4(x5)
        x6=(self.dense_block6(x5))
        x6=(self.trans_block6(x6))

        x61 = self.relu((self.conv61(x6)))

        feature = x61 + f0

        return self.final_conv(feature)

class Single(nn.Module):
    def __init__(self):
        super(Single, self).__init__()
        self.dense2=Dense_base_down2()

    def forward(self, x, edge, labels):
        x3=self.dense2(x, edge, labels)
        return torch.Tensor([1,1]), x3

class Dense_base_down2(nn.Module):
    def __init__(self):
        super(Dense_base_down2, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1)
        self.conv2 = nn.Conv2d(128, 64, 3, 1, 1)
        self.conv1_e = nn.Conv2d(3, 64, 3, 1, 1)
        self.se_tmp = SEBlock(128, 64)
        # self.se = RNEDB(block_num=4, inter_channel=32, channel=64+3, grid=[8, 8])

        self.dense_block1=BottleneckBlock2(64,64)
        self.trans_block1=TransitionBlock1(128,64)
        #self.se0 = SEBlock(67,32)
        #self.se0 = RNEDB(block_num=4, inter_channel=32, channel=64, grid=[4, 4])
        ############# Block2-down 32-32  ##############
        self.dense_block2=BottleneckBlock2(64,64)
        self.trans_block2=TransitionBlock1(128,64)
        #self.se1 = SEBlock(16, 8)
        #self.se1 = RNEDB(block_num=4, inter_channel=32, channel=64, grid=[2, 2])
        ############# Block3-down  16-16 ##############
        self.dense_block3=BottleneckBlock2(64,64)
        self.trans_block3=TransitionBlock1(128,64)
        # self.se2 = SEBlock(16, 8)
        #self.se2 = NEDB(block_num=4, inter_channel=32, channel=64)
        ############# Block4-up  8-8  ##############
        self.dense_block4=BottleneckBlock2(64,64)
        self.trans_block4=TransitionBlock(128,64)
       #  self.se3 = SEBlock(32, 16)
        #self.se3 = RNEDB(block_num=4, inter_channel=64, channel=128, grid=[2, 2])
        ############# Block5-up  16-16 ##############
        self.dense_block5=BottleneckBlock2(128,64)
        self.trans_block5=TransitionBlock(192,64)
        #self.se4 = SEBlock(16, 8)
       # self.se4 = RNEDB(block_num=4, inter_channel=64, channel=128, grid=[4, 4])
        self.dense_block6=BottleneckBlock2(128,64)
        self.trans_block6=TransitionBlock(192,64)
        self.se5 = SEBlock(448, 220)
        self.se6 = SEBlock(64+3, 32)
        # self.conv_refin=nn.Conv2d(11,20,3,1,1)
        self.tanh=nn.Tanh()


        self.conv11 = nn.Conv2d(64, 64, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv21 = nn.Conv2d(64, 64, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv31 = nn.Conv2d(64, 64, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv41 = nn.Conv2d(128, 64, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv51 = nn.Conv2d(128, 64, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv61 = nn.Conv2d(64, 64, kernel_size=1,stride=1,padding=0)  # 1mm

        #self.refine3= nn.Conv2d(20+4, 3, kernel_size=3,stride=1,padding=1)
        # self.refine3= nn.Conv2d(20+4, 3, kernel_size=7,stride=1,padding=3)

        self.upsample = F.upsample_nearest

        self.relu=nn.LeakyReLU(0.2, inplace=True)


        #self.batchnorm20=nn.BatchNorm2d(20)
        #self.batchnorm1=nn.BatchNorm2d(1)

        self.fusion = nn.Sequential(
            nn.Conv2d(448, 64, 1, 1, 0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.fusion2 = nn.Sequential(
            nn.Conv2d(64+3, 64, 1, 1, 0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 3, 3, 1, 1),
            nn.Tanh(),
        )

        self.cnt = 0

    def forward(self, x, e, labels):
        ## 256x256
        f0 = self.conv1(x)
        f0_e = self.conv1_e(e)
        tmp = torch.cat([f0, f0_e], 1)
        tmp = self.se_tmp(tmp)
        f1 = self.conv2(tmp)
        S = f1.size()
        label_0, label_1, label_2 = labels
        # print(50*'-')
        # print(label_0)
        val_0 = []
        val_1 = []
        val_2 = []
        for i in range(S[0]):
            val_0.append(label_0[i].item())
            val_1.append(label_1[i].item())
            val_2.append(label_2[i].item())
        label_0.resize_((S[0], 1, S[2], S[3]))
        label_1.resize_((S[0], 1, S[2], S[3]))
        label_2.resize_((S[0], 1, S[2], S[3]))
        for i in range(S[0]):
            label_0[i].fill_(val_0[i])
            label_1[i].fill_(val_1[i])
            label_2[i].fill_(val_2[i])

        # print(50 * '-')
        # print(label_0)
        # t_f1 = self.se0(torch.cat([f1, label_0, label_1, label_2], 1))
        # #
        # #
        # #
        # # print("t_f1")
        # # print(t_f1.shape)
        # f1_ = t_f1
        x1=self.dense_block1(f1)
        # print("x1_")
        # print(x1_.shape)
        x1=self.trans_block1(x1)
        # x1 = self.se0(x1)
        ###  32x32
        x2=(self.dense_block2(x1))
        x2=self.trans_block2(x2)
        # x2 = self.se1(x2)

        x3=(self.dense_block3(x2))
        x3=self.trans_block3(x3)
        # x3 = self.se2(x3)
        ## Classifier  ##

        x4=(self.dense_block4(x3))
        x4=self.trans_block4(x4)

        # x4=x4+x2
        x4=torch.cat([x4,x2],1)
        # x4 = self.se3(x4)
        x5=(self.dense_block5(x4))
        x5=self.trans_block5(x5)

        # x5=x5+x1
        x5=torch.cat([x5,x1],1)
        # x5 = self.se4(x5)
        x6=(self.dense_block6(x5))
        x6=(self.trans_block6(x6))
        # print("x6")
        # print(x6.shape)
        # r1 = torch.cat([f1, x6], dim=1)
        # r2 = torch.cat([r1, f1_], dim=1)
        # print(50*'-')
        # print(r2.shape)


        shape_out = x6.data.size()
        # print(shape_out)
        shape_out = shape_out[2:4]

        x11 = self.upsample(self.relu((self.conv11(x1))), size=shape_out)
        x21 = self.upsample(self.relu((self.conv21(x2))), size=shape_out)
        x31 = self.upsample(self.relu((self.conv31(x3))), size=shape_out)
        x41 = self.upsample(self.relu((self.conv41(x4))), size=shape_out)
        x51 = self.upsample(self.relu((self.conv51(x5))), size=shape_out)
        x61 = self.relu((self.conv61(x6)))
        # os.mkdir('./record'+os.sep + str(self.cnt))

        # HeatMap(x11, './record'+os.sep+str(self.cnt) + os.sep+'x11')
        # HeatMap(x21,  './record'+os.sep+str(self.cnt) + os.sep+'x21')
        # HeatMap(x31,  './record'+os.sep+str(self.cnt) + os.sep+'x31')
        # HeatMap(x41,  './record'+os.sep+str(self.cnt) + os.sep+'x41')
        # HeatMap(x51, './record'+os.sep +str(self.cnt) + os.sep+'x51')
        # HeatMap(x61, './record'+os.sep +str(self.cnt) + os.sep+'x61')

        r1 = torch.cat([f1, x11, x21, x31, x41, x51, x61], dim=1)
        r1 = self.se5(r1)
        # HeatMap(r1, './record' + os.sep + str(self.cnt) + os.sep + 'r1')
        # print(50*'-')
        # print(r1.shape)
        feature = self.fusion(r1)
        # HeatMap(feature, './record' + os.sep + str(self.cnt) + os.sep + 'fusion')
        feature = torch.cat([feature, label_0, label_1, label_2], dim=1)
        feature = self.se6(feature)
        feature = self.fusion2(feature)
        feature = feature + f0
        self.cnt +=1
        return self.final_conv(feature)

        #
        #
        # x6=torch.cat([x6,x51,x41,x31,x21,x11,x],1)
        #
        # return x6

