# MopNet
This code is the official implementation of ICCV 2019 paper "Mop Moire Patterns Using Mopnet".

## Prerequisites:
1. Linux
2. python2 or 3
3. NVIDIA GPU + CUDA CuDNN (CUDA 8.0)

## Insrallation:
1. Install PyTorch from  http://pytorch.org
2. Install Torch vision from https://github.com/pytorch/vision 
3. Install python package: numpy, scipy, PIL, math, skimage, visdom

## Download pre-trained model:
1. `VGG16`  https://drive.google.com/open?id=1wNHZOyTr3veCHU-JaQwmSV7JbKWIMbAT
2. `classifier`  https://drive.google.com/drive/folders/1MkSVkzwWeKmaIRIzq9-J4ZINEFq4TRFA
3. `caorse pre-trained edge predictor`  https://drive.google.com/drive/folders/1MkSVkzwWeKmaIRIzq9-J4ZINEFq4TRFA
4. `totally pre-trained mopnet`  https://drive.google.com/drive/folders/1MkSVkzwWeKmaIRIzq9-J4ZINEFq4TRFA

## Testing:
Download totally pre-trained mopnet and classifier
put color_epoch_95.pth and geo_epoch_95.pth into folder classifier 
put netEdge_epoch_150.pth and netG_epoch_150.pth into folder mopnet
Download testset from 
https://drive.google.com/open?id=1a-4iwy3ujCfC8llBaimjXnVfOM9oGKAV

Change the dataroot in run_test.sh
Create folders:
    `mkdir results`
    `mkdir results/d`
    `mkdir results/o`
    `mkdir results/g`

execute
`bash run_test.sh`
Then you will get moire free images.
For fair comparison, we compute the PNSR and SSIM in Matlab which is the same as TIP18. 
(Moire Photo Restoration Using Multiresolution Convolutional Neural Networks)
So you can run 
`matlab test_with_matlabcode.m`
to get quantitative results.

## Training:
Download caorse pre-trained edge predicotr and put it into folder edge.
Download VGG and put it into folder models
Download the dataset from 
https://drive.google.com/open?id=1a-4iwy3ujCfC8llBaimjXnVfOM9oGKAV
The whole benchmark training set please contact the author of TIP18. 
change the dataroot and valDataroot in run_train.sh
open the visualization:
`python -m visdom.server -port 8098`

execute
`bash run_train.sh`

If you have any question, feel free to concat me with cs_hebin@pku.edu.cn





