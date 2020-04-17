import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np

import sys

IMG_EXTENSIONS = [
  '.jpg', '.JPG', '.jpeg', '.JPEG',
  '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '',
]

def is_image_file(filename):
  return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
  images = []
  if not os.path.isdir(dir):
    raise Exception('Check dataroot')
  for root, _, fnames in sorted(os.walk(dir)):
    for fname in fnames:
      if is_image_file(fname):
        path = os.path.join(dir, fname)
        item = path
        images.append(item)
  return images

def make_labels(dir, k):
    D = {}
    f = open(dir)
    line = f.readline()
    while line:
        key, v = line.split(':')
        v = v.strip('\n')
        v = int(v.split(' ')[k])
        D[key] = v
        line = f.readline()
    f.close()
    return D

# Crop input as Sun et al.
def default_loader(path):
  img = Image.open(path).convert('RGB')
  w, h = img.size
  region = img.crop((1+int(0.15*w), 1+int(0.15*h), int(0.85*w), int(0.85*h)))
  return region

# Original dataloader
def default_loader2(path):
  img = Image.open(path).convert('RGB')
  return img

# load paired images with labels
class my_loader(data.Dataset):
  def __init__(self, root, transform=None, loader=default_loader, seed=None, pre="", label_file=''):
    self.source_path = root+os.sep+pre+"source"
    self.target_path = root+os.sep+pre+"target"
    src_imgs = make_dataset(self.source_path)
    self.label_file = label_file
    if label_file!='':
        self.labels_0 = make_labels(label_file, 0)
        self.labels_1 = make_labels(label_file, 1)
        self.labels_2 = make_labels(label_file, 2)
    if len(src_imgs) == 0:
      raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                 "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
    self.root = root

    self.src_imgs = src_imgs
    self.transform = transform
    self.loader = loader
    if seed is not None:
      np.random.seed(seed)

  def __getitem__(self, index):

    src_path = self.src_imgs[index]
    (filename, tempfilename) = os.path.split(src_path)
    (short_name, extension) = os.path.splitext(tempfilename)
    tmp = short_name.split('_')
    tar_path = self.target_path + os.sep + tmp[0] + '_' + tmp[1] + "_" + tmp[2] + "_" + tmp[3] + "_" + "target" + extension

    imgA = self.loader(src_path)
    imgB = self.loader(tar_path)

    if self.label_file!='':
        label_0 = self.labels_0[tempfilename]
        label_1 = self.labels_1[tempfilename]
        label_2 = self.labels_2[tempfilename]

    if self.transform is not None:
      # NOTE preprocessing for each pair of images
      imgA, imgB = self.transform(imgA, imgB)

    if self.label_file != '':
        return imgA, imgB, label_0, label_1, label_2
    else:
        return imgA, imgB

  def __len__(self):
    return len(self.src_imgs)
