import os 
from PIL import Image
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
import numpy as np
import math
from option import opt

class MyPolDataset(Dataset):
    def __init__(self, path_dir, transform=None):
        self.path_dir = path_dir
        self.transform = transform
        self.images_s0 = os.listdir(os.path.join(self.path_dir,'S0'))
        self.images_dolp = os.listdir(os.path.join(self.path_dir,'DoLP'))
    
    def __len__(self):
        return len(self.images_s0)

    def __getitem__(self, index):
        image_s0_path = os.path.join(self.path_dir,'S0',self.images_s0[index])
        image_dolp_path = os.path.join(self.path_dir,'DoLP',self.images_dolp[index])
        image_s0 = Image.open(image_s0_path)
        image_dolp = Image.open(image_dolp_path)
        if self.transform is not None:
            image_s0 = self.transform(image_s0)
            image_dolp = self.transform(image_dolp)
        return image_s0, image_dolp

def gradient(input):
    kernel_x = [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
    kernel_x = torch.FloatTensor(kernel_x).unsqueeze(0).unsqueeze(0).to(opt.device)

    kernel_y = [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]
    kernel_y = torch.FloatTensor(kernel_y).unsqueeze(0).unsqueeze(0).to(opt.device)

    grad_x = F.conv2d(input, kernel_x)
    grad_y = F.conv2d(input, kernel_y)
    gradient = torch.abs(grad_x) + torch.abs(grad_y)

    return gradient

def rgb2ycrcb(img_rgb):
    R = torch.unsqueeze(img_rgb[:,0,:,:], dim=1)
    G = torch.unsqueeze(img_rgb[:,1,:,:], dim=1)
    B = torch.unsqueeze(img_rgb[:,2,:,:], dim=1)
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = -0.1687 * R - 0.3313 * G + 0.5 * B + 128/255
    Cr = 0.5 * R - 0.4187 * G - 0.0813 * B + 128/255
    # img_ycbcr = torch.cat([Y, Cr, Cb], 1)
    return Y, Cr, Cb

def ycrcb2rgb(Y, Cr, Cb):
    B = (Cb - 0.5) * 1. / 0.564 + Y
    R = (Cr - 0.5) * 1. / 0.713 + Y
    G = 1. / 0.587 * (Y - 0.299 * R - 0.114 * B)
    rgb = torch.cat((R,G,B),1)
    return rgb


def entropy(image, width=480, height=640):
    g = np.histogram(image,bins = 256,range=(0,256))
    num = np.array(g[0])/(width*height)

    result=0.
    for i in range(0,256):
        if num[i] > 0 :
            result += (-num[i])*math.log(num[i],2)
    return result

def data_augmentation(imagein, mode):
    image = imagein
    if mode == 0:
        # original
        return image
    elif mode == 1:
        # flip up and down
        image = torch.flipud(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        image = torch.rot90(image,1,[2,3])
    elif mode == 3:
        # rotate 90 degree and flip up and down
        image = torch.rot90(image,1,[2,3])
        image = torch.flipud(image)
    elif mode == 4:
        # rotate 180 degree
        image = torch.rot90(image,2,[2,3])
    elif mode == 5:
        # rotate 180 degree and flip
        image = torch.rot90(image,2,[2,3])
        image = torch.flipud(image)
    elif mode == 6:
        # rotate 270 degree
        image = torch.rot90(image,3,[2,3])
    elif mode == 7:
        # rotate 270 degree and flip
        image = torch.rot90(image, 3,[2,3])
        image = torch.flipud(image)
    imageout = image
    return imageout