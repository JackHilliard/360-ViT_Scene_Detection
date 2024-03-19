from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import imageio as im
import cv2
import torch
import torch.utils.data
import random
from os.path import join, splitext, basename
from random import randint
from torchvision import transforms
from torchvision.transforms import ToTensor, Normalize, Resize, CenterCrop, ColorJitter, GaussianBlur
import torchvision.transforms.functional as F
import pandas as pd

def h_rotate_torch(img, angle):
    #_, _, W = img.shape
    rot = int((angle/360) * img.shape[2])
    return torch.cat((img[...,int(rot):],img[...,:int(rot)]),dim=-1)

class dataset_sceneDetection(Dataset):
    # prepare data for self-reconstruction,
    # where the two input photos and the intermediate region are obtained from the same image

    def __init__(self,args,root=''):
        # --PARAMS--
        # root: the path of the data
        # crop: 'rand' or 'center' or 'none', which way to crop the image into target size
        # imgSize: the size of the returned image if crop is not 'none'

        #self.transforms = transformations
        self.path = args.path
        self.dataset_dir = args.dataset_dir
        self.width = args.width
        self.height = args.height

        df = pd.read_csv(root)
        cols = df.columns
        self.img_list = df['filename']
        if 'rotation' in cols:
            self.rot_list = df['rotation']
            self.flip_list = df['flip']
        else:
            self.rot_list = np.zeros_like(self.img_list)
            self.flip_list = np.zeros_like(self.img_list)

        self.label_list = df['label']

        self.size = len(self.img_list)

        self.transform = transforms.Compose([
            transforms.Resize((self.height,self.width),interpolation=InterpolationMode.BICUBIC),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
            ])

    def __getitem__(self, index):
        index = index % self.size
        name = self.img_list[index]

        #get image
        ldr_img = Image.open(self.path+self.dataset_dir+name+".png")
        ldr_img = np.asarray(ldr_img.resize((self.width,self.height),Image.BICUBIC))[...,:3]

        if self.flip_list[index]:
            ldr_img = cv2.flip(ldr_img,1)
        ldr_img = F.to_tensor(ldr_img).float().cuda()
        if self.rot_list[index]:
            ldr_img = h_rotate_torch(ldr_img,self.rot_list[index])

        ldr_img = self.transform(ldr_img)
        #ldr_img = ldr_img*2 - 1

        label = torch.as_tensor(self.label_list[index]).cuda().float()

        return ldr_img, label.cuda().float(), f"{splitext(basename(name))[0]}_{self.rot_list[index]}_{self.flip_list[index]}"

    def __len__(self):
        return self.size