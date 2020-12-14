import os, cv2
import numpy as np
import pandas as pd
import random, tqdm
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import albumentations as album
import segmentation_models_pytorch as smp

def displayimages(**images):
    n_images = len(images)
    plt.figure(figsize=(16,8))
    for idx,(name,image) in enumerate(images.items()):
        plt.subplot(1, n_images, idx + 1)
        plt.xticks([]); 
        plt.yticks([])
        plt.title(name.replace('_',' ').title(), fontsize=20)
        plt.imshow(image)
    plt.savefig("./predictions/plot.png")
    
def onehotencode(label,labelvals):
    semanticmap = []
    for color in labelvals:
        equality = np.equal(label,color)
        classmap = np.all(equality,axis=-1)
        semanticmap.append(classmap)
    semanticmap = np.stack(semanticmap, axis=-1)
    return semanticmap

def reverseonehot(image):
    rev = np.argmax(image,axis=-1)
    return rev

def color_code_segment(image,labelvals):
    colorcodes = np.array(labelvals)
    ccs = colorcodes[image.astype(int)]
    return ccs

def training_augmentations():
    transform = [    
        album.RandomCrop(height=256, width=256, always_apply=True),
        album.OneOf([album.HorizontalFlip(p=1),album.VerticalFlip(p=1),album.RandomRotate90(p=1)],p=0.75)]
    return album.Compose(transform)

def validation_augmentations():   
    transform = [album.PadIfNeeded(min_height=1536, min_width=1536, always_apply=True, border_mode=0)]
    return album.Compose(transform)

def convert_to_tensor(x,**kwargs):
    return x.transpose(2,0,1).astype("float32")

def func_for_preprocessing(preprocessing_fn=None):
    transform = []
    if preprocessing_fn:
        transform.append(album.Lambda(image=preprocessing_fn))
    transform.append(album.Lambda(image=convert_to_tensor,mask=convert_to_tensor))
    return album.Compose(transform)

def crop_image(image, target_image_dims=[1500,1500,3]):
   
    target_size = target_image_dims[0]
    image_size = len(image)
    padding = (image_size - target_size) // 2

    return image[
        padding:image_size - padding,
        padding:image_size - padding,
        :,]
    
class DatasetCreation(torch.utils.data.Dataset):
    def __init__(self,images_dir,masks_dir,class_rgb_vals=None, augment=None,preprocess=None):
        self.imagespath = [os.path.join(images_dir,imageid) for imageid in sorted(os.listdir(images_dir))]
        self.maskspath = [os.path.join(masks_dir,maskid) for maskid in sorted(os.listdir(masks_dir))]
        self.class_rgb_vals = class_rgb_vals
        self.augment = augment
        self.preprocess = preprocess
        
    def __len__(self):
        return len(self.imagespath)
        
    def __getitem__(self,i):
        image = cv2.cvtColor(cv2.imread(self.imagespath[i]),cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(cv2.imread(self.maskspath[i]),cv2.COLOR_BGR2RGB)
        mask = onehotencode(mask,self.class_rgb_vals).astype("float")
        if self.augment:
            sample = self.augment(image=image, mask=mask)
            image,mask = sample['image'],sample['mask']
        if self.preprocess:
            sample = self.preprocess(image=image,mask=mask)
            image,mask = sample['image'],sample['mask']
        return image,mask
    