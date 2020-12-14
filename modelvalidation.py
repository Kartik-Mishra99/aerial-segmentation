import os, cv2
import numpy as np
import pandas as pd
import random, tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import sys

import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import albumentations as album
import segmentation_models_pytorch as smp

from utils import *


modelpath = "savedmodels/myfinalmodel.pth"


encoder = "resnet101"
encoder_weights = "imagenet"
activation = "sigmoid"
preprocess_func = smp.encoders.get_preprocessing_fn(encoder,encoder_weights)
rootdir = "tiff/"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loss = smp.utils.losses.DiceLoss()
metrics = [smp.utils.metrics.IoU(threshold=0.5)]
bestmodel = torch.load(modelpath, map_location=device)

print('Loaded DeepLabV3 plus model.')

x_test_dir = os.path.join(rootdir, 'test')
y_test_dir = os.path.join(rootdir, 'test_labels')

classlabeldict = pd.read_csv("label_class_dict.csv")
clasnames = classlabeldict['name'].tolist()
class_rgb_values = classlabeldict[['r','g','b']].values.tolist()

select_class_indices = [clasnames.index(cls.lower()) for cls in clasnames]
select_class_rgb_values =  np.array(class_rgb_values)[select_class_indices]

testdata = DatasetCreation(x_test_dir,y_test_dir,augment = validation_augmentations(),
                           preprocess = func_for_preprocessing(preprocess_func),
                           class_rgb_vals = select_class_rgb_values)

testloader = DataLoader(testdata)

testdata_without_preprocess = DatasetCreation(x_test_dir,y_test_dir,augment = validation_augmentations(),
                           class_rgb_vals = select_class_rgb_values)


sample_preds_folder = 'predictions/'
if not os.path.exists(sample_preds_folder):
    os.makedirs(sample_preds_folder)

print("Making Predictions on test images now !")
for idx in range(len(testdata)):
    image, gt_mask = testdata[idx]
    image_vis = crop_image(testdata_without_preprocess[idx][0].astype('uint8'))
    x_tensor = torch.from_numpy(image).to(device).unsqueeze(0)
    pred_mask = bestmodel(x_tensor)
    pred_mask = pred_mask.detach().squeeze().cpu().numpy()
    pred_mask = np.transpose(pred_mask,(1,2,0))
    pred_building_heatmap = pred_mask[:,:,clasnames.index('building')]
    pred_mask = crop_image(color_code_segment(reverseonehot(pred_mask), select_class_rgb_values))
    gt_mask = np.transpose(gt_mask,(1,2,0))
    gt_mask = crop_image(color_code_segment(reverseonehot(gt_mask), select_class_rgb_values))
    f, axarr = plt.subplots(nrows=1,ncols=3,figsize=(16,8))
    plt.sca(axarr[0]); 
    plt.imshow(image_vis); plt.title('Original Image')
    plt.sca(axarr[1]); 
    plt.imshow(gt_mask); plt.title('Ground Truth')
    plt.sca(axarr[2]); 
    plt.imshow(pred_mask); plt.title('Prediction Mask')
    plt.savefig(os.path.join(sample_preds_folder,f"predimage_{idx}.png"))
    # displayimages(original_image = image_vis,ground_truth_mask = gt_mask,predicted_mask = pred_mask)
print("Saved Predicted masks to predictions folder !")


testepoch = smp.utils.train.ValidEpoch(bestmodel,loss=loss, metrics=metrics, device=device,verbose=True)
valid_logs = testepoch.run(testloader)

print("Evaluation Results on Test Data: ")
print(f"Mean IoU Score: {valid_logs['iou_score']:.4f}")
print(f"Mean Dice Loss: {valid_logs['dice_loss']:.4f}")

