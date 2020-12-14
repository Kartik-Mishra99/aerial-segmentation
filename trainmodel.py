import os, cv2
import numpy as np
import pandas as pd
import random, tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import albumentations as album
import segmentation_models_pytorch as smp

from utils import *



rootdir = "tiff/"
x_train_dir = os.path.join(rootdir, 'train')
y_train_dir = os.path.join(rootdir, 'train_labels')

x_valid_dir = os.path.join(rootdir, 'val')
y_valid_dir = os.path.join(rootdir, 'val_labels')


classlabeldict = pd.read_csv("label_class_dict.csv")
clasnames = classlabeldict['name'].tolist()
class_rgb_values = classlabeldict[['r','g','b']].values.tolist()

select_class_indices = [clasnames.index(cls.lower()) for cls in clasnames]
select_class_rgb_values =  np.array(class_rgb_values)[select_class_indices]

# # ORIGINAL IMAGES
# dataset = DatasetCreation(x_train_dir, y_train_dir, class_rgb_vals=select_class_rgb_values)
# random_idx = random.randint(0, len(dataset)-1)
# image,mask = dataset[random_idx]

# displayimages(original_image=image,ground_truth_mask=color_code_segment(reverseonehot(mask),select_class_rgb_values),
#             onehot_encoded_image = reverseonehot(mask))

# # AUGMENTED IMAGES

# augment_dataset = DatasetCreation(x_train_dir, y_train_dir, class_rgb_vals=select_class_rgb_values,
#                                 augment=training_augmentations())

# random_idx = random.randint(0, len(augment_dataset)-1)

# for i in range(5):
#     image,mask = image,mask = augment_dataset[random_idx]
#     displayimages(original_image=image,ground_truth_mask=color_code_segment(reverseonehot(mask),select_class_rgb_values),
#             onehot_encoded_image = reverseonehot(mask))

            
############## MODEL TRAINING ##########

# DeepLabV3 model 

encoder = "resnet101"
encoder_weights = "imagenet"
activation = "sigmoid"

model = smp.DeepLabV3Plus(encoder_name=encoder,encoder_weights=encoder_weights,\
                        classes=len(clasnames),activation="sigmoid")

preprocess_func = smp.encoders.get_preprocessing_fn(encoder,encoder_weights)

traindata = DatasetCreation(x_train_dir,y_train_dir,augment = training_augmentations(),
                        preprocess = func_for_preprocessing(preprocess_func),
                        class_rgb_vals = select_class_rgb_values)

validdata = DatasetCreation(x_valid_dir,y_valid_dir,augment = validation_augmentations(),
                        preprocess = func_for_preprocessing(preprocess_func),
                        class_rgb_vals = select_class_rgb_values)

trainloader = DataLoader(traindata,batch_size=16,shuffle=True)
validloader = DataLoader(validdata,batch_size=1,shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loss = smp.utils.losses.DiceLoss()
metrics = [smp.utils.metrics.IoU(threshold=0.5)]
optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=0.0001)])
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2, eta_min=5e-5)

if os.path.exists('./deeplabv3-using-pytorch/best_model.pth'):
    model = torch.load('./deeplabv3using-pytorch/best_model.pth', map_location=device)

trainepoch = smp.utils.train.TrainEpoch(model,loss=loss,optimizer=optimizer,metrics=metrics,device=device,verbose=True)
validepoch = smp.utils.train.ValidEpoch(model,loss=loss,metrics=metrics,device=device,verbose=True)


def train(epochs=20):
    print("Starting model training !")
    best_iou_score = 0.0 
    train_logs_list, valid_logs_list = [], []
    for i in range(0,epochs):
        print('\nEpoch: {}'.format(i))
        trainlogs = trainepoch.run(trainloader)
        validlogs = validepoch.run(validloader)
        train_logs_list.append(trainlogs)
        valid_logs_list.append(validlogs)
        if best_iou_score < validlogs['iou_score']:
            best_iou_score = validlogs['iou_score']
            torch.save(model, './savedmodels/bestmodel.pth')
    print("Model Training completed successfully !")
    train_logs_df = pd.DataFrame(train_logs_list)
    valid_logs_df = pd.DataFrame(valid_logs_list)
    train_logs_df.to_csv("./result-plots-files/trainlogs.csv",index=None)
    valid_logs_df.to_csv("./result-plots-files/validlogs.csv",index=None)

    print("train and validation logs saved !")
    plt.figure(figsize=(20,8))
    plt.plot(train_logs_df.index.tolist(), train_logs_df.iou_score.tolist(),'g-',lw=3, label = 'Train')
    plt.plot(valid_logs_df.index.tolist(), valid_logs_df.iou_score.tolist(),'ro' ,lw=3, label = 'Valid')
    plt.xlabel('Epochs', fontsize=20)
    plt.ylabel('IoU Score', fontsize=20)
    plt.title('IoU Score Plot', fontsize=20)
    plt.legend(loc='best', fontsize=16)
    plt.grid()
    plt.savefig('./result-plots-files/iou_score_plot.png')

    plt.figure(figsize=(20,8))
    plt.plot(train_logs_df.index.tolist(), train_logs_df.dice_loss.tolist(),'g-', lw=3, label = 'Train')
    plt.plot(valid_logs_df.index.tolist(), valid_logs_df.dice_loss.tolist(),'ro', lw=3, label = 'Valid')
    plt.xlabel('Epochs', fontsize=20)
    plt.ylabel('Dice Loss', fontsize=20)
    plt.title('Dice Loss Plot', fontsize=20)
    plt.legend(loc='best', fontsize=16)
    plt.grid()
    plt.savefig('./result-plots-files/dice_loss_plot.png')
    print("Plots saved successfully !")




if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",help="Enter number of epochs to train the model",type=int)
    args = parser.parse_args()
    epochs = args.epochs
    train(epochs)

