"""
This file is used to load HAM10000(Human against Data 10000)  dataset from the following link:
https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000 

"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from glob import glob
from PIL import Image

# Read the csv file containing image names and corresponding labels (metadata file)
skin_dataframe = pd.read_csv('archive/HAM10000_metadata.csv')
print(skin_dataframe['dx'].value_counts())


#The original data are in 2 folder with 1st folder having 5000 images and the other having 50015 images.
#Reorganize data into subfolders based on their labels
#Use pytorch ImageFolder to read images with folder names as labels

#Sort images to subfolders first 
import pandas as pd
import os
import shutil

# Dump all images into a folder and specify the path:
data_dir = os.getcwd() + "/archive/all_images/"

# Path to destination directory where we want subfolders
destination_dir = os.getcwd() + "/archive/reorganized/"



label=skin_dataframe['dx'].unique().tolist()  #Extract labels into a list
label_images = []


# Copy images to new folders
for i in label:
    os.mkdir(destination_dir + str(i) + "/")
    sample = skin_dataframe[skin_dataframe['dx'] == i]['image_id']
    label_images.extend(sample)
    for id in label_images:
        shutil.copyfile((data_dir + "/"+ id +".jpg"), (destination_dir + i + "/"+id+".jpg"))
    label_images=[]    

# Now ready to work with images in subfolders

### PYTORCH using ImageFolder
import torchvision
from torchvision import transforms
import torch.utils.data as data
import numpy as np

#Define root directory with subdirectories
train_dir = os.getcwd() + "/data/reorganized/"

#If you want to apply ransforms
TRANSFORM_IMG = transforms.Compose([
    transforms.Resize(32),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),       
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5] )
    ])

#With transforms
#train_data = torchvision.datasets.ImageFolder(root=TRAIN_DATA_PATH, transform=TRANSFORM_IMG)

#Without transforms
train_data_torch = torchvision.datasets.ImageFolder(root=train_dir)
    
print("Number of train samples: ", len(train_data_torch))    
print("Detected Classes are: ", train_data_torch.class_to_idx) # classes are detected by folder structure    

labels = np.array(train_data_torch.targets)
(unique, counts) = np.unique(labels, return_counts=True)
frequencies = np.asarray((unique, counts)).T
print(frequencies)
    
