import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import medmnist
from medmnist import PathMNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
<<<<<<< HEAD
import cv2
=======
>>>>>>> e76f99c3762c635b3b8035e8a8b50adb6dd1deed
from tqdm import tqdm
from skimage.util import view_as_windows
from PIL import Image

# train_dataset = np.load('../../data/processed/pathmnist_train.npz')
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = PathMNIST(split='train', download=True, transform=transform)
test_dataset = PathMNIST(split = 'test', download = True, transform = transform)

all_train_images = []
all_train_labels = []
for i in range(len(train_dataset)):
    image, label = train_dataset[i]  
    all_train_images.append(image.numpy())
    all_train_labels.append(label)
all_train_images = np.array(all_train_images)
all_train_labels = np.array(all_train_labels)


all_test_images = []
all_test_labels = []
for i in range(len(test_dataset)):
    image, label = test_dataset[i]  
    all_test_images.append(image.numpy())
    all_test_labels.append(label)
all_test_images = np.array(all_test_images)
all_test_labels = np.array(all_test_labels)


max_size = 5000
train_images = all_train_images[:max_size]
train_labels = all_train_labels[:max_size].flatten()
test_images = all_test_images[:max_size]
test_labels = all_test_labels[:max_size].flatten()
train_labels = [str(x) for x in train_labels]
test_labels = [str(x) for x in test_labels]

def generate_feature(images, labels, kernel, step):
    image = images[0]
    _, height, width = image.shape
    column_name = ["mean", "var"]
    for c,channel in enumerate(image):
        number_of_block = 0
        for h in range(0, height - kernel + 1, step):
            for w in range(0, width - kernel + 1, step):
                column_name.append(f"Channel {c} - block {number_of_block} mean")
                column_name.append(f"Channel {c} - block {number_of_block} var")        
                number_of_block += 1
    column_name.append(f"label")
    res_df = pd.DataFrame(columns=column_name)
    for i, image in enumerate(images):
        rec = [np.mean(image), np.var(image)]
        for c, channel in enumerate(image):
            windows = view_as_windows(channel, (kernel, kernel), step)
            means = np.mean(windows, axis=(-2, -1))
            vars = np.var(windows, axis=(-2, -1)) 
            rec.extend(means.flatten())  
            rec.extend(vars.flatten())
        rec.append(int(labels[i]))
        res_df.loc[len(res_df)] = rec
    res_df = res_df.astype(int)
    return res_df

train = generate_feature(train_images, train_labels, 7, 7)
test  = generate_feature(test_images, test_labels, 7, 7)