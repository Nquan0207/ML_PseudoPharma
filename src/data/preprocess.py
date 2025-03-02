import os
import numpy as np
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from medmnist import INFO, PathMNIST
import medmnist

# Create directory for preprocessed data
preprocessed_dir = "../../data/preprocessed"
os.makedirs(preprocessed_dir, exist_ok=True)

# Data Handling & Preprocessing
data_flag = 'pathmnist'
info = INFO[data_flag]
DataClass = getattr(medmnist, info['python_class'])

data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])

train_data = DataClass(split='train', transform=data_transform, download=True)
val_data = DataClass(split='val', transform=data_transform, download=True)
test_data = DataClass(split='test', transform=data_transform, download=True)

# Flatten images and collect labels
x_train, y_train, x_val, y_val, x_test, y_test = [], [], [], [], [], []
for img, label in train_data:
    x_train.append(img.flatten())
    y_train.append(label)
for img, label in val_data:
    x_val.append(img.flatten())
    y_val.append(label)
for img, label in test_data:
    x_test.append(img.flatten())
    y_test.append(label)

x_train, y_train = np.array(x_train), np.array(y_train)
x_val, y_val = np.array(x_val), np.array(y_val)
x_test, y_test = np.array(x_test), np.array(y_test)

# Feature Engineering
means_train = np.mean(x_train, axis=1)
variances_train = np.var(x_train, axis=1)
x_train_with_stats = np.column_stack((x_train, means_train, variances_train))

means_val = np.mean(x_val, axis=1)
variances_val = np.var(x_val, axis=1)
x_val_with_stats = np.column_stack((x_val, means_val, variances_val))

means_test = np.mean(x_test, axis=1)
variances_test = np.var(x_test, axis=1)
x_test_with_stats = np.column_stack((x_test, means_test, variances_test))

# Save preprocessed data in npz format
np.savez(os.path.join(preprocessed_dir, "train_data.npz"), x_train=x_train_with_stats, y_train=y_train)
np.savez(os.path.join(preprocessed_dir, "val_data.npz"), x_val=x_val_with_stats, y_val=y_val)
np.savez(os.path.join(preprocessed_dir, "test_data.npz"), x_test=x_test_with_stats, y_test=y_test)

