import os
import numpy as np
import torch
import torchvision.transforms as transforms
from medmnist import INFO, PathMNIST
import medmnist
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
import joblib
import random

# Set a fixed random seed for reproducibility
seed = 42
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)

# Create directory for preprocessed data
preprocessed_dir = "../../data/preprocessed/"
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

# Preprocess: Flatten and normalize the images
def preprocess_images(data):
    x_data = []
    for img, label in data:
        img = img.numpy().flatten()  # Flatten and convert to numpy array
        x_data.append(img)
    return np.array(x_data)

# Preprocess the training and test data
x_train = preprocess_images(train_data)
y_train = np.array([label for _, label in train_data])
x_test = preprocess_images(test_data)
y_test = np.array([label for _, label in test_data])

# Ensure y_train and y_test are 1D arrays
y_train = y_train.ravel()  # Flatten the target to 1D array
y_test = y_test.ravel()    # Flatten the target to 1D array

x_train = x_train / 255.0
x_test = x_test / 255.0

# Feature Engineering: Add additional statistical features (mean, variance)
means_train = np.mean(x_train, axis=1)
variances_train = np.var(x_train, axis=1)
x_train_with_stats = np.column_stack((x_train, means_train, variances_train))

means_test = np.mean(x_test, axis=1)
variances_test = np.var(x_test, axis=1)
x_test_with_stats = np.column_stack((x_test, means_test, variances_test))

# Feature Selection: Apply PCA to reduce dimensionality
pca = PCA(n_components=100)  # Reduce to 100 components (tune this value as needed)
x_train_pca = pca.fit_transform(x_train_with_stats)
x_test_pca = pca.transform(x_test_with_stats)

# Feature Selection: Select top 50 features using SelectKBest with ANOVA F-test (f_classif)
selector = SelectKBest(f_classif, k=50)
x_train_selected = selector.fit_transform(x_train_pca, y_train)
x_test_selected = selector.transform(x_test_pca)

# Save preprocessed data in npz format
np.savez("C:\\Users\\Alan Phan\\Desktop\\Bach Khoa Studies\\HK242\\Machine Learning\\Assignments\\AI-MedMNIST-Classification\\AI-MedMNIST-Classification\\data\\preprocessed\\train_data.npz", x_train=x_train_selected, y_train=y_train)
np.savez("C:\\Users\\Alan Phan\\Desktop\\Bach Khoa Studies\\HK242\\Machine Learning\\Assignments\\AI-MedMNIST-Classification\\AI-MedMNIST-Classification\\data\\preprocessed\\test_data.npz", x_test=x_test_selected, y_test=y_test)

print("Preprocessed data saved successfully!")

