from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import numpy as np
import joblib
import os

# Load training data
train_data_path = os.path.join(os.path.dirname(__file__), "../../../data/preprocessed/train_data.npz")

# Load the dataset
train_data = np.load(train_data_path, allow_pickle=True)
x_train = train_data_path['x_train']
y_train = train_data_path['y_train']
test_data = np.load("../../../data/preprocessed/test_data.npz")
x_test = test_data['x_test']
y_test = test_data['y_test']
print(x_train[0:10])
# Define a smaller, optimized parameter grid
param_grid = {
    "criterion": ["gini", "entropy"],
    "max_depth": [5, 10, 15],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}

param_grid1 = {
    "var_smoothing": [10**i for i in range(-9, 0)]
}
