from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import numpy as np
import joblib
import os


base_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'preprocessed')
)
train_path = os.path.join(base_dir, 'train_data.npz')
# Load the dataset
train_data = np.load(train_path, allow_pickle=True)
x_train = train_data['x_train']
y_train = train_data['y_train']

test_path = os.path.join(base_dir, 'test_data.npz')
test_data = np.load(test_path, allow_pickle=False)
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

# Hyper-parameter grid for a quick search         
param_grid_svm = {'svm__C':  [0.1, 1, 10],
              'svm__gamma': ['scale', 0.01, 0.001]}

param_grid_en = {
    "learning_rate":   [0.05, 0.07, 0.1],
    "max_depth":       [3, 4, 5],
    "max_iter":        [200, 300],     
    "max_leaf_nodes":  [31, 63],        
}
