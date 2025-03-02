from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import numpy as np
import joblib
import os

# Load training data
train_data_path = os.path.join(os.path.dirname(__file__), "../../data/preprocessed/train_data.npz")

# Load the dataset
train_data = np.load(train_data_path, allow_pickle=True)
x_train_with_stats = train_data["x_train"]
y_train = train_data["y_train"]
test_data = np.load("data/preprocessed/test_data.npz")
x_test_with_stats = test_data["x_test"]
y_test = test_data["y_test"]

# Define a smaller, optimized parameter grid
param_grid = {
    "criterion": ["gini", "entropy"],
    "max_depth": [5, 10, 15],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}

dt = DecisionTreeClassifier()

grid_search = GridSearchCV(dt, param_grid, cv=3, scoring='accuracy', n_jobs=-1)

# Fit the model (this finds the best hyperparameters)
grid_search.fit(x_train_with_stats, y_train)

# Get best model
best_model = grid_search.best_estimator_

# Make predictions
y_pred = best_model.predict(x_test_with_stats)

# Evaluate performance
print(classification_report(y_test, y_pred))
print("Best Parameters:", grid_search.best_params_)
# save model
model_save_path = os.path.join("models", "trained", "decision_tree_best_grid.pkl")

# Ensure the directory exists
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

# Save the model
joblib.dump(best_model, model_save_path)
print(f"Model saved successfully at: {model_save_path}")