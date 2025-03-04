from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import numpy as np
import joblib
import os
from hyperparam_tuning import *

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