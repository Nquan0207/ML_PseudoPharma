from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import numpy as np
import joblib
import os
from hyperparam_tuning import *

dt_classifier = DecisionTreeClassifier()

grid_search = GridSearchCV(estimator=dt_classifier, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(x_train, y_train)

# Output the best parameters and best score from cross-validation
print("Best Parameters Found:", grid_search.best_params_)
print("Best Cross-Validation Score:", grid_search.best_score_)
# Get best model
best_model = grid_search.best_estimator_

# Make predictions
y_pred = best_model.predict(x_test)

# Evaluate performance
print(classification_report(y_test, y_pred))
print("Best Parameters:", grid_search.best_params_)

# save model
model_save_path = os.path.join("models", "trained", "best_decision_tree_model.pkl")

# Ensure the directory exists
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

# Save the model
joblib.dump(best_model, model_save_path)
print(f"Model saved successfully at: {model_save_path}")