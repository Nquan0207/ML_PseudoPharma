from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns, matplotlib.pyplot as plt, joblib
from sklearn.metrics import classification_report
import numpy as np
import joblib
import os
from hyperparam_tuning import *         
from sklearn.pipeline import Pipeline


hgb = HistGradientBoostingClassifier(
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    random_state=42
)

grid = GridSearchCV(
    estimator=hgb,
    param_grid=param_grid_en,
    cv=3,                  # 3-fold stratified CV
    scoring="accuracy",
    n_jobs=-1,             
    verbose=2
)

grid.fit(x_train, y_train)
best_gb = grid.best_estimator_

print("Best params:", grid.best_params_)

# ----------------------  Evaluate on the hold-out test set  -------
y_pred = best_gb.predict(x_test)
print("\n=== HistGradientBoosting (GridSearch) – Report ===")
print(classification_report(y_test, y_pred))

# ----------------------  Persist for later use  -------------------
joblib.dump(best_gb, "pathmnist_histgb_grid.pkl")
print("Model saved -> pathmnist_histgb_grid.pkl")