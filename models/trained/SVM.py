from sklearn.metrics import classification_report
import numpy as np
import joblib
import os
from hyperparam_tuning import *
from sklearn.svm import SVC            
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix

svm_pipe = Pipeline([
    ('scale', StandardScaler()),       
    ('svm',   SVC(kernel='rbf',          
                  class_weight='balanced',   
                  probability=False,          
                  random_state=42))
])
cv = StratifiedKFold(5, shuffle=True, random_state=42)
grid = GridSearchCV(svm_pipe,
                    param_grid_svm,
                    cv=cv,
                    scoring='accuracy',
                    n_jobs=-1,
                    verbose=1)

grid.fit(x_train, y_train)

print(f"Best CV accuracy: {grid.best_score_:.3f} "
      f"with params: {grid.best_params_}")

y_pred = grid.predict(x_test)

print("\nClassification report:")
print(classification_report(y_test, y_pred, digits=4))

print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))

joblib.dump(grid.best_estimator_, 'svm.pkl')