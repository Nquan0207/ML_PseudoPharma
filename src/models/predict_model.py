import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Define the model path
model_path = os.path.join(os.path.dirname(__file__), "../../models/trained/decision_tree_best_grid.pkl")
# Load the trained model
best_tree_model = joblib.load(model_path)
print("Model loaded successfully!")
# Load the dataset
test_data_path = os.path.join(os.path.dirname(__file__), "../../data/preprocessed/test_data.npz")
test_data = np.load(test_data_path, allow_pickle=True)
x_test_with_stats = test_data["x_test"]
y_test = test_data["y_test"]

# Make predictions using the loaded model
y_pred = best_tree_model.predict(x_test_with_stats)

# Evaluate performance
print("\n Classification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()