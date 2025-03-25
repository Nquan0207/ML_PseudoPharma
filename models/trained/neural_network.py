# Library
import numpy as np
import pandas as pd
import random
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
train = np.load("../data/preprocessed/train_data.npz") 
test = np.load("../data/preprocessed/test_data.npz")


# Get the data for training and testing
X_train = torch.tensor(train["x_train"], dtype=torch.float32)
y_train = torch.tensor(train['y_train'], dtype=torch.long)
X_test = torch.tensor(test["x_test"], dtype=torch.float32)
y_test = torch.tensor(test['y_test'], dtype=torch.long)


print(X_train[:1])
print(y_train[:1])
print()

# # Train model

num_classes = len(np.unique(y_train))
train_size = int(0.8 * len(X_train))
val_size = len(X_train) - train_size
train_data, val_data = random_split(TensorDataset(X_train, y_train), [train_size, val_size])

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Define the MLP model
class MLPModel(nn.Module):
    def __init__(self, input_size=50, hidden_sizes=[256, 128, 128], output_size=9):
        super(MLPModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.Tanh(),
            nn.Linear(hidden_sizes[1], hidden_sizes[2]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[2], output_size),
        )
    
    def forward(self, x):
        return self.model(x)

# Initialize model, loss, and optimizer
model = MLPModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

# Create DataLoader
batch_size = 64
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

# Function to train and evaluate the model
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10):
    best_val_acc = 0.0
    best_val_loss = float('inf')
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == batch_y).sum().item()
                total += batch_y.size(0)
        val_accuracy = correct / total
        avg_val_loss = val_loss / len(val_loader)
        model.train()

        # Save model if validation accuracy improves
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save(model.state_dict(), "best_model_acc.pth")
            print(f"✅Model saved at epoch {epoch+1} with Val Acc: {val_accuracy:.4f}")
        
        # Save model if validation loss improves
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_model_loss.pth")
            print(f"🚹Model saved at epoch {epoch+1} with Val Loss: {avg_val_loss:.4f}")
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_accuracy:.4f}")
    
    return best_val_acc, best_val_loss  # Return validation accuracy and loss

# Train the model with fixed hyperparameters
train_model(model, train_loader, val_loader, criterion, optimizer, epochs=2)


# # Hyperparameter


# Parameter Tuning Section
print("\nStarting Parameter Tuning...")

# Define hyperparameter grid for tuning
param_grid = {
    'lr': [0.01, 0.1],  # Learning rates to try
    'batch_size': [64, 128],  # Batch sizes to try
    'hidden_sizes': [[256, 128, 128], [256, 256, 128], [256, 256, 256]],  # Architectures to try
    'momentum': [0.9, 0.95],  # Momentum values to try
}

# Perform grid search
best_hyperparams = None
best_accuracy = 0.0

for params in ParameterGrid(param_grid):
    print(f"\nTraining with hyperparameters: {params}")
    
    # Create DataLoader with current batch size
    train_loader = DataLoader(train_data, batch_size=params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_data, batch_size=params['batch_size'], shuffle=False)
    
    # Initialize model, loss, and optimizer with current hyperparameters
    model = MLPModel(hidden_sizes=params['hidden_sizes']).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=params['lr'], momentum=params['momentum'])
    
    # Train the model
    val_accuracy, val_loss = train_model(model, train_loader, val_loader, criterion, optimizer, epochs=2)
    
    # Track the best hyperparameters
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        best_hyperparams = params

print(f"\nBest hyperparameters: {best_hyperparams} with validation accuracy: {best_accuracy:.4f}")


# # Cross Validation
# Define the number of folds (k)
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)

# Lists to store results for each fold
fold_results = []

# Perform k-fold cross-validation
for fold, (train_idx, val_idx) in enumerate(kf.split(train_data)):
    print(f"\nFold {fold + 1}/{k}")
    
    # Split dataset into training and validation sets for this fold
    train_fold = torch.utils.data.Subset(train_data, train_idx)
    val_fold = torch.utils.data.Subset(train_data, val_idx)
    
    # Create DataLoader for this fold
    train_loader = DataLoader(train_fold, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_fold, batch_size=batch_size, shuffle=False)
    
    # Initialize a new model for this fold
    model = MLPModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    
    # Train and evaluate the model
    val_accuracy, val_loss = train_model(model, train_loader, val_loader, criterion, optimizer, epochs=1)
    
    # Store results for this fold
    fold_results.append({
        'fold': fold + 1,
        'val_accuracy': val_accuracy,
        'val_loss': val_loss
    })
    print(f"Fold {fold + 1} completed. Val Acc: {val_accuracy:.4f}, Val Loss: {val_loss:.4f}")

# Aggregate results across all folds
avg_val_acc = np.mean([result['val_accuracy'] for result in fold_results])
avg_val_loss = np.mean([result['val_loss'] for result in fold_results])
print(f"\nCross-Validation Results: Avg Val Acc: {avg_val_acc:.4f}, Avg Val Loss: {avg_val_loss:.4f}")


# # Prediction
# Create DataLoader for testing
test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Function to evaluate the model
def evaluate_model(model, test_loader):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(batch_y.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    return y_true, y_pred

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(9), yticklabels=range(9))
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# Load the best model based on validation accuracy
model_acc = MLPModel().to(device)
model_acc.load_state_dict(torch.load("best_model_acc.pth"))
model_acc.eval()

# Evaluate the best accuracy model
y_true_acc, y_pred_acc = evaluate_model(model_acc, test_loader)
print("Classification Report for Best Accuracy Model:")
print(classification_report(y_true_acc, y_pred_acc))
plot_confusion_matrix(y_true_acc, y_pred_acc, "Confusion Matrix for Best Accuracy Model")

# Load the best model based on validation loss
model_loss = MLPModel().to(device)
model_loss.load_state_dict(torch.load("best_model_loss.pth"))
model_loss.eval()

# Evaluate the best loss model
y_true_loss, y_pred_loss = evaluate_model(model_loss, test_loader)
print("Classification Report for Best Loss Model:")
print(classification_report(y_true_loss, y_pred_loss))
plot_confusion_matrix(y_true_loss, y_pred_loss, "Confusion Matrix for Best Loss Model")





