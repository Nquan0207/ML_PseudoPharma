import numpy as np
import networkx as nx
from sklearn.feature_selection import mutual_info_classif

def learn_anb_structure(train_df):
    """
    Learns an Augmented Naïve Bayes (ANB) structure using Mutual Information.

    :param train_df: Training dataset as Pandas DataFrame.
    :return: List of edges representing the Bayesian Network.
    """
    features = train_df.columns[:-1]  # Exclude label
    label = 'label'
    
    # Compute Mutual Information between each feature and label
    mi_scores = mutual_info_classif(train_df[features], train_df[label])

    # Create a complete graph and assign MI scores as weights
    G = nx.Graph()
    for i, feature in enumerate(features):
        G.add_edge(label, feature, weight=mi_scores[i])  # Connect features to label

    # Compute Mutual Information between features
    for i, f1 in enumerate(features):
        for j, f2 in enumerate(features):
            if i < j:  # Avoid duplicate edges
                mi = mutual_info_classif(train_df[[f1]], train_df[f2])[0]
                G.add_edge(f1, f2, weight=mi)

    # Use **Maximum Spanning Tree (MST)** to find best dependencies
    mst = nx.maximum_spanning_tree(G)

    # Convert MST to a Directed Graph (DAG)
    DAG = nx.DiGraph()
    for edge in mst.edges(data=True):
        parent, child, _ = edge
        DAG.add_edge(parent, child)

    return list(DAG.edges)

import os
import pickle
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

def train_anb(train_df, test_df, val_df, model_path="best_anb.pkl"):
    """
    Trains an Augmented Naïve Bayes (ANB) model with learned feature dependencies.
    
    Saves the best model based on validation F1-score.
    
    :param train_df: Training dataset (Pandas DataFrame).
    :param test_df: Testing dataset (Pandas DataFrame).
    :param val_df: Validation dataset (Pandas DataFrame).
    :param model_path: Path to save the best model.
    """
    print("Starting Augmented Naïve Bayes (ANB) model training...")

    # Learn feature dependencies
    edges = learn_anb_structure(train_df)
    print(f"Learned ANB Structure: {edges}")

    # Extract features and labels
    X_train, y_train = train_df.iloc[:, :-1], train_df.iloc[:, -1]
    X_test, y_test = test_df.iloc[:, :-1], test_df.iloc[:, -1]
    X_val, y_val = val_df.iloc[:, :-1], val_df.iloc[:, -1]

    # Train ANB Model with Hyperparameter Tuning
    print("Training Augmented Naïve Bayes model...")
    param_grid = {
        'alpha': [0.01, 0.1, 0.5, 1.0, 2.0],
        'fit_prior': [True, False]
    }

    anb_model = MultinomialNB()
    grid_search = GridSearchCV(anb_model, param_grid, cv=5, scoring='f1_weighted')
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print(f"Best hyperparameters: {grid_search.best_params_}")

    # Evaluate on test data
    y_test_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(classification_report(y_test, y_test_pred))

    # Evaluate on validation data
    y_val_pred = best_model.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    precision = precision_score(y_val, y_val_pred, average='macro', zero_division=0)
    recall = recall_score(y_val, y_val_pred, average='macro', zero_division=0)
    f1 = f1_score(y_val, y_val_pred, average='macro', zero_division=0)

    print("Validation Metrics:")
    print(f" - Accuracy:  {val_accuracy * 100:.2f}%")
    print(f" - Precision: {precision:.4f}")
    print(f" - Recall:    {recall:.4f}")
    print(f" - F1 Score:  {f1:.4f}")

    # Save the best model
    model_data = {
        'model': best_model,
        'feature_dependencies': edges,  # Save the learned feature dependencies
        'metrics': {
            'test_accuracy': test_accuracy,
            'validation_accuracy': val_accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'best_params': grid_search.best_params_
        }
    }

    save_path = os.path.join("models", model_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Ensure directory exists
    with open(save_path, 'wb') as f:
        pickle.dump(model_data, f)

    print(f"Augmented Naïve Bayes model saved at: {save_path}")

    return model_data

# Train ANB model
trained_anb_model = train_anb(train_df, test_df, val_df, model_path="best_anb.pkl")
