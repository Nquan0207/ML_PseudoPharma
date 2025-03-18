import torch
import networkx as nx
import matplotlib.pyplot as plt

class BayesianNetwork:
    def __init__(self, edges, device="cpu"):
        """
        Bayesian Network with GPU support.

        :param edges: List of (parent, child) edges representing the DAG.
        :param device: 'cpu' or 'cuda' for GPU computation.
        """
        self.device = torch.device(device)
        self.graph = nx.DiGraph()
        self.graph.add_edges_from(edges)
        self.nodes = list(self.graph.nodes())
        self.parents = {node: list(self.graph.predecessors(node)) for node in self.nodes}
        self.cpts = {}  # Conditional Probability Tables (stored as tensors)

    def visualize(self):
        """Visualizes the Bayesian Network structure."""
        plt.figure(figsize=(8, 6))
        nx.draw(self.graph, with_labels=True, node_size=3000, node_color="lightblue", edge_color="gray", font_size=12)
        plt.title("Bayesian Network Structure")
        plt.show()

import torch

class OnlineBayesianEstimator:
    def __init__(self, model, alpha=1):
        """
        Online Bayesian Parameter Estimation using PyTorch (GPU accelerated).
        
        :param model: BayesianNetwork object.
        :param alpha: Laplace smoothing parameter.
        """
        self.model = model
        self.alpha = alpha  # Laplace smoothing factor
        self.counts = {}  # Stores accumulated counts for incremental learning

    def update_counts(self, batch_data):
        """
        Updates counts for each node based on new batch data.
        
        :param batch_data: Pandas DataFrame containing the new batch.
        """
        for node in self.model.nodes:
            parents = self.model.parents[node]
            if parents:
                grouped_data = batch_data.pivot_table(index=parents, columns=node, aggfunc='size', fill_value=0)
            else:
                grouped_data = batch_data[node].value_counts().to_frame().T

            if node not in self.counts:
                self.counts[node] = grouped_data
            else:
                self.counts[node] += grouped_data  # Accumulate counts across batches

    def estimate_cpds(self):
        """
        Recomputes CPDs based on updated counts.
        """
        cpts = {}
        for node, count_matrix in self.counts.items():
            smoothed_counts = count_matrix + self.alpha
            cpt_tensor = torch.tensor(smoothed_counts.div(smoothed_counts.sum(axis=1), axis=0).values,
                                      dtype=torch.float32, device=self.model.device)
            cpts[node] = cpt_tensor

        self.model.cpts = cpts  # Update model CPDs

class BayesianInference:
    def __init__(self, model):
        """
        Inference in a Bayesian Network using PyTorch (GPU Accelerated).
        
        :param model: BayesianNetwork object.
        """
        self.model = model

    def compute_posterior(self, evidence):
        """
        Computes P(Label | Features) on GPU.

        :param evidence: Dictionary of observed values {feature_name: value}
        :return: Dictionary {label_value: probability}
        """
        labels = list(range(self.model.cpts['label'].shape[1]))  # Get possible label values
        posterior_probs = {}

        for label_idx in labels:
            prob = self.model.cpts['label'][0, label_idx].item()  # Extract single probability from tensor

            for feature, value in evidence.items():
                if feature in self.model.cpts:
                    feature_cpt = self.model.cpts[feature]
                    if value in range(feature_cpt.shape[0]):  # Ensure value is within valid range
                        prob *= feature_cpt[value, min(label_idx, feature_cpt.shape[1] - 1)].item()
                    else:
                        prob *= 1e-6  # Small probability for unseen values

            posterior_probs[label_idx] = prob

        # Normalize probabilities (sum-to-1 constraint)
        total = sum(posterior_probs.values())
        if total > 0:
            for label_idx in posterior_probs:
                posterior_probs[label_idx] /= total

        return posterior_probs

def predict(bn_model, test_data):
    """
    Performs batch classification using Bayesian Network inference (GPU Accelerated).

    :param bn_model: Trained BayesianNetwork object.
    :param test_data: Pandas DataFrame of test instances.
    :return: Predicted labels (PyTorch tensor).
    """
    inference = BayesianInference(bn_model)
    predictions = []

    for _, row in test_data.iterrows():
        evidence = row.to_dict()
        del evidence['label']  # Remove true label (to predict it)
        posterior_probs = inference.compute_posterior(evidence)
        predicted_label = max(posterior_probs, key=posterior_probs.get)  # Argmax P(L | F)
        predictions.append(predicted_label)

    return torch.tensor(predictions, dtype=torch.int64, device=bn_model.device)

import medmnist
from medmnist import PathMNIST
import torch
import torchvision.transforms as transforms
from sklearn.decomposition import PCA
from sklearn.preprocessing import KBinsDiscretizer
import numpy as np
import pandas as pd

transform = transforms.Compose([transforms.ToTensor()])
train_dataset = PathMNIST(split='train', download=True, transform=transform)
val_dataset = PathMNIST(split='val', download=True, transform=transform)
test_dataset = PathMNIST(split='test', download=True, transform=transform)

# Convert dataset into a structured list
def preprocess_dataset(dataset, n_components=20, bins=5):
    image_list = []
    label_list = []

    for img_tensor, label_array in dataset:
        # Convert torch.Tensor to numpy.ndarray and flatten
        img_np = img_tensor.numpy().reshape(-1)  # Shape: (3*28*28,)
        
        # Extract label from numpy array
        label = label_array[0]  # Convert from [0] to scalar
        
        image_list.append(img_np)
        label_list.append(label)
    
    # Convert list to numpy array
    images_np = np.array(image_list)
    labels_np = np.array(label_list)

    # Apply PCA to reduce dimensions
    pca = PCA(n_components=n_components)
    reduced_features = pca.fit_transform(images_np)

    # Convert PCA features and labels into DataFrame
    df = pd.DataFrame(reduced_features, columns=[f'feature_{i}' for i in range(n_components)])
    df['label'] = labels_np  # Add label column

    # Discretize features
    discretizer = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='uniform')
    df.iloc[:, :-1] = discretizer.fit_transform(df.iloc[:, :-1])  # Only discretize feature columns

    # Convert to integers (Fixes float output from KBinsDiscretizer)
    df = df.astype(int)

    return df

# Apply preprocessing to training, validation, and test datasets
train_df = preprocess_dataset(train_dataset)
val_df = preprocess_dataset(val_dataset)
test_df = preprocess_dataset(test_dataset)

print(train_df.shape)
print(train_df.describe())
print(train_df.head())

def online_train_and_validate(train_df, val_df, batch_size=5000, model_path="best_bayesian_network.pkl", device="cuda"):
    """
    Trains a Bayesian Network using online learning (batch updates).
    
    Saves the best model based on validation F1-score.
    
    :param train_df: Training dataset (Pandas DataFrame).
    :param val_df: Validation dataset (Pandas DataFrame).
    :param batch_size: Number of samples per batch.
    :param model_path: Path to save the best model.
    :param device: "cuda" for GPU or "cpu".
    """
    # Initialize Bayesian Network structure
    edges = [(f'feature_{i}', 'label') for i in range(20)]
    bn_model = BayesianNetwork(edges, device=device)

    # Initialize Online Bayesian Estimator
    estimator = OnlineBayesianEstimator(bn_model, alpha=1)

    # Shuffle dataset before splitting into batches
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    num_batches = len(train_df) // batch_size + 1  # Number of batches
    batches = np.array_split(train_df, num_batches)

    # Online Training: Process batches sequentially
    for i, batch in enumerate(batches):
        print(f"Training on batch {i+1}/{num_batches} with {len(batch)} samples...")
        estimator.update_counts(batch)  # Incrementally update counts

    # Estimate CPDs after accumulating counts
    estimator.estimate_cpds()
    print("✅ Finished online training.")

    # Run inference on validation set
    val_predictions = predict(bn_model, val_df).cpu().numpy()  # Move back to CPU for evaluation

    # Compute metrics
    y_true = val_df['label'].values
    y_pred = val_predictions

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    print(f"Validation Metrics:")
    print(f" - Accuracy:  {accuracy * 100:.2f}%")
    print(f" - Precision: {precision:.4f}")
    print(f" - Recall:    {recall:.4f}")
    print(f" - F1 Score:  {f1:.4f}")

    # Save the best model based on F1-score
    best_model = {"model": bn_model, "f1_score": f1}
    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)

    print(f"✅ Best model saved at: {model_path}")

    return bn_model


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import pickle


# Define BayesianNetwork and OnlineBayesianEstimator classes (same as before)

# Helper function for hyperparameter tuning
def hyperparameter_tuning(train_df, val_df, device="cuda", n_splits=5, alpha_values=[0.1, 1, 10], batch_sizes=[1000, 5000, 10000]):
    """
    Perform hyperparameter tuning with cross-validation.

    :param train_df: Training dataset (Pandas DataFrame).
    :param val_df: Validation dataset (Pandas DataFrame).
    :param device: Device to run the model on ('cpu' or 'cuda').
    :param n_splits: Number of splits for cross-validation.
    :param alpha_values: List of alpha values to test.
    :param batch_sizes: List of batch sizes to test.
    :return: Best hyperparameters and model.
    """
    best_f1 = -1
    best_params = None
    best_model = None
    
    # K-Fold Cross-Validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    for alpha in alpha_values:
        for batch_size in batch_sizes:
            print(f"Testing alpha={alpha}, batch_size={batch_size}...")
            
            fold_f1_scores = []
            
            for train_index, val_index in kf.split(train_df):
                train_fold = train_df.iloc[train_index]
                val_fold = train_df.iloc[val_index]
                
                # Initialize Bayesian Network structure
                edges = [(f'feature_{i}', 'label') for i in range(20)]
                bn_model = BayesianNetwork(edges, device=device)

                # Initialize Online Bayesian Estimator
                estimator = OnlineBayesianEstimator(bn_model, alpha=alpha)

                # Shuffle dataset before splitting into batches
                train_fold = train_fold.sample(frac=1, random_state=42).reset_index(drop=True)
                num_batches = len(train_fold) // batch_size + 1  # Number of batches
                batches = np.array_split(train_fold, num_batches)

                # Online Training: Process batches sequentially
                for i, batch in enumerate(batches):
                    estimator.update_counts(batch)  # Incrementally update counts
                
                # Estimate CPDs after accumulating counts
                estimator.estimate_cpds()

                # Run inference on validation set
                val_predictions = predict(bn_model, val_fold).cpu().numpy()  # Move back to CPU for evaluation

                # Compute metrics
                y_true = val_fold['label'].values
                y_pred = val_predictions

                f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
                fold_f1_scores.append(f1)
            
            avg_f1 = np.mean(fold_f1_scores)
            print(f"Average F1 score for alpha={alpha}, batch_size={batch_size}: {avg_f1:.4f}")

            # Update best model if necessary
            if avg_f1 > best_f1:
                best_f1 = avg_f1
                best_params = {'alpha': alpha, 'batch_size': batch_size}
                best_model = bn_model
    
    print(f"Best hyperparameters found: {best_params}")
    return best_model, best_params

import warnings
warnings.filterwarnings('ignore')

best_model, best_params = hyperparameter_tuning(train_df, val_df, device="cuda", n_splits=5)