import os
import pickle
import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

class HiddenMarkovModelTrainer:
    def __init__(self):
        self.hmm_models = None  # Store trained HMM models

    def train_hmm(self, train_df, test_df, val_df, n_components_range=[2, 3, 4], random_state=42, model_path="hmm_models.pkl"):
        """
        Trains a Hidden Markov Model (HMM) for each class in the dataset with cross-validation and hyperparameter tuning.

        Args:
            train_df: Training dataset (Pandas DataFrame).
            test_df: Testing dataset (Pandas DataFrame).
            val_df: Validation dataset (Pandas DataFrame).
            n_components_range: List of possible numbers of hidden states (components) for hyperparameter tuning.
            random_state: Random seed for reproducibility.
            model_path: Path to save the trained HMM models.
            
        Returns:
            A dictionary containing the trained HMM models.
        """
        print("Starting Hidden Markov Model (HMM) training...")

        # Extract features and labels
        X_train, y_train = train_df.iloc[:, :-1], train_df.iloc[:, -1]
        X_test, y_test = test_df.iloc[:, :-1], test_df.iloc[:, -1]
        X_val, y_val = val_df.iloc[:, :-1], val_df.iloc[:, -1]

        # Normalize the feature data using StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        X_val_scaled = scaler.transform(X_val)

        # Perform K-fold cross-validation for hyperparameter tuning
        kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
        best_hmm_models = {}
        best_f1 = -1  # Initialize best F1 score as a very low value

        # Iterate over possible n_components for hyperparameter tuning
        for n_components in n_components_range:
            hmm_models = {}
            avg_f1_score = 0  # Average F1 score for this n_components across folds

            for train_idx, val_idx in kf.split(X_train_scaled):
                X_train_fold, X_val_fold = X_train_scaled[train_idx], X_train_scaled[val_idx]
                y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

                # Train HMM for each class
                for label in np.unique(y_train_fold):
                    X_label = X_train_fold[y_train_fold == label]

                    if len(X_label) < n_components:
                        print(f"Warning: Not enough samples ({len(X_label)}) for label {label} with {n_components} components")
                        continue

                    model = hmm.GaussianHMM(
                        n_components=n_components,
                        covariance_type="full",
                        n_iter=200,
                        random_state=random_state
                    )

                    try:
                        model.fit(X_label)  # Fit the model to the class-specific data
                        hmm_models[label] = model
                    except Exception as e:
                        print(f"Error training HMM for label {label}: {e}")

                # Evaluate on validation fold
                y_val_pred = self.predict_hmm(hmm_models, X_val_fold)
                f1 = f1_score(y_val_fold, y_val_pred, average='macro', zero_division=0)
                avg_f1_score += f1

            avg_f1_score /= kf.get_n_splits()
            print(f"Average F1 score for n_components={n_components}: {avg_f1_score:.4f}")

            # Update best model if necessary
            if avg_f1_score > best_f1:
                best_f1 = avg_f1_score
                best_hmm_models = hmm_models
                print(f"Updated best model with n_components={n_components}")

        # Save the best model
        save_path = os.path.join("models", model_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(best_hmm_models, f)

        print(f"HMM models saved at: {save_path}")
        self.hmm_models = best_hmm_models  # Store the best models for later use
        return best_hmm_models

    def predict_hmm(self, hmm_models, X_test_scaled):
        """
        Predicts labels using trained HMM models.

        Args:
            hmm_models: Dictionary of trained HMM models.
            X_test_scaled: Scaled feature matrix.

        Returns:
            List of predicted labels.
        """
        X_test_scaled = np.array(X_test_scaled)
        predictions = []

        for i in range(X_test_scaled.shape[0]):
            x = X_test_scaled[i].reshape(1, -1)  # Correct reshaping for single sample
            max_log_prob = float('-inf')
            best_label = None

            for label, model in hmm_models.items():
                try:
                    log_prob = model.score(x)
                    if log_prob > max_log_prob:
                        max_log_prob = log_prob
                        best_label = label
                except:
                    continue

            predictions.append(best_label if best_label is not None else -1)

        return np.array(predictions)

    def evaluate_hmm(self, hmm_models, X_test_scaled, y_test, X_val_scaled, y_val):
        """
        Evaluates the HMM model on test and validation sets.

        Args:
            hmm_models: Dictionary of trained HMM models.
            X_test_scaled: Scaled test features.
            y_test: Test labels.
            X_val_scaled: Scaled validation features.
            y_val: Validation labels.
        
        Returns:
            A dictionary of evaluation metrics.
        """
        # Predict on test set
        y_test_pred = self.predict_hmm(hmm_models, X_test_scaled)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(classification_report(y_test, y_test_pred))

        # Predict on validation set
        y_val_pred = self.predict_hmm(hmm_models, X_val_scaled)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        precision = precision_score(y_val, y_val_pred, average='macro', zero_division=0)
        recall = recall_score(y_val, y_val_pred, average='macro', zero_division=0)
        f1 = f1_score(y_val, y_val_pred, average='macro', zero_division=0)

        print("Validation Metrics:")
        print(f" - Accuracy:  {val_accuracy * 100:.2f}%")
        print(f" - Precision: {precision:.4f}")
        print(f" - Recall:    {recall:.4f}")
        print(f" - F1 Score:  {f1:.4f}")

        return {
            'test_accuracy': test_accuracy,
            'validation_accuracy': val_accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

# Initialize and train the HMM model
hmm_trainer = HiddenMarkovModelTrainer()
trained_hmm_models = hmm_trainer.train_hmm(train_df, test_df, val_df)

# Load and evaluate the model
hmm_metrics = hmm_trainer.evaluate_hmm(
    trained_hmm_models, 
    test_df.iloc[:, :-1].values, test_df.iloc[:, -1].values, 
    val_df.iloc[:, :-1].values, val_df.iloc[:, -1].values
)