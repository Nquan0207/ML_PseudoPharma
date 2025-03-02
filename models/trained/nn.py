from train_set import train, test
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import random


type(train["label"].values.tolist()[0])


print(len(train))
print(len(test))

X_train = train.drop(columns=["label"]).values
y_train = train["label"].values
X_test = test.drop(columns=["label"]).values
y_test = test["label"].values

num_classes = len(set(y_train))

if num_classes > 2:
    y_train = np.eye(num_classes)[train["label"].values.tolist()]
    y_test = np.eye(num_classes)[test["label"].values.tolist()]
else:
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

class NeuralNetwork:
    def __init__(
        self,
        input_size,
        hidden1,
        hidden2,
        hidden3,
        output_size,
        learning_rate=0.01,
        seed=None,
    ):
        self.lr = learning_rate
        if seed is not None:
            np.random.seed(seed)

        self.W1 = np.random.randn(input_size, hidden1) * 0.1
        self.b1 = np.zeros((1, hidden1))
        self.W2 = np.random.randn(hidden1, hidden2) * 0.1
        self.b2 = np.zeros((1, hidden2))
        self.W3 = np.random.randn(hidden2, hidden3) * 0.1
        self.b3 = np.zeros((1, hidden3))
        self.W4 = np.random.randn(hidden3, output_size) * 0.1
        self.b4 = np.zeros((1, output_size))

    def activation(self, x):
        """ReLU Activation Function"""
        return np.maximum(0, x)

    def activation_derivative(self, x):
        """Derivative of ReLU"""
        return (x > 0).astype(float)

    def softmax(self, x):
        """Softmax activation for multi-class classification"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def sigmoid(self, x):
        """Sigmoid activation for binary classification"""
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        """Derivative of sigmoid"""
        return x * (1 - x)

    def forward(self, X):
        """Forward propagation"""
        self.A1 = self.activation(np.dot(X, self.W1) + self.b1)
        self.A2 = self.activation(np.dot(self.A1, self.W2) + self.b2)
        self.A3 = self.activation(np.dot(self.A2, self.W3) + self.b3)
        self.output = np.dot(self.A3, self.W4) + self.b4

        # Apply activation based on problem type
        if num_classes > 2:
            self.output = self.softmax(self.output)
        else:
            self.output = self.softmax(self.output)

        return self.output

    def backward(self, X, y):
        """Backpropagation"""
        m = X.shape[0]  # Number of samples

        # Compute error
        if num_classes > 2:
            error = self.output - y
        # else:
        #     error = self.output - y  # For binary classification

        # Backpropagation)
        dW4 = np.dot(self.A3.T, error) / m
        db4 = np.sum(error, axis=0, keepdims=True) / m

        dA3 = np.dot(error, self.W4.T) * self.activation_derivative(self.A3)
        dW3 = np.dot(self.A2.T, dA3) / m
        db3 = np.sum(dA3, axis=0, keepdims=True) / m

        dA2 = np.dot(dA3, self.W3.T) * self.activation_derivative(self.A2)
        dW2 = np.dot(self.A1.T, dA2) / m
        db2 = np.sum(dA2, axis=0, keepdims=True) / m

        dA1 = np.dot(dA2, self.W2.T) * self.activation_derivative(self.A1)
        dW1 = np.dot(X.T, dA1) / m
        db1 = np.sum(dA1, axis=0, keepdims=True) / m

        # Update weights and biases (Delta Rule)
        self.W4 -= self.lr * dW4
        self.b4 -= self.lr * db4
        self.W3 -= self.lr * dW3
        self.b3 -= self.lr * db3
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def train(self, X, y, epochs=100):
        """Train the model using backpropagation"""
        for epoch in range(epochs):
            self.forward(X)
            self.backward(X, y)

            loss = np.mean(-y * np.log(self.output + 1e-9)) 
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")
        print("Final loss: ", loss)


nn = NeuralNetwork(
    input_size=X_train.shape[1],
    hidden1=128,
    hidden2=64,
    hidden3=32,
    output_size=num_classes,
    learning_rate=0.01,
    seed=62,
)
nn.train(X_train, y_train, epochs=100)

y_pred = nn.forward(X_test)
predictions = (
    np.argmax(y_pred, axis=1) if num_classes > 2 else (y_pred > 0.5).astype(int)
)
predictions = np.eye(num_classes)[predictions]

accuracy = np.mean(predictions.flatten() == y_test.flatten())
print(f"Test Accuracy: {accuracy:.6f}")
