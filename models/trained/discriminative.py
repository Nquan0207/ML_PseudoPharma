# import data
import numpy as np
train_df = np.load('../data/preprocessed/train_data.npz')
test_df = np.load('../data/preprocessed/test_data.npz')

X_train = train_df['x_train']
y_train = train_df['y_train']
X_test = test_df['x_test']
y_test = test_df['y_test']

# Logistic Regression using pretrain module
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

X_tr_np = X_train
y_tr_np = y_train
X_te_np = X_test
y_te_np = y_test

clf = LogisticRegression(max_iter=1000)
clf.fit(X_tr_np, y_tr_np)

print("============== Logistic Regression ==============")
y_pred = clf.predict(X_te_np)
print(classification_report(y_te_np, y_pred))

# Logistic Regression implementation
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        return self.linear(x)
    
input_dim = X_train_tensor.shape[1] 
num_classes = len(torch.unique(y_train_tensor))
model = LogisticRegressionModel(input_dim, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)

epochs = 500
model.train()
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    # print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")

model.eval()
with torch.no_grad():
    outputs = model(X_test_tensor)
    preds = outputs.argmax(dim=1)

print("============== Logistic Regression Implementation ==============")
y_true = y_test_tensor.cpu().numpy()
y_pred = preds.cpu().numpy()
print(classification_report(y_true, y_pred))

# Conditional Random Field
import numpy as np
import sklearn_crfsuite
from sklearn_crfsuite import metrics
from sklearn.model_selection import train_test_split

SEQ_LEN = 20

crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',                
    max_iterations=500,              
    all_possible_transitions=True   
)

def extract_features(vec):
    """Convert feature vector to dictionary for CRF """
    return {f'f{i}': vec[i] for i in range(len(vec))}

def create_sequences(X, y):
    X_seq = []
    y_seq = []
    for i in range(len(X)):
        features = {f'f{j}': str(X[i][j]) for j in range(X.shape[1])}
        X_seq.append([features]) 
        y_seq.append([str(y[i])]) 
    return X_seq, y_seq

X_train = np.array(X_train.tolist())
y_train = np.array(y_train.tolist())
X_test = np.array(X_test.tolist())
y_test = np.array(y_test.tolist())

X_train_seq, y_train_seq = create_sequences(X_train, y_train)
X_test_seq, y_test_seq = create_sequences(X_test, y_test)

crf.fit(X_train_seq, y_train_seq)
y_pred_seq = crf.predict(X_test_seq)

y_true_flat = [label for seq in y_test_seq for label in seq]
y_pred_flat = [label for seq in y_pred_seq for label in seq]

print("============== Conditional Random Field ==============")
print(metrics.flat_classification_report(y_true_flat, y_pred_flat, digits=4))
