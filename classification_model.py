import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

# Load data
df = pd.read_csv(r"G:\waveguide_ai\train_data_without_lengths.csv")

# Define features and target variable
X = df[['iris_1', 'iris_2', 'iris_3', 'iris_4']]

#y = (df['error'] < 1.17).astype(int)  # Binary classification: 1 if error < threshold, else 0

target_column = [col for col in df.columns if 'Goal_Met' in col][0]
y = df[target_column].map({'Yes': 1, 'No': 0})  # Binary classification: 1 if error < threshold, else 0

# Normalize iris widths by the max waveguide width
waveguide_width = 15.7
X = X / waveguide_width

# Scale features using MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Convert data to NumPy arrays
X_np, y_np = np.array(X_scaled, dtype=np.float32), np.array(y, dtype=np.float32).reshape(-1, 1)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_np, y_np, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train)
X_test_tensor = torch.tensor(X_test)

#y_train_tensor = torch.tensor(y_train)
#y_test_tensor = torch.tensor(y_test)

y_train_tensor = torch.tensor(y_train.squeeze(), dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.squeeze(), dtype=torch.float32)

# Define Neural Network Model for Classification
class ClassificationNN(nn.Module):
    def __init__(self):
        super(ClassificationNN, self).__init__()
        self.fc1 = nn.Linear(4, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return self.sigmoid(x)

# Initialize model, loss function, and optimizer
model = ClassificationNN()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    y_pred = model(X_train_tensor).squeeze()
    loss = criterion(y_pred, y_train_tensor)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# Evaluate the model
model.eval()
y_pred_test = model(X_test_tensor).detach().numpy()
y_pred_test = (y_pred_test > 0.5).astype(int)

y_train_pred = model(X_train_tensor).detach().numpy()
y_train_pred = (y_train_pred > 0.5).astype(int)

train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_pred_test)
print(f"Train Accuracy: {train_acc * 100:.2f}%")
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# Load and preprocess new test data
def test_new_data(test_df):
    test_X = test_df[['iris_1', 'iris_2', 'iris_3', 'iris_4']]
    test_X = test_X / waveguide_width
    test_X_scaled = scaler.transform(test_X)
    test_X_tensor = torch.tensor(test_X_scaled, dtype=torch.float32)
    
    model.eval()
    
    test_predictions = model(test_X_tensor).detach().numpy()
    test_df['Predicted_Error'] = test_predictions
    test_df['Goal_Met'] = (test_predictions > 0.5).astype(int)
    
    return test_df

# Example usage
test_df = pd.read_csv(r"G:\waveguide_ai\test_data_without_lengths.csv")
results = test_new_data(test_df)

test_df_sorted = results[results['Goal_Met'] == 1].sort_values(by='Predicted_Error')
test_df_sorted.to_excel("classification_results.xlsx", index=False)
print("Results saved to classification_results.xlsx")
