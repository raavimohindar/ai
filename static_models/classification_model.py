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
features = df[['iris_1', 'iris_2', 'iris_3', 'iris_4']]
target_column = [col for col in df.columns if 'Goal_Met' in col][0]
target = df[target_column].map({'Yes': 1, 'No': 0})  # Binary classification: 1 if error < threshold, else 0

# Normalize iris widths by the max waveguide width
norm_factor = 15.7
features = features / norm_factor

# Scale features using MinMaxScaler
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

# Convert data to NumPy arrays
features_np, target_np = np.array(features_scaled, dtype=np.float32), np.array(target, dtype=np.float32).reshape(-1, 1)

# Split data into training and testing sets
features_train, features_test, target_train, target_test = train_test_split(features_np, target_np, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
features_train_tensor = torch.tensor(features_train)
features_test_tensor = torch.tensor(features_test)

target_train_tensor = torch.tensor(target_train.squeeze(), dtype=torch.float32)
target_test_tensor = torch.tensor(target_test.squeeze(), dtype=torch.float32)

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
epochs = 1000
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    target_predicition = model(features_train_tensor).squeeze()
    loss = criterion(target_predicition, target_train_tensor)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# Evaluate the model
model.eval()

target_train_prediction = model(features_train_tensor).detach().numpy()
target_train_prediction = (target_train_prediction > 0.5).astype(int)

target_test_prediction = model(features_test_tensor).detach().numpy()
target_test_prediction = (target_test_prediction > 0.5).astype(int)

train_accuracy = accuracy_score(target_train, target_train_prediction)
test_accuracy = accuracy_score(target_test, target_test_prediction)

print(f"Train Accuracy: {train_accuracy * 100:.2f}%")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Load and preprocess new test data
def test_user_data(user_data_test_results):
    user_data = user_data_test_results[['iris_1', 'iris_2', 'iris_3', 'iris_4']]
    user_data = user_data / norm_factor
    user_data_scaled = scaler.transform(user_data)
    user_data_tensor = torch.tensor(user_data_scaled, dtype=torch.float32)
    
    model.eval()
    
    test_predictions = model(user_data_tensor).detach().numpy()
    
    user_data_test_results['Predicted_Error'] = test_predictions
    user_data_test_results['Goal_Met'] = (test_predictions > 0.5).astype(int)
    
    return user_data_test_results

# Example usage
user_data = pd.read_csv(r"G:\waveguide_ai\test_data_without_lengths.csv")
results = test_user_data(user_data)

sorted_results = results[results['Goal_Met'] == 1].sort_values(by='Predicted_Error')
sorted_results.to_excel("classification_results.xlsx", index=False)
print("Results saved to classification_results.xlsx")
