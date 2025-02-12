import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
file_path = "G:\\training_data\\real_train_final.csv"
data = pd.read_csv(file_path)

# Extract features and target variable
features = ["iris_1", "iris_2", "iris_3", "iris_4", "len_1", "len_2", "len_3"]
target = "error_in_dB"
X = data[features]
y = data[target]

# Apply MinMaxScaler to scale input features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Convert data to NumPy arrays
X_np = np.array(X_scaled)
y_np = np.array(y)

# Split the data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_np, y_np, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
X_test_tensor = torch.FloatTensor(X_test)
y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)  # Reshape for PyTorch
y_test_tensor = torch.FloatTensor(y_test).view(-1, 1)

# Define Neural Network Model for Regression
class WaveguideRegressionNN(nn.Module):
    def __init__(self):
        super(WaveguideRegressionNN, self).__init__()
        self.fc1 = nn.Linear(len(features), 16)  # Input to hidden layer
        self.fc2 = nn.Linear(16, 16)  # Hidden layer
        self.fc3 = nn.Linear(16, 1)  # Output layer
        self.relu = nn.ReLU()  # Activation function

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # No activation in output layer for regression
        return x

# Initialize model, loss function, and optimizer
model = WaveguideRegressionNN()
criterion = nn.MSELoss()  # Mean Squared Error Loss
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training Loop
epochs = 500
losses = []

for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

# Plot the training loss curve
plt.plot(losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.show()

# Evaluate model on test data
with torch.no_grad():
    y_pred_test = model(X_test_tensor).numpy()

# Calculate Mean Squared Error on Test Set
mse = np.mean((y_pred_test - y_test.reshape(-1, 1)) ** 2)
print(f"Mean Squared Error on Test Set: {mse}")

# Define column names
column_names = ["iris_1", "iris_2", "iris_3", "iris_4", "l_1", "l_2", "l_3"]

# Generate 5 random test samples within known parameter ranges
random_designs = np.random.uniform(
    low=[7.86159, 5.325177, 4.660957, 4.558468, 8, 9.673862, 10],
    high=[8.689125, 5.885722, 5.151584, 5.038306, 12, 10.692163, 11],
    size=(5, 7)
)

# Convert random test samples to PyTorch tensor
random_designs_scaled = scaler.transform(random_designs)
random_designs_tensor = torch.FloatTensor(random_designs_scaled)

# Predict error_in_dB for the generated test samples
with torch.no_grad():
    predicted_errors = model(random_designs_tensor).numpy()

# Display predictions
for i, (design, pred) in enumerate(zip(random_designs, predicted_errors)):
    print(f"Test Sample {i+1}: {design} -> Predicted Error (dB): {pred[0]}")
