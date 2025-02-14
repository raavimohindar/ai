import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

# Load data
df = pd.read_csv(r"G:\waveguide_ai\train_data_without_lengths.csv")

#waveguide width
waveguide_width = 15.8  # mm

# Define features and target variable
X = df[['iris_1', 'iris_2', 'iris_3', 'iris_4']]
y = df['error']

# Normalize iris widths by the max waveguide width
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
y_train_tensor = torch.tensor(y_train)
y_test_tensor = torch.tensor(y_test)

# Define Neural Network Model for Regression
class RegressionNN(nn.Module):
    def __init__(self):
        super(RegressionNN, self).__init__()
        self.fc1 = nn.Linear(4, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize model, loss function, and optimizer
model = RegressionNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    y_pred = model(X_train_tensor)
    loss = criterion(y_pred, y_train_tensor)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# Evaluate the model
model.eval()
y_pred_test = model(X_test_tensor).detach().numpy()
rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
r2 = r2_score(y_test, y_pred_test)

print(f"RMSE: {rmse}")
print(f"R^2 Score: {r2}")

# Load and preprocess new test data
def test_new_data(test_df, threshold=2.5):
    test_X = test_df[['iris_1', 'iris_2', 'iris_3', 'iris_4']]
    test_X = test_X / waveguide_width
    test_X_scaled = scaler.transform(test_X)
    test_X_tensor = torch.tensor(test_X_scaled, dtype=torch.float32)
    
    model.eval()
    test_predictions = model(test_X_tensor).detach().numpy()
    
    test_df['Predicted_Error'] = test_predictions
    test_df['Goal_Met'] = test_df['Predicted_Error'] < threshold
    
    return test_df

# Example usage
test_df = pd.read_csv(r"G:\waveguide_ai\test_data_without_lengths.csv")
results = test_new_data(test_df)
print(results)
