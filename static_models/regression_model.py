import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

# Load data
#df = pd.read_csv(r"G:\waveguide_ai\train_data_without_lengths.csv")
df = pd.read_csv(r"/home/raavi/research/ai/static_models/train_data/train_data_without_lengths.csv")

scaler = MinMaxScaler()

target = df[['error']].copy()
target['error_scaled'] = np.log1p(target['error']) 
target_scaled = pd.DataFrame(scaler.fit_transform(target[['error_scaled']]), columns=['error_scaled'])

# Define features and target variable

norm_factor = 15.7
features = df[['iris_1', 'iris_2', 'iris_3', 'iris_4']]
features /= norm_factor
features_scaled = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)

# Convert data to NumPy arrays
features_np, target_np = np.array(features_scaled, dtype=np.float32), np.array(target_scaled, dtype=np.float32).reshape(-1, 1)

# Split data into training and testing sets
features_train, features_test, target_train, target_test = train_test_split(features_np, target_np, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
features_train_tensor = torch.tensor(features_train)
features_test_tensor = torch.tensor(features_test)

target_train_tensor = torch.tensor(target_train)
target_test_tensor = torch.tensor(target_test)

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
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 1000
for epoch in range(epochs):
    model.train()    
    optimizer.zero_grad()    
    target_prediction = model(features_train_tensor)    
    loss = criterion(target_prediction, target_train_tensor)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# Evaluate the model
model.eval()

target_train_prediction = model(features_train_tensor).detach().numpy()
train_rmse = np.sqrt(mean_squared_error(target_train, target_train_prediction))
print(f"Train RMSE: {train_rmse}")

target_test_prediction = model(features_test_tensor).detach().numpy()
test_rmse = np.sqrt(mean_squared_error(target_test, target_test_prediction))
print(f"Test RMSE: {test_rmse}")

train_r2 = r2_score(target_train, target_train_prediction)
print(f"Train R^2 Score: {train_r2}")

test_r2 = r2_score(target_test, target_test_prediction)
print(f"Test R^2 Score: {test_r2}")

# Load and preprocess new test data
def test_user_data(user_data_test_results, threshold=0.75):
    user_data = user_data_test_results[['iris_1', 'iris_2', 'iris_3', 'iris_4']]
    user_data = user_data / norm_factor
    user_data_scaled = scaler.transform(user_data)    
    user_data_tensor = torch.tensor(user_data_scaled, dtype=torch.float32)
    
    model.eval()

    user_data_predictions = model(user_data_tensor).detach().numpy()    

    user_data_test_results['Predicted_Error'] = user_data_predictions
    user_data_test_results['Goal_Met'] = user_data_test_results['Predicted_Error'] < threshold
        
    return user_data_test_results

# Example usage
user_data = pd.read_csv(r"/home/raavi/research/ai/static_models/train_data/test_data_without_lengths.csv")
results = test_user_data(user_data)
#print(results[results['Goal_Met']].sort_values(by='Predicted_Error').to_string(index=False))

sorted_results = results[results['Goal_Met']].sort_values(by='Predicted_Error')
sorted_results.to_excel("regression_results.xlsx", index=False)
print("Results saved to regression_results.xlsx")

