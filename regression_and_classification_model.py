import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
file_path = r"G:\waveguide_ai\train_data_without_lengths.csv"
data = pd.read_csv(file_path)

# Extract features and target variables
features = ["iris_1", "iris_2", "iris_3", "iris_4"]
regression_target = "error"
classification_target = "Goal_Met"
X = data[features]
y_regression = data[regression_target]
data[classification_target] = data[classification_target].map({'Yes': 1, 'No': 0}).astype(int)
y_classification = data[classification_target]

# Normalize iris widths by waveguide width
waveguide_width = 15.8  # mm
X.loc[:, "iris_1"] /= waveguide_width
X.loc[:, "iris_2"] /= waveguide_width
X.loc[:, "iris_3"] /= waveguide_width
X.loc[:, "iris_4"] /= waveguide_width

# Apply MinMaxScaler to scale input features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Convert data to NumPy arrays
X_np = np.array(X_scaled)
y_reg_np = np.array(y_regression)
y_class_np = np.array(y_classification, dtype=np.int64)

# Split the data for both regression and classification
X_train, X_test, y_train_reg, y_test_reg, y_train_cls, y_test_cls = train_test_split(
    X_np, y_reg_np, y_class_np, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
X_test_tensor = torch.FloatTensor(X_test)
y_train_reg_tensor = torch.FloatTensor(y_train_reg).view(-1, 1)
y_test_reg_tensor = torch.FloatTensor(y_test_reg).view(-1, 1)
y_train_cls_tensor = torch.LongTensor(y_train_cls)
y_test_cls_tensor = torch.LongTensor(y_test_cls)

# Define Neural Network Model for Regression
class WaveguideRegressionNN(nn.Module):
    def __init__(self):
        super(WaveguideRegressionNN, self).__init__()
        self.fc1 = nn.Linear(len(features), 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)  # Ensure final output has a single value per sample
        return x
        

# Define Neural Network Model for Classification
class WaveguideClassificationNN(nn.Module):
    def __init__(self):
        super(WaveguideClassificationNN, self).__init__()
        self.fc1 = nn.Linear(len(features), 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 2)
        self.relu = nn.ReLU()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x
        

# Initialize models
regression_model = WaveguideRegressionNN()
classification_model = WaveguideClassificationNN()

# Loss functions and optimizers
regression_criterion = nn.MSELoss()
classification_criterion = nn.CrossEntropyLoss()
optimizer_reg = optim.Adam(regression_model.parameters(), lr=0.01)
optimizer_cls = optim.Adam(classification_model.parameters(), lr=0.01)

# Training Loops
for epoch in range(1000):
    # Train Regression Model
    optimizer_reg.zero_grad()
    outputs_reg = regression_model(X_train_tensor)
    loss_reg = regression_criterion(outputs_reg, y_train_reg_tensor)
    loss_reg.backward()
    optimizer_reg.step()
    
    # Train Classification Model
    optimizer_cls.zero_grad()
    outputs_cls = classification_model(X_train_tensor)
    loss_cls = classification_criterion(outputs_cls, y_train_cls_tensor)
    loss_cls.backward()
    optimizer_cls.step()

# Evaluate models
with torch.no_grad():
    y_pred_reg_test = regression_model(X_test_tensor).numpy()
    y_pred_cls_test = torch.argmax(classification_model(X_test_tensor), dim=1).numpy()

# Apply threshold to determine if goal is met in regression model
threshold = 1.1876
y_pred_reg_goal_met = ["Yes" if error < threshold else "No" for error in y_pred_reg_test.flatten()]

# Generate random test samples
random_designs = np.random.uniform(
    low=[7.86159, 5.325177, 4.660957, 4.558468],
    high=[8.689125, 5.885722, 5.151584, 5.038306],
    size=(500, 4)
)

# Normalize test samples by waveguide width
random_designs[:, 0] /= waveguide_width
random_designs[:, 1] /= waveguide_width
random_designs[:, 2] /= waveguide_width
random_designs[:, 3] /= waveguide_width

# Scale test samples
random_designs_df = pd.DataFrame(random_designs, columns=features)  # Convert to DataFrame
random_designs_scaled = scaler.transform(random_designs_df)
random_designs_tensor = torch.FloatTensor(random_designs_scaled)

# Predict error_in_dB for test samples
with torch.no_grad():
    predicted_errors = regression_model(random_designs_tensor).numpy()

# Apply threshold to determine if goal is met for test samples
predicted_goal_met = ["Yes" if error < threshold else "No" for error in predicted_errors.flatten()]

# Display test sample predictions
for i, (design, pred_error, goal) in enumerate(zip(random_designs, predicted_errors, predicted_goal_met)):
    if goal == 'Yes':
        print(f"Test Sample {i+1}: {design * waveguide_width} -> Predicted Error = {pred_error[0]:.2f}, Goal Met: {goal}")
