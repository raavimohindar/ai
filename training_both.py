import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
file_path = r"G:\training_data\real_train_processed.csv"
#file_path = r"/home/raavi/research/ai/train_data_without_len.csv"
data = pd.read_csv(file_path)

# Define features and target variables
features = ["iris_1", "iris_2", "iris_3", "iris_4", "len_1", "len_2", "len_3"]
regression_target = "error"
classification_target = "Goal_Met"

# Extract features and targets
X = data[features]
y_regression = data[regression_target]
data[classification_target] = data[classification_target].map({'Yes': 1, 'No': 0}).astype(int)
y_classification = data[classification_target]

# Normalize features using waveguide dimensions and calculated max_length
waveguide_width = 15.8  # mm
thickness = 1  # mm for all irises

# Compute max_length dynamically
max_length = (thickness + thickness + thickness + thickness) + (X['len_1'] + X['len_2'] + X['len_3'])

# Normalize iris widths by waveguide width
X.loc[:, "iris_1"] /= waveguide_width
X.loc[:, "iris_2"] /= waveguide_width
X.loc[:, "iris_3"] /= waveguide_width
X.loc[:, "iris_4"] /= waveguide_width

# Normalize distances between irises by max_length
X.loc[:, "len_1"] /= max_length
X.loc[:, "len_2"] /= max_length
X.loc[:, "len_3"] /= max_length

# Apply MinMaxScaler to input features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Convert to NumPy arrays
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

# Define Regression Neural Network
class WaveguideRegressionNN(nn.Module):
    def __init__(self):
        super(WaveguideRegressionNN, self).__init__()
        self.fc1 = nn.Linear(len(features), 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Define Classification Neural Network
class WaveguideClassificationNN(nn.Module):
    def __init__(self):
        super(WaveguideClassificationNN, self).__init__()
        self.fc1 = nn.Linear(len(features), 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize models
regression_model = WaveguideRegressionNN()
classification_model = WaveguideClassificationNN()

# Loss functions and optimizers
regression_criterion = nn.MSELoss()
classification_criterion = nn.CrossEntropyLoss()
optimizer_reg = optim.Adam(regression_model.parameters(), lr=0.01)
optimizer_cls = optim.Adam(classification_model.parameters(), lr=0.01)

# Generate 5 random test samples within known parameter ranges
random_designs = np.random.uniform(
    low=[7.86159, 5.325177, 4.660957, 4.558468, 8, 9.673862, 10],
    high=[8.689125, 5.885722, 5.151584, 5.038306, 12, 10.692163, 11],
    size=(125, 7)
)
random_designs_df = pd.DataFrame(random_designs, columns=features)
random_designs_scaled = scaler.transform(random_designs_df)
random_designs_tensor = torch.FloatTensor(random_designs_scaled)

# Predict error_in_dB for the generated test samples
with torch.no_grad():
    predicted_errors = regression_model(random_designs_tensor).numpy()

# Predict classification (Goal_met) for the generated test samples
with torch.no_grad():
    predicted_classes = torch.argmax(classification_model(random_designs_tensor), dim=1).numpy()

# Display predictions
for i, (design, pred_reg, pred_cls) in enumerate(zip(random_designs, predicted_errors, predicted_classes)):
    print(f"Test Sample {i+1}: {design} -> Predicted Error: {pred_reg[0]}, Predicted Goal Met: {pred_cls}")
