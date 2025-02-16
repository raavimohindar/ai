import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from torch.optim.lr_scheduler import ReduceLROnPlateau
import joblib

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load data
df = pd.read_csv(r"/home/raavi/research/ai/train_data_with_S11.csv")

# Define features and target variables
X = df[['iris_1', 'iris_2', 'iris_3', 'iris_4'] + [f'S_11_{i}' for i in range(100)]]
y = df[['error'] + [f'S_11_{i}' for i in range(100)]]

# ✅ Apply Log Transformation to 'error' (log(1+error) to avoid log(0))
y['error'] = np.log1p(y['error'])

# Normalize target variable
y_scaler = MinMaxScaler()
y_scaled = y_scaler.fit_transform(y)

# Save the target scaler for future inverse transformation
joblib.dump(y_scaler, "y_scaler.pkl")

# Normalize iris widths by the max waveguide width
waveguide_width = 15.7
X.iloc[:, :4] = X.iloc[:, :4] / waveguide_width

# Scale features using MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Save the feature scaler
joblib.dump(scaler, "feature_scaler.pkl")

# Convert data to NumPy arrays
X_np, y_np = np.array(X_scaled, dtype=np.float32), np.array(y_scaled, dtype=np.float32)

# Split data into training, validation, and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(X_np, y_np, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

# Define Neural Network Model for Regression
class RegressionNN(nn.Module):
    def __init__(self):
        super(RegressionNN, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 100)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Hyperparameter tuning
learning_rates = [0.001, 0.005, 0.01]
batch_sizes = [16, 32, 64]
best_rmse = float('inf')
best_hyperparams = {}
best_model_state = None

for lr in learning_rates:
    for batch_size in batch_sizes:
        model = RegressionNN().to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)

        train_loader = torch.utils.data.DataLoader(list(zip(X_train_tensor, y_train_tensor)), batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(list(zip(X_val_tensor, y_val_tensor)), batch_size=batch_size, shuffle=False)
        
        # Training loop with early stopping
        best_val_loss = float('inf')
        patience, patience_counter = 20, 0
        
        for epoch in range(100):
            model.train()
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                y_pred = model(batch_X)
                loss = criterion(y_pred, batch_y)
                loss.backward()
                optimizer.step()
            
            # Validation step
            model.eval()
            with torch.no_grad():
                val_preds = model(X_val_tensor)
                val_loss = criterion(val_preds, y_val_tensor).item()
                scheduler.step(val_loss)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch}")
                        break
        
        # Evaluate on test data
        model.eval()
        y_pred_test = y_scaler.inverse_transform(model(X_test_tensor).cpu().detach().numpy())
        
        # ✅ Undo log transform for error predictions
        y_pred_test[:, 0] = np.expm1(y_pred_test[:, 0])  # First column is 'error'

        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        
        if test_rmse < best_rmse:
            best_rmse = test_rmse
            best_hyperparams = {'learning_rate': lr, 'batch_size': batch_size}
            best_model_state = model.state_dict()

# Save the best model
if best_model_state:
    torch.save(best_model_state, "best_model.pth")
    print(f"Best hyperparameters: {best_hyperparams}, Best RMSE: {best_rmse}")

# Load and preprocess new test data
def predict_S11(test_df):
    test_X = test_df[['iris_1', 'iris_2', 'iris_3', 'iris_4']]
    
    # Normalize iris dimensions
    test_X = test_X / waveguide_width

    # Load the saved scaler and apply transformation
    feature_scaler = joblib.load("feature_scaler.pkl")
    test_X_scaled = feature_scaler.transform(test_X)

    test_X_tensor = torch.tensor(test_X_scaled, dtype=torch.float32).to(device)

    # Load the best model
    model = RegressionNN().to(device)
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()

    # Predict S_11
    S11_predictions = model(test_X_tensor).cpu().detach().numpy()

    # Inverse transform predictions
    y_scaler = joblib.load("y_scaler.pkl")
    S11_predictions = y_scaler.inverse_transform(S11_predictions)

    # ✅ Undo log transform for predicted error
    S11_predictions[:, 0] = np.expm1(S11_predictions[:, 0])

    # Store results in DataFrame
    for i in range(100):
        test_df[f'Predicted_S_11_{i}'] = S11_predictions[:, i]

    return test_df

# Example usage
test_df = pd.read_csv("test_data_without_lengths.csv")
results = predict_S11(test_df)
results.to_excel("predicted_S11_results.xlsx", index=False)
print("Results saved to predicted_S11_results.xlsx")
