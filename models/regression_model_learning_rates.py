import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load data
df = pd.read_csv(r"/home/raavi/research/ai/train_data_without_lengths.csv")

# Define features and target variable
X = df[['iris_1', 'iris_2', 'iris_3', 'iris_4']]
y = df[['error']]

# Normalize target variable
y_scaler = MinMaxScaler()
y_scaled = y_scaler.fit_transform(y)

# Normalize iris widths by the max waveguide width
waveguide_width = 15.7
X = X / waveguide_width

# Scale features using MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Convert data to NumPy arrays
X_np, y_np = np.array(X_scaled, dtype=np.float32), np.array(y_scaled, dtype=np.float32).reshape(-1, 1)

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
def test_new_data(test_df, threshold=1.17):
    test_X = test_df[['iris_1', 'iris_2', 'iris_3', 'iris_4']]
    test_X = test_X / waveguide_width
    test_X_scaled = scaler.transform(test_X)
    test_X_tensor = torch.tensor(test_X_scaled, dtype=torch.float32).to(device)
    
    model = RegressionNN().to(device)
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()
    
    test_predictions = y_scaler.inverse_transform(model(test_X_tensor).cpu().detach().numpy())
    test_df['Predicted_Error'] = test_predictions
    test_df['Goal_Met'] = test_df['Predicted_Error'] < threshold
    
    return test_df

# Example usage
test_df = pd.read_csv("test_data_without_lengths.csv")
results = test_new_data(test_df)
test_df_sorted = test_df[test_df['Goal_Met']].sort_values(by='Predicted_Error')
test_df_sorted.to_excel("filtered_results.xlsx", index=False)
print("Results saved to filtered_results.xlsx")
