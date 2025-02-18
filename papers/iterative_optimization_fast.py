import torch
import numpy as np

import matplotlib
matplotlib.use('GTK3Agg')  # or 'GTK3Cairo'

import matplotlib.pyplot as plt
from scipy.interpolate import RBFInterpolator

# Generate initial data (simulated)
def generate_data(n_samples):
    x = np.random.uniform(-1, 1, (n_samples, 2))  
    y = np.sin(x[:, 0]) + np.cos(x[:, 1])  
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).view(-1, 1)

# Interpolation function using Radial Basis Function (RBF)
def interpolate_rbf(X_train, y_train, X_test):
    interpolator = RBFInterpolator(X_train.numpy(), y_train.numpy().flatten(), kernel='thin_plate_spline')
    y_pred = interpolator(X_test.numpy())
    return torch.tensor(y_pred, dtype=torch.float32).view(-1, 1)

# Function to visualize non-linear subregions
def plot_non_linear_subregions(X_test, y_pred, y_test, method, threshold=0.1):
    errors = torch.abs(y_pred - y_test).detach().numpy()
    
    plt.figure(figsize=(7, 6))
    plt.scatter(X_test[:, 0], X_test[:, 1], c=errors, cmap='plasma', s=50, edgecolors='k')
    plt.colorbar(label="Prediction Error")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.title(f"Non-Linear Subregions - {method} (Higher Error in Darker Regions)")
    plt.show()

# Initial Training Data
X_train, y_train = generate_data(10)
E_d = 0.01  # Desired error threshold

print("Running Equation Solving Process (Fast - RBF Interpolation)...")

for iteration in range(5):
    X_test, y_test = generate_data(100)
    y_pred = interpolate_rbf(X_train, y_train, X_test)

    # Compute Error
    criterion = torch.nn.MSELoss()
    E_test = criterion(y_pred, y_test).item()
    
    print(f"Iteration {iteration+1}, Testing Error: {E_test}")

    if E_test < E_d:
        print("Desired accuracy reached!")
        break  

    # Identify High-Error Regions
    high_error_indices = (torch.abs(y_pred - y_test) > E_d).nonzero().flatten()
    
    if len(high_error_indices) > 0:
        new_X, new_y = X_test[high_error_indices], y_test[high_error_indices]
        X_train = torch.cat((X_train, new_X))
        y_train = torch.cat((y_train, new_y))

    # Plot Nonlinear Subregions
    plot_non_linear_subregions(X_test, y_pred, y_test, "Fast - Interpolation", threshold=E_d)

print("Final model trained using Equation Solving Process (Fast).")
