import matplotlib.pyplot
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import matplotlib
matplotlib.use('GTK3Agg')  # or 'GTK3Cairo'

import matplotlib.pyplot as plt

# Define ANN Model with Swish Activation
class ANNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ANNModel, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        self.activation = nn.SiLU()  # Using Swish activation

    def forward(self, x):
        x = self.activation(self.hidden(x))
        x = self.output(x)
        return x

# Generate initial data (simulated)
def generate_data(n_samples):
    x = np.random.uniform(-1, 1, (n_samples, 2))  
    y = np.sin(x[:, 0]) + np.cos(x[:, 1])  
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).view(-1, 1)

# Training function
def train_model(model, criterion, optimizer, X_train, y_train, epochs=100):
    for epoch in range(epochs):
        optimizer.zero_grad()
        predictions = model(X_train)
        loss = criterion(predictions, y_train)
        loss.backward()
        optimizer.step()
    return loss.item()

# Function to visualize non-linear subregions
def plot_non_linear_subregions(X_test, y_pred, y_test, threshold=0.1):
    errors = torch.abs(y_pred - y_test).detach().numpy()
    
    plt.figure(figsize=(7, 6))
    plt.scatter(X_test[:, 0], X_test[:, 1], c=errors, cmap='plasma', s=50, edgecolors='k')
    plt.colorbar(label="Prediction Error")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.title("Non-Linear Subregions (Higher Error in Darker Regions)")
    plt.show()

# Initialize ANN Model
input_size, hidden_size, output_size = 2, 10, 1
model = ANNModel(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Initial Training Data
X_train, y_train = generate_data(10)
E_d = 0.01  # Desired error threshold

print("Running Iterative Optimization Process (Slow - ANN Training)...")
for iteration in range(5):
    loss = train_model(model, criterion, optimizer, X_train, y_train)
    
    # Check Testing Error
    X_test, y_test = generate_data(100)
    with torch.no_grad():
        y_pred = model(X_test)
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
    
    # Adjust ANN Structure if Necessary
    if loss > E_d:
        hidden_size += 5  
        model = ANNModel(input_size, hidden_size, output_size)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        print(f"Model structure updated: Hidden size = {hidden_size}")

    # Plot Nonlinear Subregions
    plot_non_linear_subregions(X_test, y_pred, y_test, threshold=E_d)

print("Final model trained using Iterative Optimization Process (Slow).")
