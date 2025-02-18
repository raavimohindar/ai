import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Define Dynamic Multilayer ANN with Efficient Adjustment (Figure 5)
class AdaptiveMultilayerANN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(AdaptiveMultilayerANN, self).__init__()
        self.hidden_layers = nn.ModuleList()
        self.output_layer = nn.Linear(hidden_sizes[-1] if hidden_sizes else input_size, output_size)
        
        # Define multiple hidden layers based on Equation (3)
        for i, h_size in enumerate(hidden_sizes):
            self.hidden_layers.append(nn.Linear(input_size if i == 0 else hidden_sizes[i - 1], h_size))
        
        self.activation = nn.SiLU()  # Swish Activation
    
    def forward(self, x):
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        return self.output_layer(x)

# Generate nonlinear dataset
def generate_data(n_samples):
    x = np.random.uniform(-1, 1, (n_samples, 2))  
    y = np.sin(x[:, 0]) + np.cos(x[:, 1])  
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).view(-1, 1)

# Train the Adaptive ANN with l1 regularization (Equation 4)
def train_model(model, criterion, optimizer, X_train, y_train, epochs=100, l1_lambda=0.001):
    for epoch in range(epochs):
        optimizer.zero_grad()
        predictions = model(X_train)
        loss = criterion(predictions, y_train)

        # Apply L1 regularization (Equation 4) to hidden layers
        l1_penalty = sum(torch.sum(torch.abs(param)) for param in model.parameters())
        loss += l1_lambda * l1_penalty  # Regularization

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
    plt.title("Non-Linear Subregions (Efficient Multilayer ANN Adjustment)")
    plt.show()

# Initialize Dynamic ANN Model
input_size, hidden_sizes, output_size = 2, [10], 1  # Start with a single hidden layer
model = AdaptiveMultilayerANN(input_size, hidden_sizes, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Initial Training Data
X_train, y_train = generate_data(10)
E_d = 0.01  # Desired error threshold
max_hidden_neurons = 100  # Prevent excessive neuron growth

print("Running AMG with Efficient Multilayer ANN Structure Adjustment (Figure 5)...")

for iteration in range(5):
    loss = train_model(model, criterion, optimizer, X_train, y_train, epochs=100, l1_lambda=0.0005)
    
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
    
    # Adjust ANN Structure Based on Equation (3)
    if loss > E_d:
        if hidden_sizes[-1] < max_hidden_neurons:
            new_neurons = hidden_sizes[-1] + 5  # Expand neurons following Eq. (3)
            hidden_sizes.append(min(new_neurons, max_hidden_neurons))
        else:
            hidden_sizes.append(5)  # If large enough, add a new hidden layer
        
        model = AdaptiveMultilayerANN(input_size, hidden_sizes, output_size)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        print(f"Model structure updated: Hidden layers = {hidden_sizes}")

    # Plot Nonlinear Subregions
    plot_non_linear_subregions(X_test, y_pred, y_test, threshold=E_d)

print("Final model trained using AMG with Efficient Multilayer ANN Structure Adjustment (Figure 5).")
