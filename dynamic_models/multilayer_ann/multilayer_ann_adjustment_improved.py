import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Define Multilayer ANN with AMG Efficient Adjustment (Figure 5)
class AMG_MultilayerANN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(AMG_MultilayerANN, self).__init__()
        self.hidden_layers = nn.ModuleList()
        self.output_layer = nn.Linear(hidden_sizes[-1] if hidden_sizes else input_size, output_size)
        
        # Create L hidden layers
        for i, H_i in enumerate(hidden_sizes):
            self.hidden_layers.append(nn.Linear(input_size if i == 0 else hidden_sizes[i - 1], H_i))
        
        self.activation = nn.SiLU()  # Swish Activation

    def forward(self, x):
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        return self.output_layer(x)

# Generate Initial Training Data
def generate_initial_data(n_samples):
    input_features = np.random.uniform(-1, 1, (n_samples, 2))  
    target_values = np.sin(input_features[:, 0]) + np.cos(input_features[:, 1])  
    return torch.tensor(input_features, dtype=torch.float32), torch.tensor(target_values, dtype=torch.float32).view(-1, 1)

# Train Model with L1 Regularization (Equation 4)
def train_model(model, criterion, optimizer, X_train, y_train, epochs=100, λ_k=0.001):
    for epoch in range(epochs):
        optimizer.zero_grad()
        predictions = model(X_train)
        loss = criterion(predictions, y_train)

        # Apply L1 regularization (Equation 4)
        l1_penalty = sum(torch.sum(torch.abs(param)) for param in model.parameters())
        loss += λ_k * l1_penalty  

        loss.backward()
        optimizer.step()
    return loss.item()

# Function to visualize non-linear subregions
def plot_error_regions(X_test, y_pred, y_test):
    prediction_errors = torch.abs(y_pred - y_test).detach().numpy()
    
    plt.figure(figsize=(7, 6))
    plt.scatter(X_test[:, 0], X_test[:, 1], c=prediction_errors, cmap='plasma', s=50, edgecolors='k')
    plt.colorbar(label="Prediction Error")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Non-Linear Subregions (AMG-Based ANN Adjustment)")
    plt.show()

# Initialize AMG Algorithm (Figure 5)
input_size, L, output_size = 2, 1, 1  # Start with L=1 hidden layer
H_i = [10]  # Initial number of neurons in first layer
α_k = 1  # Start with α_k = 1
λ_k = 0.0005  # Initial regularization coefficient
max_neurons = 100  # Limit neuron growth

# Initialize ANN Model
model = AMG_MultilayerANN(input_size, H_i, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Generate Initial Data
X_train, y_train = generate_initial_data(10)
E_d = 0.01  # Desired error threshold

print("Running AMG with Efficient Multilayer ANN Structure Adjustment (Figure 5)...")

for iteration in range(5):
    # Step 1: Train the model with L1 Regularization
    loss = train_model(model, criterion, optimizer, X_train, y_train, epochs=100, λ_k=λ_k)
    
    # Step 2: Evaluate Testing Error
    X_test, y_test = generate_initial_data(100)
    with torch.no_grad():
        y_pred = model(X_test)
        E_test = criterion(y_pred, y_test).item()
    
    print(f"Iteration {iteration+1}, Testing Error: {E_test}")

    # Step 3: Check Termination Condition
    if E_test < E_d:
        print("Desired accuracy reached!")
        break  

    # Step 4: Dynamic Data Generation (High-Error Regions)
    high_error_indices = (torch.abs(y_pred - y_test) > E_d).nonzero().flatten()
    
    if len(high_error_indices) > 0:
        new_X, new_y = X_test[high_error_indices], y_test[high_error_indices]

        # Control Growth (Limit to 5 new samples)
        if len(new_X) > 5:
            new_X, new_y = new_X[:5], new_y[:5]

        X_train = torch.cat((X_train, new_X))
        y_train = torch.cat((y_train, new_y))

    # Step 5: ANN Structure Adjustment (Equation 3)
    if loss > E_d:
        if H_i[-1] < max_neurons:
            new_neurons = H_i[-1] + 5  # Expand neurons per Equation (3)
            H_i.append(min(new_neurons, max_neurons))
        else:
            H_i.append(5)  # If max neurons reached, add a new hidden layer

        # Step 6: Prune Unused Neurons (If Needed)
        if H_i[-1] == 0:
            H_i.pop()  # Remove unneeded layer if neurons are zero

        # Step 7: Update Model Structure
        model = AMG_MultilayerANN(input_size, H_i, output_size)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        print(f"Updated ANN Structure: Hidden Layers = {L}, Neurons Per Layer = {H_i}")

    # Step 8: Final Fine-Tuning of Adjusted Model
    train_model(model, criterion, optimizer, X_train, y_train, epochs=50, λ_k=λ_k)

    # Step 9: Plot Non-Linear Subregions
    plot_error_regions(X_test, y_pred, y_test)

print("Final model trained using AMG-Based Efficient ANN Adjustment.")
