import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Knowledge-Based Neural Network (Figure 6)
class KBNN(nn.Module):
    def __init__(self, input_size, freq_size, hidden_sizes, output_size):
        super(KBNN, self).__init__()
        
        # Input Mapping ANN
        self.input_mapping = nn.Linear(input_size, hidden_sizes[0])

        # Frequency Mapping ANN
        self.freq_mapping = nn.Linear(freq_size, hidden_sizes[1])

        # Output Mapping ANN
        self.output_mapping = nn.Linear(hidden_sizes[1], output_size)

        self.activation = nn.SiLU()  # Swish activation for better gradient flow

    def forward(self, iris_dims, freq_samples):
    # Input Mapping
        x = self.activation(self.input_mapping(iris_dims))

        # Frequency Mapping
        f = self.activation(self.freq_mapping(freq_samples))

        # Resize x to match f using a Linear Transformation
        x_resized = nn.Linear(x.shape[1], f.shape[1]).to(x.device)(x)

        # Output Mapping (Prediction of S11)
        s11_pred = self.output_mapping(f + x_resized)  # Now dimensions match
        return s11_pred    
     

# Generate Training Data (Simulated based on Prior Knowledge)
def generate_training_data(n_samples):
    iris_dims = np.random.uniform(0.1, 2.0, (n_samples, 3))  # Simulated iris dimensions
    freq_samples = np.linspace(3, 10, num=n_samples).reshape(n_samples, 1)  # Passband frequencies
    s11 = np.exp(-np.abs(freq_samples - 6))  # Simulated S11 response with prior knowledge effect
    return (
        torch.tensor(iris_dims, dtype=torch.float32),
        torch.tensor(freq_samples, dtype=torch.float32),
        torch.tensor(s11, dtype=torch.float32).view(-1, 1)
    )
    

# Loss Function Incorporating Goal Function Error
def loss_function(predicted_s11, true_s11, goal_error_lambda=0.001):
    mse_loss = nn.MSELoss()(predicted_s11, true_s11)
    goal_error = torch.mean(torch.abs(predicted_s11 - true_s11))  # Goal function error
    return mse_loss + goal_error_lambda * goal_error

# Training Function
def train_kbnn(model, optimizer, iris_dims, freq_samples, s11_true, epochs=500, l1_lambda=0.0001):
    for epoch in range(epochs):
        optimizer.zero_grad()
        s11_pred = model(iris_dims, freq_samples)
        loss = loss_function(s11_pred, s11_true)

        # Apply L1 regularization to prune unnecessary neurons
        l1_penalty = sum(torch.sum(torch.abs(param)) for param in model.parameters())
        loss += l1_lambda * l1_penalty  

        loss.backward()
        optimizer.step()

    return loss.item()

# Function to visualize the predicted vs true S11
def plot_s11_results(freq_samples, s11_pred, s11_true):
    plt.figure(figsize=(7, 6))
    plt.plot(freq_samples.numpy(), s11_true.numpy(), 'b-', label="True S11 (Ground Truth)")
    plt.plot(freq_samples.numpy(), s11_pred.detach().numpy(), 'r--', label="Predicted S11 (KBNN)")
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("S11 Magnitude")
    plt.title("KBNN Prediction of S11 vs Frequency")
    plt.legend()
    plt.show()

# Initialize Model


input_size, freq_size, output_size = 3, 1, 1  # KBNN structure

hidden_sizes = [20, 30, 20]  # Increase layers and neurons

kbnn_model = KBNN(input_size, freq_size, hidden_sizes, output_size)

optimizer = optim.Adam(kbnn_model.parameters(), lr=0.01)

# Generate Training Data
iris_dims, freq_samples, s11_true = generate_training_data(100)

# Train Model
print("Training KBNN Model for Microwave Iris Filter S11 Prediction...")
train_loss = train_kbnn(kbnn_model, optimizer, iris_dims, freq_samples, s11_true)
print(f"Final Training Loss: {train_loss}")

# Evaluate Model on New Data
iris_test, freq_test, s11_test = generate_training_data(100)
with torch.no_grad():
    s11_pred = kbnn_model(iris_test, freq_test)

# Plot Results
plot_s11_results(freq_test, s11_pred, s11_test)

print("KBNN Model Successfully Trained and Evaluated.")
