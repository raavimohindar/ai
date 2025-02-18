import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Knowledge-Based Neural Network (KBNN) - Figure 6 Implementation
class KBNN(nn.Module):
    def __init__(self, input_size, freq_size, hidden_sizes, output_size):
        super(KBNN, self).__init__()

        # Input Mapping ANN (Linear + Nonlinear Mapping)
        self.input_linear_mapping = nn.Linear(input_size, hidden_sizes[0])
        self.input_nonlinear_mapping = nn.Linear(input_size, hidden_sizes[0])

        # Frequency Mapping ANN (Linear + Nonlinear Mapping)
        self.freq_linear_mapping = nn.Linear(freq_size, hidden_sizes[1])
        self.freq_nonlinear_mapping = nn.Linear(freq_size, hidden_sizes[1])

        # Projection Layer to Match Sizes
        self.projection_layer = nn.Linear(hidden_sizes[0], hidden_sizes[1])

        # Output Mapping ANN
        self.output_mapping = nn.Linear(hidden_sizes[1], output_size)

        self.activation = nn.SiLU()

    def forward(self, x_parameters, frequency_parameters):
        # Input Mapping
        x_linear = self.input_linear_mapping(x_parameters)
        x_nonlinear = self.activation(self.input_nonlinear_mapping(x_parameters))
        x_combined = x_linear + x_nonlinear  # Size: hidden_sizes[0]

        # Frequency Mapping
        f_linear = self.freq_linear_mapping(frequency_parameters)
        f_nonlinear = self.activation(self.freq_nonlinear_mapping(frequency_parameters))
        f_combined = f_linear + f_nonlinear  # Size: hidden_sizes[1]

        # Project x_combined to match f_combined size
        x_projected = self.projection_layer(x_combined)  # Now same size as f_combined

        # Combine Features
        combined_features = x_projected + f_combined

        # Output Mapping (Prediction of S11)
        s11_predicted = self.output_mapping(combined_features)
        return s11_predicted


# Simulated Prior Knowledge Function (Optional)
def prior_knowledge_s11(frequency_parameters):
    """ Simulated analytical formula for S11 behavior based on prior knowledge """
    return torch.exp(-torch.abs(frequency_parameters - 6))  # Example: Resonance near 6 GHz

# Generate Training Data (Simulated Based on Figure 6)
def generate_training_data(n_samples):
    x_parameters = np.random.uniform(0.1, 2.0, (n_samples, 3))  # Physical Design Parameters
    frequency_parameters = np.linspace(3, 10, num=n_samples).reshape(n_samples, 1)  # Passband Frequencies
    s11_prior = np.exp(-np.abs(frequency_parameters - 6))  # Prior Knowledge Effect
    s11_response = s11_prior + np.random.normal(0, 0.02, size=s11_prior.shape)  # Adding noise to simulate real-world data

    return (
        torch.tensor(x_parameters, dtype=torch.float32),
        torch.tensor(frequency_parameters, dtype=torch.float32),
        torch.tensor(s11_response, dtype=torch.float32).view(-1, 1)
    )

# Loss Function Incorporating Goal Function Error (Equation 5)
def loss_function(predicted_s11, true_s11, goal_error_lambda=0.001):
    mse_loss = nn.MSELoss()(predicted_s11, true_s11)
    goal_error = torch.mean(torch.abs(predicted_s11 - true_s11))  # Goal Function Error
    return mse_loss + goal_error_lambda * goal_error

# Two-Stage Training Function (Figure 7)
def train_kbnn(model, optimizer, x_parameters, frequency_parameters, s11_true, epochs=500, l1_lambda=0.0001, stage=1):
    for epoch in range(epochs):
        optimizer.zero_grad()
        s11_predicted = model(x_parameters, frequency_parameters)
        loss = loss_function(s11_predicted, s11_true)

        # Stage 1: Apply L1 Regularization to Select Mapping (Figure 7)
        if stage == 1:
            l1_penalty = sum(torch.sum(torch.abs(param)) for param in model.parameters())
            loss += l1_lambda * l1_penalty  

        loss.backward()
        optimizer.step()

    return loss.item()

# Function to visualize the predicted vs true S11
def plot_s11_results(frequency_parameters, s11_predicted, s11_true):
    plt.figure(figsize=(7, 6))
    plt.plot(frequency_parameters.numpy(), s11_true.numpy(), 'b-', label="True S11 (Ground Truth)")
    plt.plot(frequency_parameters.numpy(), s11_predicted.detach().numpy(), 'r--', label="Predicted S11 (KBNN)")
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("S11 Magnitude")
    plt.title("KBNN Prediction of S11 vs Frequency")
    plt.legend()
    plt.show()

# Initialize Model (Figure 6)
input_size, freq_size, output_size = 3, 1, 1  # KBNN Structure
hidden_sizes = [20, 30, 20]  # Increase Layers and Neurons for Flexibility
kbnn_model = KBNN(input_size, freq_size, hidden_sizes, output_size)
optimizer = optim.Adam(kbnn_model.parameters(), lr=0.01)

# Generate Training Data
x_parameters, frequency_parameters, s11_true = generate_training_data(100)

# Stage 1 Training: L1 Regularization-Based Mapping Selection (Figure 7)
print("Stage 1 Training: Selecting Mapping Types Using L1 Regularization...")
train_kbnn(kbnn_model, optimizer, x_parameters, frequency_parameters, s11_true, stage=1)
print("Stage 1 Completed.")

# Stage 2 Training: Fine-Tuning for Final Accuracy (Figure 7)
print("Stage 2 Training: Fine-Tuning Model After Mapping Selection...")
train_kbnn(kbnn_model, optimizer, x_parameters, frequency_parameters, s11_true, stage=2)
print("Stage 2 Completed.")

# Evaluate Model on New Data
x_test, freq_test, s11_test = generate_training_data(100)
with torch.no_grad():
    s11_predicted = kbnn_model(x_test, freq_test)

# Plot Results
plot_s11_results(freq_test, s11_predicted, s11_test)

print("KBNN Model Successfully Trained and Evaluated.")
