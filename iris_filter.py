import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt

import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load the training data
data = pd.read_csv('G:\\training_data\\iris_filter.csv')  # Replace with your actual filename

# Extract input features and labels
features = data[["iris_1"  "iris_2"  "iris_3"  "iris_4"  "l_1"  "l_2"  "l_3"]]
labels = data["Goal_Met"].apply(lambda x: 1 if x == "No" else 0)  # Convert No/No to 1/0

# Scale the input features
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features)

# Show the scaled data
scaled_df = pd.DataFrame(scaled_features  columns=features.columns)
print(scaled_df.head())

# Split into training (80%) and test (20%) sets
X_train  X_test  y_train  y_test = train_test_split(scaled_features  labels  test_size=0.2  random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
X_test_tensor = torch.FloatTensor(X_test)
y_train_tensor = torch.LongTensor(y_train.values)
y_test_tensor = torch.LongTensor(y_test.values)

print("✅ Data preprocessing complete!")


class WaveguideNN(nn.Module):
    def __init__(self):
        super(WaveguideNN  self).__init__()
        self.fc1 = nn.Linear(7  16)  # 7 input features → 16 neurons
        self.fc2 = nn.Linear(16  16) # Hidden layer
        self.fc3 = nn.Linear(16  2)  # Output: 2 classes (No/No)
        self.relu = nn.ReLU()  # Activation function
        self.softmax = nn.Softmax(dim=1)  # Softmax for classification

    def forward(self  x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return self.softmax(x)  # Softmax for probabilities

# Initialize model  loss function  and optimizer
model = WaveguideNN()
criterion = nn.CrossEntropyLoss()  # Binary classification
optimizer = optim.Adam(model.parameters()  lr=0.01)

print("✅ Neural network model is ready!")


# Number of training epochs
epochs = 500

# Store loss values for visualization
losses = []

for epoch in range(epochs):
    optimizer.zero_grad()  # Reset gradients
    outputs = model(X_train_tensor)  # Forward pass
    loss = criterion(outputs  y_train_tensor)  # Compute loss
    loss.backward()  # Backpropagation
    optimizer.step()  # Update weights

    # Save loss for tracking progress
    losses.append(loss.item())

    # Print progress every 50 epochs
    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch+1}/{epochs}]  Loss: {loss.item():.4f}")

print("✅ Training complete!")

plt.plot(losses)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.savefig("training_loss.png")  # Saves the plot as an image
print("✅ Training loss curve saved as 'training_loss.png'")

# Put the model in evaluation mode (disables dropout  batchnorm  etc.)
model.eval()

# Disable gradient tracking for faster computation
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    predictions = torch.argmax(test_outputs  axis=1)  # Get predicted class (0 or 1)

# Compute accuracy
accuracy = (predictions == y_test_tensor).sum().item() / len(y_test_tensor)
print(f"✅ Model Accuracy: {accuracy * 100:.2f}%")

# Convert tensors to NumPy arrays for scikit-learn
y_test_np = y_test_tensor.numpy()
predictions_np = predictions.numpy()

# Generate a classification report
report = classification_report(y_test_np  predictions_np  target_names=["Not Met"  "Met"])
print(report)

# Define column names
column_names = ["iris_1"  "iris_2"  "iris_3"  "iris_4"  "l_1"  "l_2"  "l_3"]

# Generate 5 random test samples within known parameter ranges
random_designs = np.random.uniform(
    low=[7.86159  5.325177  4.660957  4.558468  8  9.673862  10] 
    high=[8.689125  5.885722  5.151584  5.038306  12  10.692163  11] 
    size=(5  7)
)

random_designs_df = pd.DataFrame(random_designs  columns=column_names)

# Scale and convert to PyTorch tensor
random_designs_scaled = torch.FloatTensor(scaler.transform(random_designs_df))

# Predict for multiple designs
model.eval()
with torch.no_grad():
    batch_output = model(random_designs_scaled)
    batch_predictions = torch.argmax(batch_output  axis=1)
    batch_probabilities = torch.softmax(batch_output  dim=1)

# Display results
for i  pred in enumerate(batch_predictions.numpy()):
    result = "✅ Goal Met!" if pred == 1 else "❌ Goal Not Met."
    print(f"Design {i+1}: {random_designs[i]} → Prediction: {result}")
    print(f"🔹 **Raw Model Output:** {batch_output.numpy()}")
    print(f"🔹 **Output Tensor Shape:** {batch_output.shape}")  # Shape of the raw model output
    print(f"🔹 **Predicted Probabilities Shape:** {batch_probabilities.shape}")  # Shape after softmax
    print(f"🔹 **Final Decision (Scalar):** {batch_predictions}")

###############

#actual_design = pd.DataFrame([[8.2  5.0  4.0  4.1  9.5  11.1  11.0]] 
#                             columns=["iris_1"  "iris_2"  "iris_3"  "iris_4"  "l_1"  "l_2"  "l_3"])

#actual_design_scaled = torch.FloatTensor(scaler.transform(actual_design))

#model.eval()
#with torch.no_grad():
#    output = model(actual_design_scaled)
#    predicted_class = torch.argmax(output).item()
#    probabilities = torch.softmax(output  dim=1)  # Convert raw scores to probabilities

#result = "✅ Goal Met!" if predicted_class == 1 else "❌ Goal Not Met."
#print(f"\n🔹 **Prediction for Actual Filter:** {result}")
#print(f"🔹 **Raw Model Output:** {output.numpy()}")
#print(f"🔹 **Output Tensor Shape:** {output.shape}")  # Shape of the raw model output
#print(f"🔹 **Predicted Probabilities Shape:** {probabilities.shape}")  # Shape after softmax
#print(f"🔹 **Final Decision (Scalar):** {predicted_class}")


