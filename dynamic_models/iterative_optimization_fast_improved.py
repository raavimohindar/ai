import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RBFInterpolator, LinearNDInterpolator

# Generate initial training data (microwave iris filter modeling)
def generate_initial_data(n_samples):
    design_parameters = np.random.uniform(-1, 1, (n_samples, 2))  # Example: iris widths & spacings
    response_s11 = np.sin(design_parameters[:, 0]) + np.cos(design_parameters[:, 1])  # Simulated S11 response
    return (
        torch.tensor(design_parameters, dtype=torch.float32),
        torch.tensor(response_s11, dtype=torch.float32).view(-1, 1),
    )

# Adaptive Interpolation Model (Equation Solving Process)
def interpolate_response(design_train, s11_train, design_test, method="rbf"):
    if method == "rbf":
        interpolator = RBFInterpolator(design_train.numpy(), s11_train.numpy().flatten(), kernel='thin_plate_spline')
    else:
        interpolator = LinearNDInterpolator(design_train.numpy(), s11_train.numpy().flatten())

    s11_predicted = interpolator(design_test.numpy())
    return torch.tensor(s11_predicted, dtype=torch.float32).view(-1, 1)

# Function to visualize non-linear subregions (Dynamic Incremental Data Generation)
def plot_error_regions(design_test, s11_predicted, s11_true, method, error_threshold=0.1):
    prediction_errors = torch.abs(s11_predicted - s11_true).detach().numpy()
    
    plt.figure(figsize=(7, 6))
    plt.scatter(design_test[:, 0], design_test[:, 1], c=prediction_errors, cmap='plasma', s=50, edgecolors='k')
    plt.colorbar(label="Prediction Error (|S11_pred - S11_true|)")
    plt.xlabel("Iris Dimension 1")
    plt.ylabel("Iris Dimension 2")
    plt.title(f"Non-Linear Subregions - {method} (Higher Error in Darker Regions)")
    plt.show()

# Initialize Training Data (Step 1: Initial Design Space Sampling)
design_train, s11_train = generate_initial_data(10)
error_threshold = 0.01  # Initial error threshold
error_threshold_decay = 0.9  # Adaptive reduction factor

print("Running AMG-Based Equation Solving Process (Fast - Figure 3 Implementation)...")

# Step 2: Iterative Optimization and Dynamic Data Generation
for iteration in range(5):
    # Step 2.1: Generate Testing Data (New Sampling Space)
    design_test, s11_true = generate_initial_data(100)
    
    # Step 2.2: Interpolation-Based Prediction
    s11_predicted = interpolate_response(design_train, s11_train, design_test, method="rbf")

    # Step 2.3: Compute Testing Error
    criterion = torch.nn.MSELoss()
    E_test = criterion(s11_predicted, s11_true).item()
    
    print(f"Iteration {iteration+1}, Testing Error: {E_test}")

    # Step 2.4: Check Termination Condition
    if E_test < error_threshold:
        print("Desired accuracy reached!")
        break  

    # Step 3: Dynamic Incremental Data Generation (High-Error Regions)
    high_error_indices = (torch.abs(s11_predicted - s11_true) > error_threshold).nonzero().flatten()
    
    if len(high_error_indices) > 0:
        new_design_samples, new_s11_samples = design_test[high_error_indices], s11_true[high_error_indices]

        # Dynamically increase sampling in high-error regions (Control Growth)
        if len(new_design_samples) > 5:
            new_design_samples, new_s11_samples = new_design_samples[:5], new_s11_samples[:5]

        design_train = torch.cat((design_train, new_design_samples))
        s11_train = torch.cat((s11_train, new_s11_samples))

        # Step 3.1: Update the Error Threshold Adaptively
        error_threshold *= error_threshold_decay

    # Step 4: Visualization of Error Distribution (High-Error Subregions)
    plot_error_regions(design_test, s11_predicted, s11_true, "Fast - AMG Interpolation", error_threshold)

print("Final model trained using AMG-Based Equation Solving Process (Fast).")
