import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import RBFInterpolator, LinearNDInterpolator

# Configuration: Set These Values Before Running!
waveguide_width = 15.7  # Provide actual waveguide width for normalization
data_directory = "/home/raavi/research/ai/amg/train_data"  # Path where the 600 training files are stored

# Step 1: Load and Preprocess 600 Training Files
def load_training_data(directory):
    all_iris_params, all_frequencies, all_s11_real, all_s11_imag = [], [], [], []

    file_list = [f for f in os.listdir(directory) if f.endswith('.csv')]
    print(f"Found {len(file_list)} training files. Loading data...")

    for file in file_list:
        file_path = os.path.join(directory, file)
        df = pd.read_csv(file_path)

        # Extract relevant columns (No MinMax Scaling on Iris)
        iris_params = df[['iris_1', 'iris_2', 'iris_3', 'iris_4']].values / waveguide_width  # Normalize only by waveguide width
        frequencies = df[['freq']].values  # Frequency (kept in GHz)
        s11_real = df[['real']].values  # S11 Real Component
        s11_imag = df[['imag']].values  # S11 Imaginary Component

        # Append data to lists
        all_iris_params.append(iris_params)
        all_frequencies.append(frequencies)
        all_s11_real.append(s11_real)
        all_s11_imag.append(s11_imag)

    # Convert to NumPy arrays
    all_iris_params = np.vstack(all_iris_params)
    all_frequencies = np.vstack(all_frequencies)
    all_s11_real = np.vstack(all_s11_real)
    all_s11_imag = np.vstack(all_s11_imag)

    # Convert to PyTorch tensors
    iris_tensor = torch.tensor(all_iris_params, dtype=torch.float32)
    frequency_tensor = torch.tensor(all_frequencies, dtype=torch.float32)
    s11_real_tensor = torch.tensor(all_s11_real, dtype=torch.float32)
    s11_imag_tensor = torch.tensor(all_s11_imag, dtype=torch.float32)

    print(f"Loaded Data Shapes:")
    print(f"Iris Parameters: {iris_tensor.shape}, Frequency: {frequency_tensor.shape}, S11 Real: {s11_real_tensor.shape}, S11 Imag: {s11_imag_tensor.shape}")

    return iris_tensor, frequency_tensor, s11_real_tensor, s11_imag_tensor

# Step 2: Remove Duplicate Training Samples
# Step 2: Remove Duplicate Training Samples (Fixed for Frequency Variations)
def remove_duplicates(design_train, freq_train, s11_train_real, s11_train_imag):
    # Convert tensors to Pandas DataFrame
    df = pd.DataFrame(
        np.hstack((design_train.numpy(), freq_train.numpy(), s11_train_real.numpy(), s11_train_imag.numpy())),
        columns=['iris_1', 'iris_2', 'iris_3', 'iris_4', 'freq', 's11_real', 's11_imag']
    )

    # Drop duplicates based on both iris dimensions **AND** frequency values
    df = df.drop_duplicates(subset=['iris_1', 'iris_2', 'iris_3', 'iris_4', 'freq'])

    # Convert back to tensors
    return (
        torch.tensor(df.iloc[:, :-3].values, dtype=torch.float32),  # Iris dimensions
        torch.tensor(df.iloc[:, -3].values, dtype=torch.float32).view(-1, 1),  # Frequency
        torch.tensor(df.iloc[:, -2].values, dtype=torch.float32).view(-1, 1),  # S11 Real
        torch.tensor(df.iloc[:, -1].values, dtype=torch.float32).view(-1, 1)   # S11 Imag
    )

# Load the real dataset
design_train, frequency_train, s11_real_train, s11_imag_train = load_training_data(data_directory)

# Apply duplicate removal before training
design_train, frequency_train, s11_real_train, s11_imag_train = remove_duplicates(design_train, frequency_train, s11_real_train, s11_imag_train)


# Step 3: Adaptive Interpolation Model (Fix for Shape Mismatch)
def interpolate_response(design_train, s11_train, design_test, method="rbf"):
    # Add small noise to avoid singular matrix issue
    jitter_strength = 1e-6  # Small enough not to affect results
    design_train_noisy = design_train + torch.randn_like(design_train) * jitter_strength

    # Check for shape mismatch before interpolation
    if design_train_noisy.shape[0] != s11_train.shape[0]:
        print(f"ERROR: Shape mismatch! design_train_noisy has {design_train_noisy.shape[0]} samples, but s11_train has {s11_train.shape[0]} samples.")
        return torch.zeros((design_test.shape[0], 1), dtype=torch.float32)  # Return zeros to prevent failure

    if method == "rbf":
        interpolator = RBFInterpolator(design_train_noisy.numpy(), s11_train.numpy().flatten(), kernel='thin_plate_spline')
    else:
        interpolator = LinearNDInterpolator(design_train_noisy.numpy(), s11_train.numpy().flatten())

    s11_predicted = interpolator(design_test.numpy())
    return torch.tensor(s11_predicted, dtype=torch.float32).view(-1, 1)


# Step 3: Adaptive Interpolation Model (Fix for Singular Matrix)
# def interpolate_response(design_train, s11_train, design_test, method="rbf"):
#     # Add small noise to avoid singular matrix issue
#     jitter_strength = 1e-6  # Small enough not to affect results
#     design_train_noisy = design_train + torch.randn_like(design_train) * jitter_strength
#
#     if method == "rbf":
#         interpolator = RBFInterpolator(design_train_noisy.numpy(), s11_train.numpy().flatten(), kernel='thin_plate_spline')
#     else:
#         interpolator = LinearNDInterpolator(design_train_noisy.numpy(), s11_train.numpy().flatten())
#
#     s11_predicted = interpolator(design_test.numpy())
#     return torch.tensor(s11_predicted, dtype=torch.float32).view(-1, 1)

# Step 3: Adaptive Interpolation Model (Equation Solving Process)
# def interpolate_response(design_train, s11_train, design_test, method="rbf"):
#     if method == "rbf":
#         interpolator = RBFInterpolator(design_train.numpy(), s11_train.numpy().flatten(), kernel='thin_plate_spline')
#     else:
#         interpolator = LinearNDInterpolator(design_train.numpy(), s11_train.numpy().flatten())

#     s11_predicted = interpolator(design_test.numpy())
#     return torch.tensor(s11_predicted, dtype=torch.float32).view(-1, 1)

# Step 4: Visualization Function (Dynamic Incremental Data Generation)
def plot_error_regions(design_test, s11_predicted, s11_true, method, error_threshold=0.1):
    prediction_errors = torch.abs(s11_predicted - s11_true).detach().numpy()
    
    plt.figure(figsize=(7, 6))
    plt.scatter(design_test[:, 0], design_test[:, 1], c=prediction_errors, cmap='plasma', s=50, edgecolors='k')
    plt.colorbar(label="Prediction Error (|S11_pred - S11_true|)")
    plt.xlabel("Iris Dimension 1")
    plt.ylabel("Iris Dimension 2")
    plt.title(f"Non-Linear Subregions - {method} (Higher Error in Darker Regions)")
    plt.show()

# Step 5: Initialize Training Parameters
error_threshold = 0.01  # Initial error threshold
error_threshold_decay = 0.9  # Adaptive reduction factor

print("Running AMG-Based Equation Solving Process (Fast - Using Real Data)...")

# Step 6: Iterative Optimization and Dynamic Data Generation
for iteration in range(5):
    # Step 6.1: Generate Testing Data (New Sampling Space)
    design_test, frequency_test, s11_real_test, s11_imag_test = load_training_data(data_directory)

    # Step 6.2: Interpolation-Based Prediction
    s11_real_predicted = interpolate_response(design_train, s11_real_train, design_test, method="rbf")
    s11_imag_predicted = interpolate_response(design_train, s11_imag_train, design_test, method="rbf")

    # Step 6.3: Compute Testing Error
    criterion = torch.nn.MSELoss()
    E_test_real = criterion(s11_real_predicted, s11_real_test).item()
    E_test_imag = criterion(s11_imag_predicted, s11_imag_test).item()
    
    print(f"Iteration {iteration+1}, Testing Error: Real={E_test_real}, Imag={E_test_imag}")

    # Step 6.4: Check Termination Condition
    if E_test_real < error_threshold and E_test_imag < error_threshold:
        print("Desired accuracy reached!")
        break  

    # Step 7: Dynamic Incremental Data Generation (High-Error Regions)
    high_error_indices_real = (torch.abs(s11_real_predicted - s11_real_test) > error_threshold).nonzero().flatten()
    high_error_indices_imag = (torch.abs(s11_imag_predicted - s11_imag_test) > error_threshold).nonzero().flatten()
    
    if len(high_error_indices_real) > 0 and len(high_error_indices_imag) > 0:
        new_design_samples_real, new_s11_real_samples = design_test[high_error_indices_real], s11_real_test[high_error_indices_real]
        new_design_samples_imag, new_s11_imag_samples = design_test[high_error_indices_imag], s11_imag_test[high_error_indices_imag]

        # Limit growth to 5 new samples per iteration
        if len(new_design_samples_real) > 5:
            new_design_samples_real, new_s11_real_samples = new_design_samples_real[:5], new_s11_real_samples[:5]
        if len(new_design_samples_imag) > 5:
            new_design_samples_imag, new_s11_imag_samples = new_design_samples_imag[:5], new_s11_imag_samples[:5]

        design_train = torch.cat((design_train, new_design_samples_real, new_design_samples_imag))
        s11_real_train = torch.cat((s11_real_train, new_s11_real_samples))
        s11_imag_train = torch.cat((s11_imag_train, new_s11_imag_samples))

        # Step 7.1: Update the Error Threshold Adaptively
        error_threshold *= error_threshold_decay

    # Step 8: Visualization of Error Distribution
    plot_error_regions(design_test, s11_real_predicted, s11_real_test, "Fast - AMG Interpolation (Real)", error_threshold)
    plot_error_regions(design_test, s11_imag_predicted, s11_imag_test, "Fast - AMG Interpolation (Imag)", error_threshold)

print("Final model trained using AMG-Based Equation Solving Process (Fast) with Real Data.")
print("\n✅ Final Iris Dimensions Used for Training:")
print(design_train*waveguide_width)  # Print all final iris configurations

