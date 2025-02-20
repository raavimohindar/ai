import os
import torch
import pandas as pd
import numpy as np

# Configuration (Set This!)
waveguide_width = 15.7  # Replace with actual waveguide width for normalization
data_directory = "/mnt/data/training_files/"  # Change to your directory path

# Initialize lists to store all data
all_iris_params = []
all_frequencies = []
all_s11_real = []
all_s11_imag = []

# Step 1: Read All 600 Files
file_list = [f for f in os.listdir(data_directory) if f.endswith('.csv')]  # Assuming CSV files

print(f"Found {len(file_list)} files. Loading data...")

for file in file_list:
    file_path = os.path.join(data_directory, file)
    
    # Read file
    df = pd.read_csv(file_path)

    # Step 2: Extract Relevant Columns
    iris_params = df[['iris_1', 'iris_2', 'iris_3', 'iris_4']].values / waveguide_width  # Normalize by waveguide width
    frequencies = df[['freq']].values  # Frequency (can normalize if needed)
    s11_real = df[['s_11_real']].values  # S11 Real Component
    s11_imag = df[['s_11_imag']].values  # S11 Imaginary Component

    # Step 3: Append to lists
    all_iris_params.append(iris_params)
    all_frequencies.append(frequencies)
    all_s11_real.append(s11_real)
    all_s11_imag.append(s11_imag)

# Step 4: Convert to NumPy Arrays
all_iris_params = np.vstack(all_iris_params)
all_frequencies = np.vstack(all_frequencies)
all_s11_real = np.vstack(all_s11_real)
all_s11_imag = np.vstack(all_s11_imag)

# Step 5: Convert to PyTorch Tensors
iris_tensor = torch.tensor(all_iris_params, dtype=torch.float32)
frequency_tensor = torch.tensor(all_frequencies, dtype=torch.float32)
s11_real_tensor = torch.tensor(all_s11_real, dtype=torch.float32)
s11_imag_tensor = torch.tensor(all_s11_imag, dtype=torch.float32)

# Final Data Format
print(f"Iris Parameters Tensor: {iris_tensor.shape}")
print(f"Frequency Tensor: {frequency_tensor.shape}")
print(f"S11 Real Tensor: {s11_real_tensor.shape}")
print(f"S11 Imag Tensor: {s11_imag_tensor.shape}")

# Now the data is ready to be used in the model training process
