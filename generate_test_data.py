# Generate random test samples
import numpy as np
import pandas as pd

random_designs = np.random.uniform(
    low=[7.85, 5.32, 4.70, 4.6],
    high=[8.15, 5.62, 5.0, 4.9],
    size=(500, 4)
)

np.savetxt("test_data_without_lengths.csv", random_designs, delimiter=",", 
           header="iris_1,iris_2,iris_3,iris_4", comments="", fmt="%.6f")

# Convert to DataFrame with specified column names
#df = pd.DataFrame(random_designs, columns=["iris_1", "iris_2", "iris_3", "iris_4"])

# Save to CSV file with headers and without index
#df.to_csv("test_data_without_lengths.csv", index=False)