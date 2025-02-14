# Generate random test samples
import numpy as np
import pandas as pd

random_designs = np.random.uniform(
    low=[7.96159, 5.325177, 4.660957, 4.558468],
    high=[8.289125, 5.885722, 5.151584, 5.038306],
    size=(500, 4)
)

np.savetxt("test_data_without_lengths.csv", random_designs, delimiter=",", 
           header="iris_1,iris_2,iris_3,iris_4", comments="", fmt="%.6f")

# Convert to DataFrame with specified column names
#df = pd.DataFrame(random_designs, columns=["iris_1", "iris_2", "iris_3", "iris_4"])

# Save to CSV file with headers and without index
#df.to_csv("test_data_without_lengths.csv", index=False)