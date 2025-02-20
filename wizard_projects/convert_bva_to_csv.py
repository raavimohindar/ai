import os
import pandas as pd
import re

def extract_values(file_path):
    """Extracts only values under [Values] from a .bva file and ensures all expected variables are present."""
    header = [
        "a", "b",
        *[f"a2_{i}" for i in range(1, 16)],
        *[f"l_{i}" for i in range(1, 16)]
    ]
    
    with open(file_path, "r") as file:
        lines = file.readlines()
    
    values_section = False
    values_dict = {key: 0.0 for key in header}  # Initialize all values to 0.0
    
    for line in lines:
        line = line.strip()
        
        if line.startswith("[Values]"):
            values_section = True
            continue
        
        if values_section:
            if line.startswith("[") and line != "[Values]":
                break  # Stop processing if another section starts
            
            if "=" in line:
                var_name, var_value = line.split("=", 1)
                var_name = re.sub(r'^[^a-zA-Z]+', '', var_name.strip())  # Remove any prefix
                var_name = re.sub(r'^.*_', '', var_name)  # Keep only base variable name
                var_value = var_value.strip()
                
                if var_name in values_dict:
                    try:
                        values_dict[var_name] = float(var_value)
                    except ValueError:
                        continue  # Ignore non-numeric values
    
    return pd.DataFrame([values_dict])  # Store as a single-row DataFrame

def convert_bva_to_csv(directory):
    """Converts all .bva files in the specified directory to .csv files and stores them in a separate directory."""
    output_directory = os.path.join(os.path.dirname(directory), "csv_files")
    os.makedirs(output_directory, exist_ok=True)
    
    for file_name in os.listdir(directory):
        if file_name.endswith(".bva"):
            file_path = os.path.join(directory, file_name)
            df = extract_values(file_path)
            csv_file_name = os.path.splitext(file_name)[0] + ".csv"
            csv_file_path = os.path.join(output_directory, csv_file_name)
            df.to_csv(csv_file_path, index=False)
            print(f"Converted {file_name} to {csv_file_name} and saved in {output_directory}")

if __name__ == "__main__":
    directory = r"G:\waveguide_ai\wizard_projects\bva_files"  # Change this to the directory containing .bva files
    convert_bva_to_csv(directory)
