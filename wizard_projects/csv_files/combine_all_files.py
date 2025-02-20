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
                var_name = var_name.strip()
                var_value = var_value.strip()
                
                # Remove prefixes and match variables correctly
                var_name_cleaned = re.sub(r'^filter_\d+_', '', var_name)  # Remove numeric filter prefixes
                
                if var_name_cleaned in values_dict:
                    try:
                        values_dict[var_name_cleaned] = float(var_value)
                    except ValueError:
                        continue  # Ignore non-numeric values
    
    return pd.DataFrame([values_dict])  # Store as a single-row DataFrame

def combine_apl_bva(directory):
    """Combines APL and BVA files based on detected prefixes in the directory."""
    output_directory = os.path.join(os.path.dirname(directory), "combined_csv_files")
    os.makedirs(output_directory, exist_ok=True)
    
    # Identify all unique prefixes dynamically
    file_prefixes = set()
    for f in os.listdir(directory):
        match = re.match(r'(filter_\d+@opt\d+@iter\d+)_apl\.csv', f)
        if match:
            file_prefixes.add(match.group(1))
    
    for prefix in sorted(file_prefixes):
        apl_file_path = os.path.join(directory, f"{prefix}_apl.csv")
        bva_file_path = os.path.join(directory, f"{prefix}_bva.csv")
        
        if os.path.exists(apl_file_path) and os.path.exists(bva_file_path):
            df_apl = pd.read_csv(apl_file_path)
            df_bva = pd.read_csv(bva_file_path)
            df_bva["order"] = 7  # Set order value to 7
            
            # Expand BVA data to match APL row count
            df_bva_expanded = pd.concat([df_bva] * len(df_apl), ignore_index=True)
            df_combined = pd.concat([df_apl, df_bva_expanded], axis=1)
            
            # Define the expected final header
            final_header = [
                "freq", "s11_real", "s11_imag", "s21_real", "s21_imag", "s22_real", "s22_imag",
                "a", "b", 
                *[f"a2_{i}" for i in range(1, 16)],
                *[f"l_{i}" for i in range(1, 16)],
                "order"
            ]
            
            df_combined = df_combined[final_header]  # Ensure correct column order
            
            output_file_path = os.path.join(output_directory, f"{prefix}_combined.csv")
            df_combined.to_csv(output_file_path, index=False)
            print(f"Combined {prefix}_apl.csv and {prefix}_bva.csv into {prefix}_combined.csv")

if __name__ == "__main__":
    directory = r"/home/raavi/research/ai/wizard_projects/csv_files"  # Change this to the directory containing .csv files
    combine_apl_bva(directory)
