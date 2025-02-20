import os
import pandas as pd
import re
from collections import defaultdict

def extract_s_parameters(file_path):
    """Extracts S-parameters from an .apl file and returns a DataFrame."""
    with open(file_path, "r") as file:
        lines = file.readlines()
    
    data_dict = defaultdict(list)
    current_section = None
    data_pattern = re.compile(r"([-+]?\d*\.\d+|\d+)\s+\((-?\d*\.\d+E?-?\d*),(-?\d*\.\d+E?-?\d*)\)")
    
    for line in lines:
        line = line.strip()
        if line.startswith("[s11"):
            current_section = "s11"
        elif line.startswith("[s21"):
            current_section = "s21"
        elif line.startswith("[s22"):
            current_section = "s22"
        
        match = data_pattern.match(line)
        if match and current_section:
            freq, real, imag = float(match.group(1)), float(match.group(2)), float(match.group(3))
            data_dict[current_section].append((freq, real, imag))
    
    if len(data_dict["s11"]) == len(data_dict["s21"]) == len(data_dict["s22"]):
        combined_data = []
        for i in range(len(data_dict["s11"])):
            freq = data_dict["s11"][i][0]
            s11_real, s11_imag = data_dict["s11"][i][1:]
            s21_real, s21_imag = data_dict["s21"][i][1:]
            s22_real, s22_imag = data_dict["s22"][i][1:]
            combined_data.append([freq, s11_real, s11_imag, s21_real, s21_imag, s22_real, s22_imag])
        
        return pd.DataFrame(combined_data, columns=["freq", "s11_real", "s11_imag", "s21_real", "s21_imag", "s22_real", "s22_imag"])
    else:
        return None

def convert_apl_to_csv(directory):
    """Converts all .apl files in the specified directory to .csv files and stores them in a separate directory."""
    output_directory = os.path.join(os.path.dirname(directory), "csv_files")
    os.makedirs(output_directory, exist_ok=True)
    
    for file_name in os.listdir(directory):
        if file_name.endswith(".apl"):
            file_path = os.path.join(directory, file_name)
            df = extract_s_parameters(file_path)
            if df is not None:
                csv_file_name = os.path.splitext(file_name)[0] + "_apl.csv"
                csv_file_path = os.path.join(output_directory, csv_file_name)
                df.to_csv(csv_file_path, index=False)
                print(f"Converted {file_name} to {csv_file_name} and saved in {output_directory}")
            else:
                print(f"Failed to extract data from {file_name}")

if __name__ == "__main__":
    directory = r"/home/raavi/research/ai/wizard_projects/apl_files"
    #directory = r"G:\waveguide_ai\wizard_projects\apl_files"
    convert_apl_to_csv(directory)
