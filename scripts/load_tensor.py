import os
import glob
import re
import numpy as np
import pandas as pd
import sys

# --- Configuration ---
LAYER_CSV_DIR = r"C:\Users\Asus\Documents\Master thesis\Deakin ddataset\output_dir\layer_other_local_tcp_count" 

# Output file path for the resulting tensor
OUTPUT_TENSOR_FILENAME = "local_tcp_count_tensor.npy" # <--- CHANGE THIS based on the layer being processed
OUTPUT_TENSOR_DIR = r"C:\Users\Asus\Documents\Master thesis\Deakin ddataset\output_dir\tensors" # Directory to save the tensor file

# Expected dimensions (verify these match your data)
EXPECTED_NUM_DEVICES = 24 # Number of rows (N)
EXPECTED_NUM_CATEGORIES = 5 # Number of columns (M) - Gateway, External, Other Local IP, Broadcast, Multicast

# --- Create output directory ---
os.makedirs(OUTPUT_TENSOR_DIR, exist_ok=True)
output_tensor_path = os.path.join(OUTPUT_TENSOR_DIR, OUTPUT_TENSOR_FILENAME)

# --- Find and Sort CSV Files ---
print(f"Scanning for daily CSV files in: {LAYER_CSV_DIR}")
csv_files = glob.glob(os.path.join(LAYER_CSV_DIR, "*.csv"))

if not csv_files:
    print(f"FATAL ERROR: No CSV files found in the specified directory.")
    sys.exit(1)

# Extract dates and sort
file_info_list = []
for f_path in csv_files:
    filename = os.path.basename(f_path)
    match = re.match(r'(\d{4}-\d{2}-\d{2})\.csv', filename) # Match YYYY-MM-DD.csv exactly
    if match:
        file_info_list.append({"path": f_path, "date": match.group(1)})
    else:
        print(f"Warning: Skipping file with unexpected name format: {filename}")

if not file_info_list:
    print(f"FATAL ERROR: No CSV files with format YYYY-MM-DD.csv found.")
    sys.exit(1)

file_info_list.sort(key=lambda x: x['date'])
num_time_steps = len(file_info_list)
print(f"Found {num_time_steps} daily CSV files to stack.")

# --- Load Matrices and Stack into Tensor ---
daily_matrices = []
expected_shape = (EXPECTED_NUM_DEVICES, EXPECTED_NUM_CATEGORIES)

print("Loading and stacking matrices...")
for i, file_info in enumerate(file_info_list):
    f_path = file_info['path']
    date = file_info['date']
    # print(f"  Loading {os.path.basename(f_path)}...") # Optional verbose print
    try:
        # index_col=0 assumes the first column is the MAC Address/Identifier index
        df = pd.read_csv(f_path, index_col=0)

        # Validation
        if df.shape != expected_shape:
            print(f"  WARNING: Skipping {os.path.basename(f_path)}. Expected shape {expected_shape}, but got {df.shape}.")
            continue # Skip this file if shape is wrong

        # Ensure data is numeric (it should be integers for counts)
        # Convert to numeric, coercing errors (though should ideally be clean)
        df_numeric = df.apply(pd.to_numeric, errors='coerce')
        if df_numeric.isnull().values.any():
             print(f"  WARNING: Non-numeric data found in {os.path.basename(f_path)} after coercion. Filling NaNs with 0.")
             df_numeric = df_numeric.fillna(0)

        # Extract NumPy array and ensure correct dtype
        matrix = df_numeric.values.astype(np.int64) # Or float64 if needed later
        daily_matrices.append(matrix)

    except Exception as e:
        print(f"  ERROR processing file {os.path.basename(f_path)}: {e}. Skipping.")
        continue # Skip problematic file

# Check if any matrices were successfully loaded
if not daily_matrices:
     print(f"FATAL ERROR: No matrices were successfully loaded and validated.")
     sys.exit(1)

# Stack along the time axis (axis=2)
try:
    # Ensure all loaded matrices have the same shape before stacking
    first_shape = daily_matrices[0].shape
    if not all(m.shape == first_shape for m in daily_matrices):
        print("FATAL ERROR: Not all loaded matrices have the same shape. Cannot stack.")
        # Add more debug here to find the offending file/shape if needed
        sys.exit(1)

    tensor = np.stack(daily_matrices, axis=2)
    print(f"Successfully stacked matrices into tensor with shape: {tensor.shape}")
    # Expected shape: (N, M, T) -> (24, 5, num_loaded_files)

except Exception as e:
    print(f"FATAL ERROR stacking matrices into tensor: {e}")
    sys.exit(1)


# --- Save the Tensor ---
print(f"\nSaving tensor to: {output_tensor_path}")
try:
    np.save(output_tensor_path, tensor)
    print("Tensor saved successfully.")
except Exception as e:
    print(f"FATAL ERROR saving tensor: {e}")
    sys.exit(1)

print("\n--- Script Finished ---")