import numpy as np
import os

# --- Configuration ---
TENSOR_DIR = r"C:\Users\Asus\Documents\Master thesis\Deakin ddataset\output_dir\tensors" # Or the full path used in the stacking script
TENSOR_FILENAME = "aggregated_ip_count_tensor.npy" # The tensor file to check
TENSOR_PATH = os.path.join(TENSOR_DIR, TENSOR_FILENAME)

EXPECTED_NUM_DEVICES = 24
EXPECTED_NUM_CATEGORIES = 5
EXPECTED_NUM_TIMESTEPS = 119 # Should match the number of CSVs processed

# --- Load the tensor ---
try:
    tensor = np.load(TENSOR_PATH)
    print(f"Successfully loaded tensor from: {TENSOR_PATH}")
except Exception as e:
    print(f"ERROR loading tensor: {e}")
    exit()

# --- Perform Checks ---
print("\n--- Sanity Checks ---")

# 1. Check Shape
print(f"1. Tensor Shape:")
actual_shape = tensor.shape
print(f"   Actual: {actual_shape}")
expected_shape = (EXPECTED_NUM_DEVICES, EXPECTED_NUM_CATEGORIES, EXPECTED_NUM_TIMESTEPS)
if actual_shape == expected_shape:
    print(f"   Matches expected shape ({expected_shape}). OK.")
else:
    print(f"   !!! WARNING: Shape mismatch! Expected {expected_shape}. !!!")

# 2. Check Data Type
print(f"\n2. Tensor Data Type:")
actual_dtype = tensor.dtype
print(f"   Actual: {actual_dtype}")
# Expecting int64 for counts, maybe float64 if bytes were aggregated
if actual_dtype == np.int64:
     print(f"   Looks like integer counts. OK.")
elif actual_dtype == np.float64:
     print(f"   Looks like floating point (maybe bytes or normalized data?). Check if expected.")
else:
     print(f"   !!! WARNING: Unexpected data type: {actual_dtype}. !!!")

# 3. Check for Negative Values (shouldn't exist for counts/bytes)
print(f"\n3. Check for Negative Values:")
if np.any(tensor < 0):
    print(f"   !!! WARNING: Negative values found in the tensor! This should not happen for counts/bytes. !!!")
else:
    print(f"   No negative values found. OK.")

# 4. Check Sums (compare with previous script output if available)
print(f"\n4. Basic Sums:")
total_sum = np.sum(tensor)
print(f"   Total sum across all devices, categories, time: {total_sum}")
# Compare this total sum to the sum of 'packets_aggregated_this_file' across all runs of the previous script, they should match.

# Sum over time (axis=2) to get total counts per device/category across all days
total_per_device_category = np.sum(tensor, axis=2)
print(f"   Shape after summing over time: {total_per_device_category.shape}") # Should be (N, M) -> (24, 5)
# print("   Total counts per device/category:\n", total_per_device_category) # Uncomment to see details

# 5. Spot Check a Slice (e.g., first device, first day)
print(f"\n5. Spot Check Slice (Device 0, Day 0):")
try:
    first_day_slice = tensor[:, :, 0] # Get the matrix for the first day (time index 0)
    first_device_first_day = first_day_slice[0, :] # Get data for device 0 on day 0
    print(f"   Data for Device 0 on the first day (Categories: Gateway, External, OtherLocal, Bcast, Mcast):")
    print(f"   {first_device_first_day}")
    # You could manually compare this array to the first row of numerical data
    # in the corresponding daily CSV file (e.g., 2023-05-15_aggregated_ip_count.csv)
    # Note: Remember the CSV has the MAC address, skip that column when comparing.
except IndexError:
    print("   Could not extract slice (tensor might be empty or have wrong dimensions).")

print("\n--- Checks Complete ---")