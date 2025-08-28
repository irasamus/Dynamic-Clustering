import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import sys
import re
import glob # To get file list for time axis

# --- Configuration ---
# Directory where the FINAL chosen factor matrices are saved
FACTOR_DIR = r"C:\Users\Asus\Documents\Master thesis\Deakin ddataset\output_dir\factors" # Assumes you saved the best run factors here
# Directory where the original daily CSVs are (needed to get the date sequence)
CSV_LAYER_DIR = r"C:\Users\Asus\Documents\Master thesis\Deakin ddataset\output_dir\layer_other_local_tcp_count" # Dir for the layer corresponding to factors
# Directory containing metadata
METADATA_DIR = r"C:\Users\Asus\Documents\Master thesis\Deakin ddataset\28013234 (1)\CSVs" #<--- ADJUST IF NEEDED
MAC_ADDRESS_FILE = os.path.join(METADATA_DIR, "macAddresses.csv")
# Base directory for saving plots
PLOT_OUTPUT_DIR = r"C:\Users\Asus\Documents\Master thesis\Deakin ddataset\output_dir\analyze_clustering_plot"

# --- Specify Layer and Rank Being Analyzed ---
# Must match the filenames of the factors you want to load
LAYER_NAME = "local_tcp_count"
CHOSEN_RANK = 2
# --- End Specify ---

# Destination category labels (MUST match the order used during preprocessing)
DESTINATION_CATEGORIES = ["Gateway", "External", "Other Local IP", "Broadcast", "Multicast"]

# --- Create output directory for plots ---
plot_layer_dir = os.path.join(PLOT_OUTPUT_DIR, f"{LAYER_NAME}_R{CHOSEN_RANK}")
os.makedirs(plot_layer_dir, exist_ok=True)

# --- Load Metadata (Device Names/MACs) ---
print("Loading metadata...")
try:
    mac_df = pd.read_csv(MAC_ADDRESS_FILE)
    mac_column_name = 'MAC Address'
    if mac_column_name not in mac_df.columns: raise ValueError(f"Column '{mac_column_name}' not found")
    known_macs = mac_df[mac_column_name].str.lower().tolist() # Ordered list

    device_name_column = 'Device Name'
    if device_name_column in mac_df.columns:
        mac_to_name = pd.Series(mac_df[device_name_column].values, index=mac_df[mac_column_name].str.lower()).to_dict()
        device_labels = [mac_to_name.get(mac, mac) for mac in known_macs] # Use name, fallback to MAC
        print(f"Using device names from '{device_name_column}'.")
    else:
        print(f"Warning: Column '{device_name_column}' not found. Using MAC addresses as labels.")
        device_labels = known_macs # Use MAC addresses if names not found

    num_iot_devices = len(known_macs)
    if num_iot_devices == 0: raise ValueError("No MAC addresses loaded.")

except Exception as e:
    print(f"FATAL ERROR loading metadata: {e}")
    sys.exit(1)

# --- Load Factor Matrices ---
print(f"\nLoading factor matrices for {LAYER_NAME}, Rank R={CHOSEN_RANK}...")
try:
    base_name = f"{LAYER_NAME}_R{CHOSEN_RANK}" # Assumes final factors don't have '_runX'
    # Adjust filenames below if you saved the best run with a specific run number
    path_A = os.path.join(FACTOR_DIR, f"{base_name}_factor_A.npy")
    path_B = os.path.join(FACTOR_DIR, f"{base_name}_factor_B.npy")
    path_C = os.path.join(FACTOR_DIR, f"{base_name}_factor_C.npy")
    path_W = os.path.join(FACTOR_DIR, f"{base_name}_weights.npy")

    factor_A = np.load(path_A) # Shape: N x R
    factor_B = np.load(path_B) # Shape: M x R
    factor_C = np.load(path_C) # Shape: T x R
    weights = np.load(path_W)  # Shape: R,

    print("Factors loaded successfully.")
    print(f"  Factor A (Device): {factor_A.shape}")
    print(f"  Factor B (Category): {factor_B.shape}")
    print(f"  Factor C (Time): {factor_C.shape}")
    print(f"  Weights: {weights.shape}")

    # Basic validation
    if factor_A.shape != (num_iot_devices, CHOSEN_RANK) or \
       factor_B.shape != (len(DESTINATION_CATEGORIES), CHOSEN_RANK) or \
       factor_C.shape[1] != CHOSEN_RANK or \
       weights.shape != (CHOSEN_RANK,):
        raise ValueError("Factor matrix dimensions do not match expected N, M, T, R.")
    T = factor_C.shape[0] # Get number of time steps from Factor C

except FileNotFoundError as e:
    print(f"FATAL ERROR: Could not find factor file: {e}. Ensure files are named correctly and in {FACTOR_DIR}.")
    sys.exit(1)
except Exception as e:
    print(f"FATAL ERROR loading factor matrices: {e}")
    sys.exit(1)

# --- Get Time Axis Labels (Dates) ---
# Load dates from the CSV filenames in the corresponding layer directory
print(f"\nLoading dates from CSV files in: {CSV_LAYER_DIR}")
csv_files = glob.glob(os.path.join(CSV_LAYER_DIR, "*.csv"))
dates = []
if not csv_files:
    print("Warning: No CSV files found to determine dates. Using numerical time steps.")
    time_labels = np.arange(T)
else:
    file_info_list = []
    for f_path in csv_files:
        filename = os.path.basename(f_path)
        match = re.match(r'(\d{4}-\d{2}-\d{2})\.csv', filename)
        if match: file_info_list.append({"date": match.group(1)})
    file_info_list.sort(key=lambda x: x['date'])
    dates = [info['date'] for info in file_info_list]
    if len(dates) != T:
        print(f"Warning: Number of dates ({len(dates)}) does not match Factor C time dimension ({T}). Using numerical time steps.")
        time_labels = np.arange(T)
    else:
        # Convert dates to datetime objects for plotting (optional but nicer)
        try:
            time_labels = pd.to_datetime(dates)
            print(f"Successfully loaded {len(time_labels)} dates for time axis.")
        except Exception as e:
            print(f"Warning: Could not convert dates to datetime objects: {e}. Using numerical time steps.")
            time_labels = np.arange(T)


# --- Analyze and Visualize Factors ---

print("\n--- Analyzing and Visualizing Factors ---")

# 1. Factor B: Destination Category Profiles per Cluster
plt.figure(figsize=(10, 6))
# Use weights to scale factor B columns for relative importance (optional)
# weighted_B = factor_B * weights[np.newaxis, :] # Or normalize columns of B
# df_B = pd.DataFrame(weighted_B, index=DESTINATION_CATEGORIES, columns=[f'Cluster {r+1}' for r in range(CHOSEN_RANK)])
# Use raw factor B for direct profile
df_B = pd.DataFrame(factor_B, index=DESTINATION_CATEGORIES, columns=[f'Cluster {r+1}' for r in range(CHOSEN_RANK)])
sns.heatmap(df_B, annot=True, fmt=".2f", cmap="viridis")
plt.title(f'Factor B: Destination Category Profile per Cluster (R={CHOSEN_RANK}) - {LAYER_NAME}')
plt.xlabel('Cluster (Component)')
plt.ylabel('Destination Category')
plt.tight_layout()
plt.savefig(os.path.join(plot_layer_dir, "factor_B_heatmap.png"))
print("Saved Factor B heatmap.")
# plt.show()
plt.close()

# 2. Factor C: Temporal Activity Profile per Cluster
plt.figure(figsize=(12, 8))
num_clusters = CHOSEN_RANK
rows = int(np.ceil(num_clusters / 2))
cols = 2
for r in range(num_clusters):
    plt.subplot(rows, cols, r + 1)
    plt.plot(time_labels, factor_C[:, r], label=f'Cluster {r+1}')
    plt.title(f'Cluster {r+1} Temporal Activity')
    plt.ylabel('Loading')
    plt.xlabel('Time')
    plt.grid(True, linestyle=':')
    # Improve date formatting if time_labels are datetimes
    if isinstance(time_labels[0], pd.Timestamp):
         plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1)) # Show month ticks
         plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
         plt.xticks(rotation=45, ha='right')

plt.suptitle(f'Factor C: Temporal Activity Profiles (R={CHOSEN_RANK}) - {LAYER_NAME}', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(plot_layer_dir, "factor_C_temporal.png"))
print("Saved Factor C temporal plots.")
# plt.show()
plt.close()

# 3. Factor A: Device Membership Profile per Cluster
plt.figure(figsize=(8, 10))
# Use weights to scale factor A columns (optional)
# weighted_A = factor_A * weights[np.newaxis, :]
# df_A = pd.DataFrame(weighted_A, index=device_labels, columns=[f'Cluster {r+1}' for r in range(CHOSEN_RANK)])
# Use raw factor A
df_A = pd.DataFrame(factor_A, index=device_labels, columns=[f'Cluster {r+1}' for r in range(CHOSEN_RANK)])
sns.heatmap(df_A, annot=False, cmap="viridis") # Annot=True might be too crowded
plt.title(f'Factor A: Device Membership Profile (R={CHOSEN_RANK}) - {LAYER_NAME}')
plt.xlabel('Cluster (Component)')
plt.ylabel('Device')
plt.tight_layout()
plt.savefig(os.path.join(plot_layer_dir, "factor_A_heatmap.png"))
print("Saved Factor A heatmap.")
# plt.show()
plt.close()

# 4. Assign Communities Over Time & Visualize
print("\nAssigning communities over time...")
# Reconstruct approximate contribution: W_i * A_ir * B_jr * C_tr (sum over j)
# Simplified approach: Assign based on max(A_ir * C_tr) for each device i at time t
# Multiply A and C element-wise for each component, then find max component per device/time
# Incorporate weights: A_weighted = factor_A * weights[np.newaxis, :]
# contribution = A_weighted @ factor_C.T # Gives N x T matrix if B sums to 1 per component? No.
# Let's use the simpler A[i,r] * C[t,r] approach first

contributions = np.zeros((num_iot_devices, T, CHOSEN_RANK))
for r in range(CHOSEN_RANK):
     # Outer product of device vector and time vector for component r, scaled by weight
     component_activity = np.outer(factor_A[:, r], factor_C[:, r]) * weights[r] # Shape N x T
     contributions[:, :, r] = component_activity

# Find the index (cluster number 0 to R-1) with the maximum contribution for each device/time
community_assignment = np.argmax(contributions, axis=2) # Shape N x T

print("Visualizing community assignments...")
plt.figure(figsize=(15, 8))
# --- CHANGE: Use imshow instead of pcolormesh ---
cmap = plt.get_cmap('viridis', CHOSEN_RANK) # Choose colormap with R distinct colors
# Transpose community_assignment so time is on x-axis, devices on y-axis matching plot layout
im = plt.imshow(community_assignment.T + 1, # Data: T x N, add 1 for colors
                aspect='auto',          # Adjust aspect ratio automatically
                cmap=cmap,
                interpolation='nearest', # Avoid blurring discrete categories
                origin='lower',         # Place device 0 at the bottom
                extent=[time_labels[0], time_labels[-1], -0.5, num_iot_devices - 0.5]) # Set extent for axes
                # Note: Extent assumes time_labels can be treated numerically for limits, may need adjustment if datetimes

plt.yticks(np.arange(num_iot_devices), device_labels) # Set ticks at center of cells
plt.xlabel('Time')
plt.ylabel('Device')
plt.title(f'Dynamic Community Assignments (R={CHOSEN_RANK}) - {LAYER_NAME}')

# Add colorbar
cbar = plt.colorbar(im, ticks=np.arange(CHOSEN_RANK) + 1) # Ticks 1, 2, ..., R
cbar.set_ticklabels([f'Cluster {r+1}' for r in range(CHOSEN_RANK)])
cbar.set_label('Assigned Community')

# Improve date formatting if time_labels are datetimes
if isinstance(time_labels[0], pd.Timestamp):
    plt.gca().xaxis_date() # Treat x-axis as dates
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45, ha='right')
else: # Handle numerical time labels
    pass # Default numerical axis is fine

plt.tight_layout()
plt.savefig(os.path.join(plot_layer_dir, "community_evolution.png"))
print("Saved community evolution plot.")
plt.show()
plt.close()
# --- END CHANGE ---

print("\n--- Script Finished ---")
print("Interpret the plots:")
print(" - Factor B Heatmap: Shows which destination categories define each cluster.")
print(" - Factor C Plots: Show when each cluster is active.")
print(" - Factor A Heatmap: Shows which devices belong most strongly to which clusters overall.")
print(" - Community Evolution Plot: Shows how device cluster assignments change over time.")