import os
import numpy as np
import tensorly as tl
from tensorly.decomposition import non_negative_parafac
import sys
import time

# --- Configuration ---
TENSOR_DIR = r"C:\Users\Asus\Documents\Master thesis\Deakin ddataset\output_dir\tensors"
FACTOR_OUTPUT_DIR = r"C:\Users\Asus\Documents\Master thesis\Deakin ddataset\output_dir\Factors_stability_check"

# --- CHOOSE THE LAYER AND ITS RANK ---
# Example: Check stability for the aggregated layer with Rank 5
TENSOR_FILENAME = "aggregated_ip_count_tensor.npy"
CHOSEN_RANK = 5
LAYER_NAME = "aggregated_ip_count"
# --- END CHOOSE ---

# --- Stability Check Parameters ---
NUM_RUNS_STABILITY = 5 # Number of runs with different random initializations

# --- CPD Parameters ---
CPD_INIT = 'random'
CPD_TOL = 1e-8
CPD_N_ITER_MAX = 500 # Use final iteration count
CPD_BASE_RANDOM_STATE = 42 # Base seed

# --- Construct Paths ---
tensor_path = os.path.join(TENSOR_DIR, TENSOR_FILENAME)
# Create the specific output directory for this layer/rank stability check
stability_output_dir = os.path.join(FACTOR_OUTPUT_DIR, f"{LAYER_NAME}_R{CHOSEN_RANK}_stability")
os.makedirs(stability_output_dir, exist_ok=True)

# --- Load the Tensor ---
print(f"Loading tensor: {tensor_path}")
try:
    tensor = np.load(tensor_path)
    tensor = tl.tensor(tensor, dtype=tl.float64)
    print(f"Tensor loaded successfully. Shape: {tensor.shape}")
except Exception as e:
    print(f"FATAL ERROR loading tensor: {e}")
    sys.exit(1)

# --- Perform Multiple CPD Runs for Stability Check ---
print(f"\nPerforming {NUM_RUNS_STABILITY} Non-Negative CPD runs for Rank R={CHOSEN_RANK} to check stability...")
print(f"  Max iterations: {CPD_N_ITER_MAX}, Tolerance: {CPD_TOL}")

all_run_factors = [] # Optional: store factors in memory if needed for immediate comparison
all_run_errors = []
start_time_stability = time.time()

for run in range(NUM_RUNS_STABILITY):
    current_random_state = CPD_BASE_RANDOM_STATE + run
    print(f"\n  Starting stability run {run+1}/{NUM_RUNS_STABILITY} (random_state={current_random_state})...")
    run_start_time = time.time()

    try:
        weights, factors = non_negative_parafac(
            tensor,
            rank=CHOSEN_RANK,
            init=CPD_INIT,
            n_iter_max=CPD_N_ITER_MAX,
            tol=CPD_TOL,
            random_state=current_random_state,
            verbose=False
        )

        # Calculate reconstruction error
        tensor_reconstructed = tl.cp_to_tensor((weights, factors))
        error = tl.norm(tensor - tensor_reconstructed) / tl.norm(tensor)
        all_run_errors.append(error)
        run_duration = time.time() - run_start_time
        print(f"    Run {run+1} finished in {run_duration:.2f}s. Reconstruction Error: {error:.6f}")

        # --- Save Factors for THIS Run ---
        factor_A = factors[0]
        factor_B = factors[1]
        factor_C = factors[2]

        base_output_name = f"{LAYER_NAME}_R{CHOSEN_RANK}_run{run+1}"
        path_A = os.path.join(stability_output_dir, f"{base_output_name}_factor_A.npy")
        path_B = os.path.join(stability_output_dir, f"{base_output_name}_factor_B.npy")
        path_C = os.path.join(stability_output_dir, f"{base_output_name}_factor_C.npy")
        path_W = os.path.join(stability_output_dir, f"{base_output_name}_weights.npy")

        try:
            np.save(path_A, factor_A)
            np.save(path_B, factor_B)
            np.save(path_C, factor_C)
            np.save(path_W, weights)
            print(f"    Saved factors for run {run+1} to: {stability_output_dir}")
        except Exception as save_e:
            print(f"    ERROR saving factors for run {run+1}: {save_e}")

        # Optional: Append factors if doing immediate comparison later
        # all_run_factors.append(factors)

    except Exception as e:
        print(f"    Run {run+1} FAILED during decomposition: {e}")
        all_run_errors.append(np.nan) # Record failure

stability_duration = time.time() - start_time_stability

# --- Print Summary Statistics ---
print(f"\nFinished {NUM_RUNS_STABILITY} stability runs in {stability_duration:.2f}s.")
valid_errors = [e for e in all_run_errors if not np.isnan(e)]
if valid_errors:
     avg_error = np.mean(valid_errors)
     std_dev_error = np.std(valid_errors)
     min_error = np.min(valid_errors)
     print(f"  Reconstruction Error Stats:")
     print(f"    Average: {avg_error:.6f}")
     print(f"    Std Dev: {std_dev_error:.6f}")
     print(f"    Min Err: {min_error:.6f}")
     if std_dev_error / avg_error > 0.1: # Example threshold for high variability
          print("    Warning: High variability in reconstruction error across runs.")
else:
     print("  No successful runs completed to calculate error stats.")


print("\n--- Script Finished ---")
print(f"Factor matrices for each run saved in: {stability_output_dir}")
print("Next step: Analyze the similarity of factor matrices across runs (e.g., using FMS).")