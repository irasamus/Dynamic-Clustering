import os
import numpy as np
import tensorly as tl
from tensorly.decomposition import non_negative_parafac
import sys
import time

# --- Configuration ---
TENSOR_DIR = r"C:\Users\Asus\Documents\Master thesis\Deakin ddataset\output_dir\tensors"
FACTOR_OUTPUT_DIR = r"C:\Users\Asus\Documents\Master thesis\Deakin ddataset\output_dir\factors" # Directory to save the resulting factor matrices

# --- CHOOSE THE LAYER AND ITS RANK ---
# Example: Process the aggregated layer with Rank 5
TENSOR_FILENAME = "local_tcp_count_tensor.npy"
CHOSEN_RANK = 2 # Set this based on your rank estimation analysis for this tensor
LAYER_NAME = "local_tcp_count" # Used for output filenames
# --- END CHOOSE ---


# --- CPD Parameters for Final Decomposition ---
# Use settings that aim for good convergence
CPD_INIT = 'random'        # Random initialization is standard
CPD_TOL = 1e-8             # Use a stricter tolerance for final run
CPD_N_ITER_MAX = 500       # Allow more iterations for convergence
CPD_RANDOM_STATE = 42      # Use a fixed state for reproducibility of this specific run
NUM_RUNS_FOR_BEST = 5      # Optional: Run multiple times and keep best fit

# --- Construct Paths ---
tensor_path = os.path.join(TENSOR_DIR, TENSOR_FILENAME)
os.makedirs(FACTOR_OUTPUT_DIR, exist_ok=True)

# --- Load the Tensor ---
print(f"Loading tensor: {tensor_path}")
try:
    tensor = np.load(tensor_path)
    # Ensure tensor is float for decomposition algorithms
    tensor = tl.tensor(tensor, dtype=tl.float64)
    print(f"Tensor loaded successfully. Shape: {tensor.shape}")
    N, M, T = tensor.shape # Get dimensions N=devices, M=categories, T=time
except FileNotFoundError:
    print(f"FATAL ERROR: Tensor file not found at {tensor_path}")
    sys.exit(1)
except Exception as e:
    print(f"FATAL ERROR loading tensor: {e}")
    sys.exit(1)

# --- Perform Non-Negative CPD ---
print(f"\nPerforming Non-Negative CPD with Rank R={CHOSEN_RANK}...")
print(f"  Max iterations: {CPD_N_ITER_MAX}, Tolerance: {CPD_TOL}")

best_error = float('inf')
best_weights = None
best_factors = None
start_time_cpd = time.time()

# Optional: Run multiple initializations and keep the best result
print(f"  Running {NUM_RUNS_FOR_BEST} initializations to find best fit...")
for run in range(NUM_RUNS_FOR_BEST):
    current_start_time = time.time()
    print(f"    Starting run {run+1}/{NUM_RUNS_FOR_BEST} (random_state={CPD_RANDOM_STATE + run})...")
    try:
        weights, factors = non_negative_parafac(
            tensor,
            rank=CHOSEN_RANK,
            init=CPD_INIT,
            n_iter_max=CPD_N_ITER_MAX,
            tol=CPD_TOL,
            random_state=CPD_RANDOM_STATE + run, # Vary seed for each run
            verbose=False # Set to True or 1 to see convergence details per run
        )

        # Calculate reconstruction error for this run
        tensor_reconstructed = tl.cp_to_tensor((weights, factors))
        error = tl.norm(tensor - tensor_reconstructed) / tl.norm(tensor)
        run_duration = time.time() - current_start_time
        print(f"    Run {run+1} finished in {run_duration:.2f}s. Reconstruction Error: {error:.6f}")

        # Check if this run is better than the best found so far
        if error < best_error:
            print(f"    *** Found new best fit ***")
            best_error = error
            best_weights = weights
            best_factors = factors

    except Exception as e:
        # Handle potential errors during decomposition (e.g., convergence issues)
        print(f"    Run {run+1} FAILED: {e}")

cpd_duration = time.time() - start_time_cpd

if best_factors is None:
     print(f"\nFATAL ERROR: CPD Decomposition failed for all runs.")
     sys.exit(1)

print(f"\nFinished CPD after {NUM_RUNS_FOR_BEST} runs in {cpd_duration:.2f}s.")
print(f"Best Reconstruction Error found: {best_error:.6f}")
print(f"Variance Explained by R={CHOSEN_RANK} model: {(1.0-best_error)*100:.2f}%")


# --- Save the Factor Matrices ---
factor_A = best_factors[0] # Shape: (N x R) -> Devices x Rank
factor_B = best_factors[1] # Shape: M x R -> Categories x Rank
factor_C = best_factors[2] # Shape: T x R -> Time x Rank

print("\nSaving factor matrices...")
try:
    # Define output filenames clearly
    base_output_name = f"{LAYER_NAME}_R{CHOSEN_RANK}"
    path_A = os.path.join(FACTOR_OUTPUT_DIR, f"{base_output_name}_factor_A.npy")
    path_B = os.path.join(FACTOR_OUTPUT_DIR, f"{base_output_name}_factor_B.npy")
    path_C = os.path.join(FACTOR_OUTPUT_DIR, f"{base_output_name}_factor_C.npy")
    path_W = os.path.join(FACTOR_OUTPUT_DIR, f"{base_output_name}_weights.npy") # Save weights too

    np.save(path_A, factor_A)
    print(f"  Saved Factor A (Devices x Rank) to: {path_A} (Shape: {factor_A.shape})")
    np.save(path_B, factor_B)
    print(f"  Saved Factor B (Categories x Rank) to: {path_B} (Shape: {factor_B.shape})")
    np.save(path_C, factor_C)
    print(f"  Saved Factor C (Time x Rank) to: {path_C} (Shape: {factor_C.shape})")
    np.save(path_W, best_weights)
    print(f"  Saved Weights (Rank,) to: {path_W} (Shape: {best_weights.shape})")

except Exception as e:
    print(f"FATAL ERROR saving factor matrices: {e}")
    sys.exit(1)

print("\n--- Script Finished ---")