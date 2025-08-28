import os
import numpy as np
import tensorly as tl
from tensorly.decomposition import non_negative_parafac
import matplotlib.pyplot as plt
import time
import sys

# --- Configuration ---
TENSOR_DIR = r"C:\Users\Asus\Documents\Master thesis\Deakin ddataset\output_dir\tensors"
TENSOR_FILENAME = "local_tcp_count_tensor.npy"
TENSOR_PATH = os.path.join(TENSOR_DIR, TENSOR_FILENAME)

# --- Rank Estimation Parameters ---
RANK_RANGE = range(2, 9)
CPD_INIT = 'random'
CPD_TOL = 1e-7
CPD_N_ITER_MAX = 100
CPD_RANDOM_STATE = 42

# --- Load the Tensor ---
print(f"Loading tensor: {TENSOR_PATH}")
try:
    tensor = np.load(TENSOR_PATH)
    tensor = tl.tensor(tensor, dtype=tl.float64)
    print(f"Tensor loaded successfully. Shape: {tensor.shape}")
except FileNotFoundError:
    print(f"FATAL ERROR: Tensor file not found at {TENSOR_PATH}")
    sys.exit(1)
except Exception as e:
    print(f"FATAL ERROR loading tensor: {e}")
    sys.exit(1)

# --- Estimate Rank ---
print(f"\nEstimating optimal rank R in range {list(RANK_RANGE)}...")
reconstruction_errors = []

start_time_estimation = time.time()

for r in RANK_RANGE:
    print(f"  Testing Rank R={r}...")
    rank_start_time = time.time()
    try:
        weights, factors = non_negative_parafac(
            tensor,
            rank=r,
            init=CPD_INIT,
            n_iter_max=CPD_N_ITER_MAX,
            tol=CPD_TOL,
            random_state=CPD_RANDOM_STATE,
            verbose=False
        )

        tensor_reconstructed = tl.cp_to_tensor((weights, factors))
        error = tl.norm(tensor - tensor_reconstructed) / tl.norm(tensor)
        reconstruction_errors.append(error)
        print(f"    Reconstruction Error: {error:.4f}")

    except Exception as e:
        print(f"    ERROR during decomposition for R={r}: {e}")
        reconstruction_errors.append(np.nan)

    rank_duration = time.time() - rank_start_time
    print(f"    Rank {r} processing time: {rank_duration:.2f}s")

estimation_duration = time.time() - start_time_estimation
print(f"\nFinished rank estimation in {estimation_duration:.2f}s.")

# --- Plot Results ---
print("\nGenerating plot...")
fig, ax = plt.subplots(figsize=(10, 6))

variance_explained = [(1.0 - err) * 100 if not np.isnan(err) else np.nan for err in reconstruction_errors]
ax.plot(list(RANK_RANGE), variance_explained, marker='o', linestyle='-', color='tab:red', label='Variance Explained')
ax.set_xlabel('Rank (R)')
ax.set_ylabel('Variance Explained (%)', color='tab:red')
ax.tick_params(axis='y', labelcolor='tab:red')
ax.grid(True, axis='y', linestyle=':')
ax.set_title(f'Rank Estimation for {TENSOR_FILENAME}')
ax.legend(loc='center right')

plot_filename = os.path.splitext(TENSOR_FILENAME)[0] + "_rank_estimation.png"
plot_path = os.path.join(TENSOR_DIR, plot_filename)
try:
    plt.savefig(plot_path)
    print(f"Saved estimation plot to: {plot_path}")
except Exception as e:
    print(f"Error saving plot: {e}")

plt.show()

print("\n--- Script Finished ---")