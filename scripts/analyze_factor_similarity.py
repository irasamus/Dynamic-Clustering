import os
import glob
import numpy as np
import itertools
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment # For optimal column matching
import sys
import re

# --- Configuration ---
# --- MUST MATCH the output directory used in 2a_check_stability.py ---
FACTOR_STABILITY_DIR_BASE = r"C:\Users\Asus\Documents\Master thesis\Deakin ddataset\output_dir\Factors_stability_check"

# --- CHOOSE THE LAYER AND RANK TO ANALYZE ---
# Example: Analyze stability for aggregated layer with Rank 5
LAYER_NAME = "aggregated_ip_count"
CHOSEN_RANK = 5
# --- END CHOOSE ---

# --- Construct the specific directory path ---
stability_dir = os.path.join(FACTOR_STABILITY_DIR_BASE, f"{LAYER_NAME}_R{CHOSEN_RANK}_stability")

# --- Parameters ---
NUM_RUNS_EXPECTED = 5 # Should match NUM_RUNS_STABILITY used in the previous script
SIMILARITY_THRESHOLD = 0.8 # Define a threshold for considering factors "similar"

# --- Load Factors from All Runs ---
print(f"Loading factors from stability runs in: {stability_dir}")

if not os.path.isdir(stability_dir):
    print(f"FATAL ERROR: Directory not found: {stability_dir}")
    sys.exit(1)

factors_A = {}
factors_B = {}
factors_C = {}
run_numbers_found = []

# Find factor files for each run
for run_num in range(1, NUM_RUNS_EXPECTED + 1):
    run_id = f"run{run_num}"
    base_name = f"{LAYER_NAME}_R{CHOSEN_RANK}_{run_id}"
    path_A = os.path.join(stability_dir, f"{base_name}_factor_A.npy")
    path_B = os.path.join(stability_dir, f"{base_name}_factor_B.npy")
    path_C = os.path.join(stability_dir, f"{base_name}_factor_C.npy")

    if os.path.exists(path_A) and os.path.exists(path_B) and os.path.exists(path_C):
        try:
            factors_A[run_id] = np.load(path_A)
            factors_B[run_id] = np.load(path_B)
            factors_C[run_id] = np.load(path_C)
            run_numbers_found.append(run_id)
            print(f"  Loaded factors for {run_id}")
        except Exception as e:
            print(f"  Warning: Could not load factors for {run_id}: {e}")
    else:
        print(f"  Warning: Factor files not found for {run_id}. Skipping.")

if len(run_numbers_found) < 2:
    print("\nFATAL ERROR: Need at least two successful runs to compare stability.")
    sys.exit(1)

print(f"\nFound factors for {len(run_numbers_found)} runs: {run_numbers_found}")

# --- Compare Factors Between Runs using Cosine Similarity ---

def match_factors_and_get_similarity(factors1, factors2):
    """
    Matches columns between two factor matrices based on maximum cosine similarity
    using the Hungarian algorithm and returns the average similarity of matched pairs.
    Assumes factors matrices have shape (Dimension x Rank R).
    """
    if factors1.shape != factors2.shape:
        print("Warning: Factor matrices have different shapes, cannot compare.")
        return 0.0, {}

    # Calculate pairwise cosine similarity between columns of the two matrices
    # sim_matrix[i, j] = similarity between column i of factors1 and column j of factors2
    sim_matrix = cosine_similarity(factors1.T, factors2.T) # Use transpose to compare columns

    # Use Hungarian algorithm (linear_sum_assignment) to find the optimal matching
    # We want to maximize similarity, so we work with (1 - similarity) or negate it for cost.
    # cost_matrix = 1.0 - sim_matrix
    # The algorithm finds assignments that minimize cost.
    # row_ind are indices for factors1, col_ind are indices for factors2
    try:
        # Negate similarity for maximization -> minimization problem
        row_ind, col_ind = linear_sum_assignment(-sim_matrix)
        # The matched similarity scores are sim_matrix[row_ind, col_ind]
        matched_similarity_scores = sim_matrix[row_ind, col_ind]
        average_similarity = np.mean(matched_similarity_scores)
        # Store the matching: {factors1_col_idx : factors2_col_idx}
        matching_dict = dict(zip(row_ind, col_ind))
        return average_similarity, matching_dict, matched_similarity_scores
    except ValueError as e:
        print(f"Warning: linear_sum_assignment failed: {e}. Returning 0 similarity.")
        return 0.0, {}, []


print("\n--- Factor Stability Analysis (Average Cosine Similarity of Matched Factors) ---")

# Compare all pairs of runs
all_avg_similarities = {'A': [], 'B': [], 'C': []}

for run1_id, run2_id in itertools.combinations(run_numbers_found, 2):
    print(f"\nComparing {run1_id} vs {run2_id}:")

    # Compare Factor A (Devices)
    avg_sim_A, _, _ = match_factors_and_get_similarity(factors_A[run1_id], factors_A[run2_id])
    all_avg_similarities['A'].append(avg_sim_A)
    print(f"  Factor A (Devices) Avg Similarity: {avg_sim_A:.4f}")

    # Compare Factor B (Categories)
    avg_sim_B, _, _ = match_factors_and_get_similarity(factors_B[run1_id], factors_B[run2_id])
    all_avg_similarities['B'].append(avg_sim_B)
    print(f"  Factor B (Categories) Avg Similarity: {avg_sim_B:.4f}")

    # Compare Factor C (Time)
    avg_sim_C, _, _ = match_factors_and_get_similarity(factors_C[run1_id], factors_C[run2_id])
    all_avg_similarities['C'].append(avg_sim_C)
    print(f"  Factor C (Time) Avg Similarity: {avg_sim_C:.4f}")

# --- Overall Summary ---
print("\n--- Overall Stability Summary ---")
overall_avg_A = np.mean(all_avg_similarities['A']) if all_avg_similarities['A'] else np.nan
overall_avg_B = np.mean(all_avg_similarities['B']) if all_avg_similarities['B'] else np.nan
overall_avg_C = np.mean(all_avg_similarities['C']) if all_avg_similarities['C'] else np.nan

print(f"Average Factor A Similarity across run pairs: {overall_avg_A:.4f}")
print(f"Average Factor B Similarity across run pairs: {overall_avg_B:.4f}")
print(f"Average Factor C Similarity across run pairs: {overall_avg_C:.4f}")

if overall_avg_A > SIMILARITY_THRESHOLD and \
   overall_avg_B > SIMILARITY_THRESHOLD and \
   overall_avg_C > SIMILARITY_THRESHOLD:
   print(f"\nFactors appear STABLE across runs (Avg Similarity > {SIMILARITY_THRESHOLD:.2f}).")
else:
   print(f"\nWarning: Factors show INSTABILITY across runs (Avg Similarity <= {SIMILARITY_THRESHOLD:.2f}). Consider re-evaluating Rank R.")

print("\n--- Script Finished ---")