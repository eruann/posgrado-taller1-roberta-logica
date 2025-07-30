#!/usr/bin/env python
"""
Performs a contrastive analysis of RoBERTa embeddings from the SNLI dataset
to isolate and evaluate logical inference information.

This script implements the following process:
1.  For each specified model layer (e.g., 9-12), it loads pre-computed
    embeddings.
2.  It runs two distinct analyses:
    a) ECN Analysis (k=3): Identifies complete triplets (Entailment,
       Contradiction, Neutral) for a given premise. It computes a 3-point
       centroid from the delta vectors and creates contrastive vectors.
    b) EC Analysis (k=2): Identifies complete pairs (Entailment,
       Contradiction), ignoring Neutral samples. It computes a 2-point
       centroid and creates contrastive vectors.
3.  For each analysis type, it performs k-means clustering on both the new
    contrastive vectors and the original delta vectors.
4.  It evaluates cluster quality using Purity, NMI, and Silhouette Score.
5.  It logs all parameters and metrics to MLflow for tracking.
6.  Finally, it saves the contrastive vectors and presents a summary table
    comparing the clustering performance for both methods across all layers.
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

import mlflow
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix, normalized_mutual_info_score
from tqdm import tqdm

# Add typing for clarity
from typing import Dict, List, Tuple, Optional

# Force GPU-specific imports and check for availability
try:
    import cudf
    import cupy
    import matplotlib.pyplot as plt
    import seaborn as sns
    from cuml.cluster import KMeans
    from cuml.manifold import UMAP
    GPU_AVAILABLE = True
except ImportError:
    print(
        "FATAL: RAPIDS (cuDF, cuML, cuPy) and plotting libraries (matplotlib, seaborn) not found.",
        file=sys.stderr
    )
    print("Please ensure all dependencies are installed.", file=sys.stderr)
    GPU_AVAILABLE = False

# --- Timestamp Helper ---
def ts():
    """Returns a formatted timestamp string part, e.g., '[2024-07-06 15:30:00.123]'."""
    return f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}]"

# --- Column and Dimension Constants ---
PREMISE_DIMS = 768
HYPOTHESIS_DIMS = 768
DELTA_DIMS = 768

PREMISE_COLS = list(range(PREMISE_DIMS))
DELTA_COLS = list(range(
    PREMISE_DIMS + HYPOTHESIS_DIMS,
    PREMISE_DIMS + HYPOTHESIS_DIMS + DELTA_DIMS
))

def parse_args():
    """Parses command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument(
        "--input_dir", type=str, default="data/snli/embeddings/filtered",
        help="Directory containing the input Parquet files."
    )
    p.add_argument(
        "--output_dir", type=str, default="data/snli/embeddings/contrastive",
        help="Directory to save the output contrastive vectors and results."
    )
    p.add_argument(
        "--layers", nargs='+', type=int, default=[9, 10, 11, 12],
        help="List of layers to process."
    )
    p.add_argument(
        "--experiment_name", type=str, default="contrastive-analysis",
        help="Name for the MLflow experiment."
    )
    p.add_argument(
        "--methods", nargs='+', type=str, 
        default=["arithmetic_mean", "geometric_median", "cross_differences"],
        help="List of contrastive methods to use."
    )
    p.add_argument(
        "--plot_sample_size", type=int, default=50000,
        help="Number of points to sample for UMAP visualization. Set to 0 for no sampling."
    )
    return p.parse_args()

def calculate_purity(y_true, y_pred) -> float:
    """Calculates the purity score for clustering."""
    if isinstance(y_true, cupy.ndarray): y_true = y_true.get()
    if isinstance(y_pred, cupy.ndarray): y_pred = y_pred.get()
    contingency_matrix = confusion_matrix(y_true, y_pred)
    row_ind, col_ind = linear_sum_assignment(-contingency_matrix)
    return contingency_matrix[row_ind, col_ind].sum() / np.sum(contingency_matrix)

def _cupy_geometric_median(
    points_by_group: cupy.ndarray, max_iter=100, tol=1e-6
) -> Tuple[cupy.ndarray, bool]:
    """
    Calculates the geometric median for batches of points using CuPy.
    `points_by_group` shape: (num_groups, group_size, dimensions)
    Returns centroids of shape: (num_groups, dimensions)
    """
    # Initialize with arithmetic mean
    centroids = points_by_group.mean(axis=1)
    
    converged = False
    # Wrap the loop with tqdm for a progress bar
    for _ in tqdm(
        range(max_iter), 
        desc="  -> GM Convergence", 
        leave=False, 
        ncols=80,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
    ):
        # Expand centroids to broadcast against all points in each group
        centroids_exp = cupy.expand_dims(centroids, axis=1)
        
        # Calculate L2 distances from each point to its group's current median
        distances = cupy.linalg.norm(points_by_group - centroids_exp, axis=2)
        distances[distances == 0] = 1e-10  # Avoid division by zero
        
        # Calculate weights (inverse distances)
        weights = 1.0 / distances
        weights_exp = cupy.expand_dims(weights, axis=2) # for broadcasting
        
        # Calculate new weighted median
        new_centroids = (points_by_group * weights_exp).sum(axis=1) / weights.sum(axis=1, keepdims=True)
        
        # Check for convergence
        if cupy.all(cupy.linalg.norm(new_centroids - centroids, axis=1) < tol):
            converged = True
            break
        centroids = new_centroids
        
    if not converged:
        print("     [!] Warning: Geometric median did not converge for all groups.")
        
    return centroids, converged

# --- Vector Calculation Methods ---

def _calculate_arithmetic_mean_vectors(
    deltas_by_group: cupy.ndarray,
) -> cupy.ndarray:
    """Calculates contrastive vectors using the arithmetic mean."""
    centroids = deltas_by_group.mean(axis=1, keepdims=True)
    contrastive_vectors = deltas_by_group - centroids
    return contrastive_vectors.reshape(-1, DELTA_DIMS)

def _calculate_geometric_median_vectors(
    deltas_by_group: cupy.ndarray,
) -> Tuple[cupy.ndarray, bool]:
    """Calculates contrastive vectors using the geometric median."""
    centroids, converged = _cupy_geometric_median(deltas_by_group)
    contrastive_vectors = deltas_by_group - cupy.expand_dims(centroids, axis=1)
    return contrastive_vectors.reshape(-1, DELTA_DIMS), converged

def _calculate_cross_difference_vectors(
    deltas_by_group: cupy.ndarray, mode: str
) -> cupy.ndarray:
    """Calculates contrastive vectors using cross-differences."""
    num_groups, group_size, _ = deltas_by_group.shape
    
    if mode == "EC":  # Entailment-Contradiction
        # For each group [d_E, d_C], compute [d_E - d_C, d_C - d_E]
        diffs = deltas_by_group[:, 0, :] - deltas_by_group[:, 1, :]
        # Stack along a new axis and reshape to interleave vectors by group
        # Order: [d(E-C)_g1, d(C-E)_g1, d(E-C)_g2, d(C-E)_g2, ...]
        return cupy.stack((diffs, -diffs), axis=1).reshape(-1, DELTA_DIMS)
        
    elif mode == "ECN":  # Entailment-Contradiction-Neutral
        # For each group [d_E, d_C, d_N], compute [d_E-d_C, d_E-d_N, d_C-d_N]
        d_e, d_c, d_n = deltas_by_group[:, 0, :], deltas_by_group[:, 1, :], deltas_by_group[:, 2, :]
        diff_ec = d_e - d_c
        diff_en = d_e - d_n
        diff_cn = d_c - d_n
        # Stack and reshape to interleave vectors by group
        # Order: [d(E-C)_g1, d(E-N)_g1, d(C-N)_g1, d(E-C)_g2, ...]
        return cupy.stack((diff_ec, diff_en, diff_cn), axis=1).reshape(-1, DELTA_DIMS)

def create_and_save_cluster_plot(
    vectors: cupy.ndarray,
    true_labels: cupy.ndarray,
    pred_labels: cupy.ndarray,
    title: str,
    save_path: Path,
    sample_size: int
):
    """
    Reduces vectors to 2D with UMAP and saves a side-by-side plot of
    true vs. predicted cluster labels.
    """
    print(f"{ts()}      [*] Generating plot: {title}")
    
    plot_vectors = vectors
    plot_true_labels = true_labels
    plot_pred_labels = pred_labels
    
    # Subsample if the dataset is larger than the specified sample size (and size > 0)
    if sample_size > 0 and len(vectors) > sample_size:
        print(f"     [i] Subsampling data from {len(vectors):,} to {sample_size:,} points for visualization.")
        indices = cupy.random.choice(len(vectors), sample_size, replace=False)
        plot_vectors = vectors[indices]
        plot_true_labels = true_labels[indices]
        plot_pred_labels = pred_labels[indices]
    
    # Perform UMAP reduction
    reducer = UMAP(n_components=2, random_state=42, verbose=True)
    vectors_2d = reducer.fit_transform(plot_vectors)
    
    # Convert to CPU for plotting
    vectors_2d_np = vectors_2d.get()
    true_labels_np = plot_true_labels.get()
    pred_labels_np = plot_pred_labels.get()

    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    sns.set_theme(style="whitegrid")
    
    # Plot 1: True Labels
    sns.scatterplot(
        x=vectors_2d_np[:, 0], y=vectors_2d_np[:, 1],
        hue=true_labels_np,
        palette=sns.color_palette("viridis", n_colors=len(np.unique(true_labels_np))),
        legend="full", ax=ax1, s=5
    )
    ax1.set_title("UMAP Projection (True Labels)")
    
    # Plot 2: K-Means Predicted Labels
    sns.scatterplot(
        x=vectors_2d_np[:, 0], y=vectors_2d_np[:, 1],
        hue=pred_labels_np,
        palette=sns.color_palette("viridis", n_colors=len(np.unique(pred_labels_np))),
        legend="full", ax=ax2, s=5
    )
    ax2.set_title("UMAP Projection (K-Means Predicted)")
    
    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save and close
    plt.savefig(save_path)
    plt.close(fig)
    mlflow.log_artifact(str(save_path))

def run_single_analysis(
    mode: str,
    method: str,
    gdf: cudf.DataFrame,
    output_dir: Path,
    layer_num: int,
    plot_sample_size: int
) -> Optional[Dict[str, float]]:
    """
    Runs a single, specific contrastive analysis (e.g., ECN mode with geometric_median).
    """
    # --- 1. Setup based on mode ---
    if mode == "ECN":
        labels_to_include, group_size, k = [0, 1, 2], 3, 3
    elif mode == "EC":
        labels_to_include, group_size, k = [0, 2], 2, 2
    else:
        raise ValueError(f"Invalid mode: {mode}")

    print(f"{ts()}   -> Running analysis: mode={mode}, method={method}")
    
    # --- 2. Filter and Group Data ---
    label_col_name = "label" # SNLI default
    gdf_mode = gdf[gdf[label_col_name].isin(labels_to_include)]
    
    # Use 'premise_id' for grouping, which already contains the hash
    if 'premise_id' not in gdf_mode.columns:
        print(f"     [!] 'premise_id' column not found. Cannot group data. Skipping.")
        return None
        
    group_counts = gdf_mode['premise_id'].value_counts()
    complete_group_ids = group_counts[group_counts == group_size].index

    if len(complete_group_ids) == 0:
        print(f"     [!] No complete {mode} groups found. Skipping.")
        return None
        
    gdf_filtered = gdf_mode[gdf_mode['premise_id'].isin(complete_group_ids)]
    num_groups = len(complete_group_ids)
    print(f"     [*] Found {num_groups:,} complete {mode} groups.")

    # Sort to ensure consistent vector order within groups
    gdf_sorted = gdf_filtered.sort_values(by=['premise_id', label_col_name])
    
    # Reshape delta vectors into (num_groups, group_size, dims)
    # Correctly slice the delta vectors from the full concatenated embeddings
    delta_start_col = PREMISE_DIMS + HYPOTHESIS_DIMS
    delta_end_col = delta_start_col + DELTA_DIMS
    delta_vectors_cupy = gdf_sorted.iloc[:, delta_start_col:delta_end_col].values
    deltas_by_group = delta_vectors_cupy.reshape(num_groups, group_size, DELTA_DIMS)

    # --- 3. Generate Contrastive Vectors ---
    print(f"     [*] Generating contrastive vectors using '{method}' method...")
    
    # Default to 1 (converged), or for methods where it's not applicable
    converged_status = 1.0
    
    if method == "geometric_median":
        contrastive_vectors_flat, converged = _calculate_geometric_median_vectors(
            deltas_by_group
        )
        if not converged:
            converged_status = 0.0
    elif method == "cross_differences":
        contrastive_vectors_flat = _calculate_cross_difference_vectors(
            deltas_by_group, mode
        )
    else:  # arithmetic_mean
        contrastive_vectors_flat = _calculate_arithmetic_mean_vectors(
            deltas_by_group
        )

    # Prepare ground truth labels and premise hashes for the new dataset
    if method == "cross_differences":
        # Create new ground truth labels for cross-diffs
        y_true = cupy.tile(cupy.arange(group_size), num_groups)
        # Repeat the premise ID for each new vector created per group
        premise_hashes_for_output = complete_group_ids.repeat(group_size)
        # Cross-difference vectors don't map to a single hypothesis. Use NA for compatibility.
        hypothesis_ids_for_output = cudf.Series([pd.NA] * len(y_true), dtype='str')
    else:
        # For other methods, vector count matches the sorted dataframe
        y_true = gdf_sorted[label_col_name].values
        premise_hashes_for_output = gdf_sorted['premise_id'].values
        hypothesis_ids_for_output = gdf_sorted['hypothesis_id'].values

    # --- 4. Save Contrastive Dataset ---
    output_file = output_dir / f"contrastive_{method}_{mode.lower()}_layer_{layer_num}.parquet"
    print(f"     [*] Saving vectors to {output_file}...")
    
    # Use 'feature_d' column names for consistency across the pipeline
    feature_cols = [f"feature_{i}" for i in range(DELTA_DIMS)]
    contrastive_df = cudf.DataFrame(contrastive_vectors_flat, columns=feature_cols)
    
    # Use 'gold_label' to be explicit about ground truth and add the hash
    contrastive_df['gold_label'] = y_true
    contrastive_df['premise_hash'] = premise_hashes_for_output
    contrastive_df['hypothesis_id'] = hypothesis_ids_for_output
    contrastive_df.to_parquet(output_file)

    # --- 5. Clustering and Metrics ---
    print("     [*] Performing clustering and calculating metrics...")
    kmeans = KMeans(n_clusters=k, random_state=42)

    preds_contrast = kmeans.fit_predict(contrastive_vectors_flat)
    purity = calculate_purity(y_true, preds_contrast)
    nmi = normalized_mutual_info_score(y_true.get(), preds_contrast.get())

    metrics = {
        "num_groups": num_groups,
        "purity": purity, 
        "nmi": nmi,
    }
    
    # Add convergence status only for the relevant method
    if method == "geometric_median":
        metrics["converged"] = converged_status
    
    # Log metrics to MLflow, prefixing with method name.
    mlflow.log_metrics({f"{method}_{name}": value for name, value in metrics.items()})
    
    # --- 6. Visualization ---
    plot_path = output_dir / f"plot_{method}_{mode.lower()}_layer_{layer_num}.png"
    plot_title = f"'{method.replace('_', ' ').title()}' ({mode}) - Layer {layer_num}"
    create_and_save_cluster_plot(
        contrastive_vectors_flat, y_true, preds_contrast,
        title=plot_title,
        save_path=plot_path,
        sample_size=plot_sample_size
    )
    
    return metrics

def analyze_layer(
    layer_num: int, input_dir: Path, output_dir: Path, methods: List[str], plot_sample_size: int
) -> List[Dict]:
    """
    Performs the full contrastive analysis for a single layer across all
    specified modes and methods.
    """
    print(f"\n--- Processing Layer {layer_num} ---")
    input_file = input_dir / f"embeddings_snli_layer_{layer_num}.parquet"
    if not input_file.exists():
        print(f"  [!] Warning: Input file not found, skipping: {input_file}")
        return []

    print(f"{ts()}   [*] Loading data from {input_file}...")
    gdf = cudf.read_parquet(input_file)
    print(f"  [*] Data loaded successfully. Shape: {gdf.shape}")
    gdf.columns = [str(c) for c in gdf.columns]
    
    layer_results = []
    
    with mlflow.start_run(run_name=f"layer_{layer_num}", nested=True):
        mlflow.log_param("layer", layer_num)
        
        for mode in ["ECN", "EC"]:
            with mlflow.start_run(run_name=f"mode_{mode}", nested=True):
                mlflow.log_param("mode", mode)
                
                for method in methods:
                    with mlflow.start_run(run_name=f"method_{method}", nested=True):
                        mlflow.log_param("method", method)
                        
                        results = run_single_analysis(
                            mode, method, gdf, output_dir, layer_num, plot_sample_size
                        )
            
                        if results:
                            layer_results.append({
                                "layer": layer_num,
                                "mode": mode,
                                "method": method,
                                **results,
                            })
                            
    return layer_results

def main():
    """Main execution function."""
    print(f"{ts()} --- Script Starting ---")
    if not GPU_AVAILABLE: sys.exit(1)
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    mlflow.set_experiment(args.experiment_name)
    with mlflow.start_run(run_name="main_analysis_run"):
        mlflow.log_params(vars(args))

        summary_data = []
        for layer in args.layers:
            layer_results = analyze_layer(
                layer, input_dir, output_dir, args.methods, args.plot_sample_size
            )
            summary_data.extend(layer_results)

    if not summary_data:
        print("\nNo analyses were completed successfully.")
        return

    print("\n--- Final Results Summary ---")
    df_summary = pd.DataFrame(summary_data)
    
    # Dynamically set columns to handle optional 'converged' column
    base_cols = ["layer", "mode", "method", "num_groups", "purity", "nmi"]
    if "converged" in df_summary.columns:
        base_cols.append("converged")
        
    df_summary = df_summary[base_cols].set_index(["layer", "mode", "method"])

    formatters = {
        "num_groups": '{:,.0f}'.format,
        "purity": '{:.4f}'.format,
        "nmi": '{:.4f}'.format,
        "converged": '{:.0f}'.format
    }
    print(df_summary.to_string(formatters=formatters))
    
    results_csv_path = output_dir / "summary_metrics.csv"
    df_summary.to_csv(results_csv_path)
    print(f"\nSummary metrics saved to {results_csv_path}")
    
    # Log the summary CSV as an artifact
    print(f"{ts()} Logging summary_metrics.csv to MLflow...")
    mlflow.log_artifact(str(results_csv_path))

if __name__ == "__main__":
    main() 