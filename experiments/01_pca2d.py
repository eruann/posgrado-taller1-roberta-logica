#!/usr/bin/env python
"""
experiments/02_pca2d.py – GPU PCA to 2D and cone visualization
===============================================================
Usage:
------
python experiments/02_pca2d.py \
       --source_path data/snli_train_embeddings.parquet \
       --out_dir data/pca2d_results \
       --precision fp32 \
       --batch_size 10000

Parameters:
----------
--source_path: Parquet input with columns 'vector','label'
--out_dir: Directory to write outputs (Parquet with 2D coords, PNG scatter)
--experiment_name: Name of the MLflow experiment
--dataset: Name of the dataset (e.g., snli, mnli, etc)
--precision: Floating-point precision for input data (choices: fp32, fp16, default: fp32)
--batch_size: Batch size for chunked GPU PCA (default: 10000)

Features:
---------
* Runs PCA to 2 components on GPU via cuML.
* Processes data in chunks to fit GPU memory.
* Saves a Parquet file with columns ['pc1','pc2','label'].
* Generates a scatter plot of PC1 vs PC2 colored by SNLI label.
* Logs parameters, metrics, and artifacts in MLflow.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
# Use Agg backend to avoid GUI errors
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
import mlflow

# ---------------------------------------------------------------------------
# GPU-only PCA (cuML)
# ---------------------------------------------------------------------------
try:
    import cupy as cp
    from cuml.decomposition import PCA as GPU_PCA
except ImportError:
    raise ImportError(
        "cuML GPU PCA not available: install cuml and ensure CUDA/GPU setup is correct."
    )

# Force MLflow to use local mlruns directory
tracking_dir = Path.cwd().joinpath("mlruns")
mlflow.set_tracking_uri(tracking_dir.as_uri())

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="GPU PCA to 2D and scatter plot")
    parser.add_argument(
        "--source_path", dest="inp", required=True,
        help="Parquet input containing columns 'vector','label'"
    )
    parser.add_argument(
        "--out_dir", required=True,
        help="Directory where outputs (Parquet, PNG) will be written"
    )
    parser.add_argument(
        "--experiment_name", default="pca_2d_snli",
        help="Name of the MLflow experiment"
    )
    parser.add_argument(
        "--dataset", default="snli",
        help="Name of the dataset (e.g., snli, mnli, etc)"
    )
    parser.add_argument(
        "--precision", choices=["fp32", "fp16"], default="fp32",
        help="Floating-point precision for input data"
    )
    parser.add_argument(
        "--batch_size", type=int, default=10000,
        help="Batch size for processing in chunks on GPU"
    )
    parser.add_argument("--layer_num", type=int, default=12, help="Layer number to use (default: 12)")
    return parser.parse_args()

# ---------------------------------------------------------------------------
# Estimate GPU memory requirements (rough)
# ---------------------------------------------------------------------------
def estimate_memory_requirements(n_samples, n_features, n_components, dtype=np.float32):
    # Input data size (bytes)
    input_size = n_samples * n_features * dtype().itemsize
    # Output data size (bytes)
    output_size = n_samples * n_components * dtype().itemsize
    # PCA components matrix size (bytes)
    components_size = n_features * n_components * dtype().itemsize
    total_gpu = input_size + output_size + components_size
    return {
        'input_size_gb': input_size / (1024**3),
        'output_size_gb': output_size / (1024**3),
        'components_size_gb': components_size / (1024**3),
        'total_gpu_gb': total_gpu / (1024**3)
    }

def get_optimal_batch_size(n_features, n_components, available_memory_gb=56):
    # Rough estimate: each sample uses (n_features + n_components)*4 bytes (fp32)
    memory_per_sample = (n_features + n_components) * 4
    safe_memory = available_memory_gb * 0.8 * (1024**3)
    optimal_batch = int(safe_memory / memory_per_sample)
    return min(optimal_batch, 10000)

# ---------------------------------------------------------------------------
# Process data in GPU chunks
# ---------------------------------------------------------------------------
def process_in_chunks(X_np, batch_size, pca):
    n_rows = X_np.shape[0]
    n_chunks = (n_rows + batch_size - 1) // batch_size
    results = []
    for i in tqdm(range(0, n_rows, batch_size),
                 desc="Processing chunks", total=n_chunks):
        chunk = X_np[i : i + batch_size]
        chunk_gpu = cp.asarray(chunk)
        chunk_red = pca.transform(chunk_gpu)
        results.append(cp.asnumpy(chunk_red))
        # Clear GPU memory
        del chunk_gpu
        cp.get_default_memory_pool().free_all_blocks()
    return np.vstack(results)

# ---------------------------------------------------------------------------
# Plot PC1 vs PC2 scatter
# ---------------------------------------------------------------------------
def plot_pc2d(pc2d: np.ndarray, labels: np.ndarray, out_png: Path):
    """
    pc2d: (N,2) array of principal component coordinates
    labels: (N,) array of integers {0,1,2}
    """
    # Map labels to colors
    color_map = np.array(["C0","C1","C2"])
    colors = color_map[labels]

    plt.figure(figsize=(6,6))
    plt.scatter(
        pc2d[:, 0], pc2d[:, 1],
        c=colors, s=2, alpha=0.4
    )
    plt.title("PC1 vs PC2 of SNLI embeddings (GPU PCA)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    # Legend entries
    handles = [
        plt.Line2D([0],[0], marker="o", color="w", markerfacecolor="C0", label="Entailment"),
        plt.Line2D([0],[0], marker="o", color="w", markerfacecolor="C1", label="Contradiction"),
        plt.Line2D([0],[0], marker="o", color="w", markerfacecolor="C2", label="Neutral"),
    ]
    plt.legend(handles=handles, loc="upper right")
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(str(out_png), dpi=120)
    plt.close()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    mlflow.set_experiment(args.experiment_name)
    
    # Auto-generate run name with key parameters
    run_name = f"pca2d_{args.dataset}_{args.batch_size}"
    
    with mlflow.start_run(run_name=run_name) as run:
        # Log parameters
        for k, v in vars(args).items():
            mlflow.log_param(k, v)
        # Log layer_num as parameter (from args)
        mlflow.log_param("layer_num", args.layer_num)
            
        # Set dataset and experiment as tags for better visibility in MLflow UI
        mlflow.set_tag("dataset", args.dataset)
        mlflow.set_tag("experiment_name", args.experiment_name)
        mlflow.set_tag("model_type", "pca2d")
        mlflow.set_tag("reduction_type", "pca2d")

        # Prepare output directory
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Load Parquet as float32 (or float16 if requested)
        df = pd.read_parquet(args.inp)
        X = np.vstack(df["vector"].to_numpy())
        if args.precision == "fp16":
            X = X.astype("float16")
            dtype = np.float16
        else:
            X = X.astype("float32")
            dtype = np.float32
        y = df["label"].to_numpy()

        n_samples, n_features = X.shape
        mlflow.log_param("n_samples", n_samples)
        mlflow.log_param("n_features", n_features)

        # Estimate memory and adjust batch_size if needed
        mem_req = estimate_memory_requirements(n_samples, n_features, 2, dtype)
        for k, v in mem_req.items():
            mlflow.log_metric(k, v)
        optimal_batch = get_optimal_batch_size(n_features, 2)
        if args.batch_size != optimal_batch:
            mlflow.log_param("adjusted_batch_size", optimal_batch)
            args.batch_size = optimal_batch

        # Fit GPU PCA to 2 components on a random sample
        sample_size = min(10000, n_samples)
        sample_idx = np.random.choice(n_samples, sample_size, replace=False)
        X_sample = X[sample_idx]
        X_sample_gpu = cp.asarray(X_sample)
        pca = GPU_PCA(n_components=2, random_state=42)
        pca.fit(X_sample_gpu)
        del X_sample_gpu
        cp.get_default_memory_pool().free_all_blocks()

        # Transform all data in chunks
        start_time = time.perf_counter()
        X2d = process_in_chunks(X, args.batch_size, pca)
        elapsed = time.perf_counter() - start_time
        mlflow.log_metric("pca2d_seconds", float(elapsed))

        # Log explained variance for the two components
        evr_cum = cp.asnumpy(cp.cumsum(pca.explained_variance_ratio_))
        mlflow.log_metric("pc1_variance_pct", float(evr_cum[0] * 100))
        mlflow.log_metric("pc2_variance_pct", float((evr_cum[1] - evr_cum[0]) * 100))
        mlflow.log_metric("pc1_pc2_cum_variance_pct", float(evr_cum[1] * 100))

        # Save 2D coordinates + labels as Parquet
        df_out = pd.DataFrame({
            "pc1": X2d[:, 0],
            "pc2": X2d[:, 1],
            "label": y
        })
        out_parquet = out_dir / "snli_pca2d.parquet"
        df_out.to_parquet(out_parquet)
        mlflow.log_artifact(str(out_parquet), artifact_path="pca2d_parquet", copy=False)

        # Plot and save the PC1 vs PC2 scatter
        out_png = out_dir / "snli_pca2d_scatter.png"
        plot_pc2d(X2d, y, out_png)
        mlflow.log_artifact(str(out_png), artifact_path="pca2d_plots", copy=False)

        print(f"✅ PCA 2D saved to {out_parquet} (time: {elapsed:.1f}s, PC1+PC2 var {evr_cum[1]*100:.2f}%)")

if __name__ == "__main__":
    main()
