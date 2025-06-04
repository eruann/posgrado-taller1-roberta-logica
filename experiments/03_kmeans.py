#!/usr/bin/env python
"""
experiments/03_kmeans.py – K-Means GPU-only with local MLflow URI
================================================================
Agrupa los vectores reducidos (PCA o UMAP) con **K-Means** usando exclusivamente cuML
y fuerza a MLflow a usar un directorio local `mlruns/` para evitar rutas Windows.

Diferencias entre PCA y UMAP:
- PCA: Reducción lineal que preserva varianza global, mejor para clusters globales
- UMAP: Reducción no-lineal que preserva estructura local/global, mejor para patrones complejos

Uso:
-----
python experiments/03_kmeans.py \
    --input_path data/snli/pca/pca_snli_50.parquet \
    --out_dir data/kmeans_outputs \
    --k 3 \
    --max_iter 300 \
    --random_state 42 \
    --dataset snli \
    --reduction_type pca \
    --experiment_name kmeans-snli

Esto generará archivos como:
  data/kmeans_outputs/kmeans_snli_pca_50_3_20240321_123456.csv
  data/kmeans_outputs/kmeans_snli_pca_50_3_20240321_123456_clusters.png
"""
import argparse
import time
from pathlib import Path
import re

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, normalized_mutual_info_score as nmi_score
import matplotlib
# Use Agg backend to avoid Qt plugin errors
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def estimate_kmeans_memory(n_samples, n_features, k, dtype=np.float32):
    input_size = n_samples * n_features * dtype().itemsize
    labels_size = n_samples * np.dtype(np.int32).itemsize
    centers_size = k * n_features * dtype().itemsize
    total = input_size + labels_size + centers_size
    return {
        "input_size_gb": input_size / (1024**3),
        "labels_size_gb": labels_size / (1024**3),
        "centers_size_gb": centers_size / (1024**3),
        "total_gb": total / (1024**3)
    }

# GPU-only K-Means
try:
    from cuml.cluster import KMeans as GPU_KMeans
    import cupy as cp
except ImportError:
    raise ImportError(
        "cuML GPU KMeans no disponible: instala cuml y verifica tu entorno CUDA/GPU"
    )

# Force MLflow to use local mlruns directory (WSL-safe)
tracking_dir = Path.cwd().joinpath("mlruns")
mlflow.set_tracking_uri(tracking_dir.as_uri())

BACKEND = "gpu"

# ---------------------------------------------------------------------------
# Purity helper
# ---------------------------------------------------------------------------
def purity_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    return np.sum(np.amax(cm, axis=0)) / np.sum(cm)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="K-Means GPU-only pipeline")
    p.add_argument("--input_path", required=True, help="Parquet con vectors y label (PCA o UMAP)")
    p.add_argument("--out_dir", required=True, help="Directorio de salida para artefactos")
    p.add_argument("--k", type=int, default=3, help="Número de clusters (default 3)")
    p.add_argument("--max_iter", type=int, default=300)
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--dataset", default="snli", help="Nombre del dataset")
    p.add_argument("--experiment_name", default="kmeans-roberta-base")
    p.add_argument("--reduction_type", choices=["pca", "umap"], required=True,
                  help="Tipo de reducción usada en los datos de entrada")
    p.add_argument("--layer_num", type=int, default=12, help="Layer number to use (default: 12)")
    p.add_argument("--input_n_components", type=str, required=False,
                        help="Número de componentes de la entrada original a UMAP (PCA/ZCA dimensions)")
    p.add_argument("--umap_n_neighbors", type=str, required=False,
                        help="Número de vecinos usados por UMAP")
    p.add_argument("--umap_metric", type=str, required=False,
                        help="Métrica usada por UMAP")
    p.add_argument("--umap_source_reduction_type", type=str, choices=["pca", "zca"], required=False,
                        help="Tipo de reducción original que alimentó a UMAP (pca/zca)")
    return p.parse_args()

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def plot_clusters(X, preds, y_true, out_png, reduction_type, title="KMeans Clustering"):
    """Plot clusters using first two dimensions of reduced data."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: True labels
    scatter1 = ax1.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', alpha=0.6)
    ax1.set_title(f"Etiquetas Originales")
    ax1.set_xlabel(f'First {reduction_type.upper()} Component')
    ax1.set_ylabel(f'Second {reduction_type.upper()} Component')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter1, ax=ax1, label='True Label')
    
    # Add text labels for true labels (only show unique labels)
    unique_labels = np.unique(y_true)
    for label in unique_labels:
        # Find the center of each label's points
        mask = y_true == label
        center_x = np.mean(X[mask, 0])
        center_y = np.mean(X[mask, 1])
        ax1.text(center_x, center_y, f'Label {label}', 
                ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.3'))
    
    # Plot 2: Cluster assignments
    scatter2 = ax2.scatter(X[:, 0], X[:, 1], c=preds, cmap='viridis', alpha=0.6)
    ax2.set_title(f"Clusters KMeans (k={len(np.unique(preds))})")
    ax2.set_xlabel(f'First {reduction_type.upper()} Component')
    ax2.set_ylabel(f'Second {reduction_type.upper()} Component')
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=ax2, label='Cluster')
    
    # Add text labels for clusters
    unique_clusters = np.unique(preds)
    for cluster in unique_clusters:
        # Find the center of each cluster
        mask = preds == cluster
        center_x = np.mean(X[mask, 0])
        center_y = np.mean(X[mask, 1])
        ax2.text(center_x, center_y, f'Cluster {cluster}', 
                ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.3'))
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=120)
    plt.close()

def main():
    args = parse_args()
    mlflow.set_experiment(args.experiment_name)

    # --- Validate conditional arguments ---
    if args.reduction_type == "umap":
        if not all([args.input_n_components, args.umap_n_neighbors, args.umap_metric, args.umap_source_reduction_type]):
            raise ValueError(
                "Si reduction_type es 'umap', se deben proporcionar: "
                "--input_n_components, --umap_n_neighbors, --umap_metric, y --umap_source_reduction_type"
            )
        # Assign to local variables for clarity, matching old parsed variable names
        n_components_str = args.input_n_components
        n_neighbors_str = args.umap_n_neighbors
        metric_str = args.umap_metric
        original_reduction_type_str = args.umap_source_reduction_type
    elif args.reduction_type == "pca":
        # For direct PCA input, these UMAP-specific params are not used for naming in the same way
        # n_components will be derived from input_dims later if not explicitly passed for PCA
        # For now, ensure they are not expected to be None if accessed generally
        n_components_str = args.input_n_components if args.input_n_components else "dims_from_input"
        n_neighbors_str = "N/A"
        metric_str = "N/A"
        original_reduction_type_str = "pca" # By definition if args.reduction_type is "pca"
    else:
        # Should not happen due to choices in argparse
        raise ValueError(f"Unsupported reduction_type: {args.reduction_type}")

    # Auto-generate MLflow run name using direct args
    mlflow_run_name_parts = ["kmeans", args.dataset]
    if args.reduction_type == "umap":
        mlflow_run_name_parts.extend([
            original_reduction_type_str, 
            n_components_str, 
            f"layer{args.layer_num}", 
            f"n{n_neighbors_str}", 
            metric_str, 
            f"k{args.k}"
        ])
    elif args.reduction_type == "pca":
        # If KMeans input is directly PCA, the run name structure is simpler
        # n_components_str here would ideally be from input_dims or a dedicated arg if direct PCA->KMeans was primary path
        mlflow_run_name_parts.extend([original_reduction_type_str, n_components_str, f"layer{args.layer_num}", f"k{args.k}"])
    run_name = "_".join(mlflow_run_name_parts)

    with mlflow.start_run(run_name=run_name) as run:
        # Log params and tags
        mlflow.log_params({
            "k": args.k, 
            "backend": BACKEND, 
            "max_iter": args.max_iter, 
            "random_state": args.random_state,
            "reduction_type": args.reduction_type
        })
        mlflow.set_tag("dataset", args.dataset)
        mlflow.set_tag("experiment_name", args.experiment_name)
        mlflow.set_tag("reduction_type", args.reduction_type)
        mlflow.log_param("layer_num", args.layer_num)

        # Load data
        df = pd.read_parquet(args.input_path)
        X = np.vstack(df["vector"].values)
        y = df["label"].values
        n_samples, n_features = X.shape
        input_dims = n_features
        mlflow.log_param("input_dims", input_dims)

        # If reduction_type is PCA and input_n_components was not given, use input_dims
        if args.reduction_type == "pca" and args.input_n_components is None:
            n_components_str = str(input_dims)
        
        # (Re-evaluate run_name if it used a placeholder and needs update, e.g. n_components_str from input_dims for PCA)
        # This logic might need refinement if the initial run_name relied on a placeholder that's now concrete.
        # For simplicity, we assume the run_name set above is now definitive.
        if mlflow.active_run().data.tags.get("mlflow.runName") != run_name:
             mlflow.active_run().data.tags["mlflow.runName"] = run_name

        # Estimate and log memory requirements
        mem_req = estimate_kmeans_memory(n_samples, n_features, args.k)
        print("Estimación de memoria para KMeans:")
        for k, v in mem_req.items():
            print(f"  {k}: {v:.2f} GB")
            mlflow.log_metric(k, v)

        # --- K-Means en GPU ---
        X_gpu = cp.asarray(X)
        kmeans = GPU_KMeans(
            n_clusters=args.k,
            max_iter=args.max_iter,
            random_state=args.random_state,
            init="k-means++",
        )
        t0 = time.perf_counter()
        kmeans.fit(X_gpu)
        duration = time.perf_counter() - t0
        preds = cp.asnumpy(kmeans.labels_)
        inertia = float(kmeans.inertia_)

        # Log metrics
        mlflow.log_metric("kmeans_seconds", duration)
        mlflow.log_metric("kmeans_minutes", duration / 60)
        mlflow.log_metric("inertia", inertia)
        purity = purity_score(y, preds)
        mlflow.log_metric("purity", purity)
        nmi = nmi_score(y, preds)
        mlflow.log_metric("nmi", nmi)

        # Save results
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Construct the base name for output files to align with run_pipeline.py expectation
        # Expected by pipeline for UMAP inputs: kmeans_snli_{n_components}_layer{layer}_n{n_neighbors}_{metric}_{original_reduction_type}_k{k}.csv

        if args.reduction_type == "umap":
            output_basename = f"kmeans_{args.dataset}_{n_components_str}_layer{args.layer_num}_n{n_neighbors_str}_{metric_str}_{original_reduction_type_str}_k{args.k}"
        elif args.reduction_type == "pca": 
            # This case is if Kmeans is run directly on PCA output. 
            # n_components_str should be set (either from new arg or input_dims)
            output_basename = f"kmeans_{args.dataset}_{original_reduction_type_str}_{n_components_str}_layer{args.layer_num}_k{args.k}"
        else: # Should not be reached given arg choices
            output_basename = f"kmeans_{args.dataset}_unknown_{Path(args.input_path).stem}_fallback_k{args.k}"
            print(f"Warning: Using fallback output basename due to unexpected reduction_type: {output_basename}")

        out_csv = out_dir / f"{output_basename}.csv"
        out_png = out_dir / f"{output_basename}_clusters.png"

        # Save cluster assignments
        results_df = pd.DataFrame({
            "cluster": preds,
            "label": y
        })
        results_df.to_csv(out_csv, index=False)
        mlflow.log_artifact(str(out_csv), artifact_path="kmeans")

        # Plot clusters
        title_plot_kmeans = f"Agrupamiento KMeans: {args.dataset} ({args.reduction_type.upper()}) - Layer {args.layer_num}, k={args.k}"
        plot_clusters(X, preds, y, out_png, args.reduction_type, title=title_plot_kmeans)
        mlflow.log_artifact(str(out_png), artifact_path="plots")

        print(f"✅ KMeans GPU k={args.k} → purity {purity:.3f}, NMI {nmi:.3f}, tiempo {duration/60:.2f} min")
        print("   Run ID:", run.info.run_id)

if __name__ == "__main__":
    main()
