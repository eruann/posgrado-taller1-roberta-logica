#!/usr/bin/env python
"""
experiments/02_umap.py – UMAP GPU-only con cuML
=============================================
Uso:
-----
python experiments/02_umap.py \
    --pca_path data/snli/pca/pca_snli_50.parquet \
    --out_dir data/snli/umap \
    --n_neighbors 15 \
    --min_dist 0.1 \
    --metric euclidean \
    --dataset snli \
    --experiment_name "umap-snli" \
    --reduction_type pca

Parámetros:
----------
--pca_path: Parquet de entrada con columnas 'vector','label'
--out_dir: Directorio de salida para artefactos
--n_neighbors: Número de vecinos para UMAP (default: 15)
--min_dist: Distancia mínima entre puntos (default: 0.1)
--metric: Métrica de distancia (default: euclidean)
--dataset: Nombre del dataset (ej: snli, mnli, etc)
--experiment_name: Nombre del experimento en MLflow
--reduction_type: Tipo de reducción usada (pca o zca)
"""

import argparse
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import matplotlib
# Use Agg backend to avoid Qt plugin errors
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# GPU-only UMAP usando el acelerador de cuML
try:
    from cuml.accel import install
    install()
    # sklearn import is not strictly needed if only umap.UMAP is used directly
    # import sklearn 
    from umap import UMAP  # Imports from umap-learn, patched by cuml.accel
except ImportError as e:
    # Better error message and re-raise
    print("Error: Failed to initialize UMAP. This may be due to issues with 'cuml' or 'umap-learn'.")
    print(f"Specific import error: {e}")
    print("Please ensure RAPIDS cuML is installed correctly and 'umap-learn' is also installed.")
    raise

# Force MLflow to use local mlruns directory (WSL-safe)
tracking_dir = Path.cwd().joinpath("mlruns")
mlflow.set_tracking_uri(tracking_dir.as_uri())

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="UMAP GPU-only con cuML")
    parser.add_argument("--pca_path", required=True,
                        help="Parquet de entrada con columnas 'vector','label'")
    parser.add_argument("--out_dir", required=True,
                        help="Directorio de salida para artefactos")
    parser.add_argument("--n_neighbors", type=int, default=15,
                        help="Número de vecinos para UMAP")
    parser.add_argument("--min_dist", type=float, default=0.1,
                        help="Distancia mínima entre puntos")
    parser.add_argument("--metric", default="euclidean",
                        help="Métrica de distancia")
    parser.add_argument("--dataset", default="snli",
                        help="Nombre del dataset (ej: snli, mnli, etc)")
    parser.add_argument("--experiment_name", default="umap-snli")
    parser.add_argument("--reduction_type", choices=["pca", "zca"], required=True,
                        help="Tipo de reducción usada en los datos de entrada")
    parser.add_argument("--layer_num", type=int, default=12, help="Layer number to use (default: 12)")
    parser.add_argument("--input_n_components", type=str, required=True,
                        help="Número de componentes de la entrada PCA/ZCA (pasado directamente)")
    return parser.parse_args()

# ---------------------------------------------------------------------------
# Plot UMAP scatter
# ---------------------------------------------------------------------------
def plot_umap(X_umap, y, out_png, reduction_type, title="UMAP Projection"):
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap='viridis', alpha=0.6)
    plt.title(f"{title} ({reduction_type.upper()})")
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.colorbar(scatter, label='Label')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=120)
    plt.close()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    mlflow.set_experiment(args.experiment_name)

    # Use the directly passed n_components
    n_components_from_arg = args.input_n_components

    # Generar nombre del run con parámetros clave
    run_name = f"umap_{args.dataset}_{args.reduction_type}_{n_components_from_arg}_layer{args.layer_num}_n{args.n_neighbors}_{args.metric}"
    
    with mlflow.start_run(run_name=run_name) as run:
        # Log parameters
        for k, v in vars(args).items():
            mlflow.log_param(k, v)
        mlflow.set_tag("dataset", args.dataset)
        mlflow.set_tag("experiment_name", args.experiment_name)
        mlflow.set_tag("reduction_type", args.reduction_type)

        # Cargar datos
        df = pd.read_parquet(args.pca_path)
        X = np.vstack(df["vector"].values)
        y = df["label"].values
        n_samples, n_features = X.shape
        mlflow.log_param("input_dims", n_features)

        # metric_params = None # No longer needed as we remove the conditional logic
        # if args.metric == "mahalanobis":
        #     print("Configuring UMAP for 'mahalanobis'. This will likely use a CPU fallback via umap-learn.")
        #     # ... entire block for mahalanobis metric_kwds calculation removed ...
        # mlflow.log_param("metric_kwds_used", True if metric_params else False) # No longer needed

        # Inicializar UMAP (usando umap-learn, potencialmente patched by cuML)
        print(f"Initializing UMAP with: n_neighbors={args.n_neighbors}, min_dist={args.min_dist}, metric='{args.metric}', random_state=42, n_jobs=1")
        # if metric_params: # No longer needed
        #     print(f"Using metric_kwds for '{args.metric}'.")

        umap_instance = UMAP(
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
            metric=args.metric,
            # metric_kwds=metric_params,  # Removed: rely on umap-learn defaults or cuml.accel patching
            random_state=42,            # Keep existing random_state
            n_jobs=1                    # Explicitly set n_jobs=1 for reproducibility with random_state
        )

        # Fit y transform
        start = time.perf_counter()
        X_umap = umap_instance.fit_transform(X)
        duration = time.perf_counter() - start
        mlflow.log_metric("umap_seconds", float(duration))

        # Guardar resultados
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Construir nombre del archivo de salida manteniendo el formato original
        out_parquet = out_dir / f"umap_{args.dataset}_{n_components_from_arg}_layer{args.layer_num}_n{args.n_neighbors}_{args.metric}_{args.reduction_type}.parquet"
        out_png = out_dir / f"umap_{args.dataset}_{n_components_from_arg}_layer{args.layer_num}_n{args.n_neighbors}_{args.metric}_{args.reduction_type}_scatter.png"

        # Guardar coordenadas UMAP
        pd.DataFrame({
            "vector": list(X_umap),
            "label": y
        }).to_parquet(out_parquet)
        mlflow.log_artifact(str(out_parquet), artifact_path="umap_parquet")

        # Plot y guardar scatter
        title = f"UMAP Projection: {args.dataset.upper()} ({args.reduction_type.upper()}) - Layer {args.layer_num}"
        plot_umap(X_umap, y, out_png, args.reduction_type, title=title)
        mlflow.log_artifact(str(out_png), artifact_path="umap_plots")

        print(f"✅ UMAP GPU guardado en {out_parquet} (tiempo {duration:.1f}s)")
        print("   Run ID:", run.info.run_id)

if __name__ == "__main__":
    main()
