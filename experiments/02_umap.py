#!/usr/bin/env python
"""
experiments/02_umap.py – UMAP 2-D (GPU-only) with local MLflow URI
=================================================================
*Proyecta la matriz PCA a 2-D usando UMAP en GPU exclusivamente* y asegura
que MLflow use un directorio local `mlruns/` en WSL para evitar rutas Windows.

Uso:
-----
python experiments/02_umap.py \
    --pca_path data/snli_deflated_pca60.parquet \
    --out_umap data/snli_deflated_umap60.npy \
    --out_plot data/snli_deflated_umap60.png \
    --n_neighbors 15 --min_dist 0.1
"""

import argparse
import time
from pathlib import Path
import os

import mlflow
import numpy as np
import pandas as pd
import cupy as cp
from matplotlib import pyplot as plt
from sklearn.metrics import silhouette_score

# GPU-only UMAP
from cuml.manifold import UMAP as GPU_UMAP

# ---------------------------------------------------------------------------
# Force MLflow to use local mlruns directory (WSL-safe)
# ---------------------------------------------------------------------------
tracking_dir = Path.cwd().joinpath("mlruns")
mlflow.set_tracking_uri(tracking_dir.as_uri())

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="UMAP 2-D GPU-only pipeline")
    parser.add_argument("--pca_path", required=True, help="Parquet con vectors y label")
    parser.add_argument("--out_umap", required=True, help="Ruta .npy destino (matriz 2-D)")
    parser.add_argument("--out_plot", required=True, help="Ruta .png del scatter")
    parser.add_argument("--n_neighbors", type=int, default=15)
    parser.add_argument("--min_dist", type=float, default=0.1)
    parser.add_argument("--metric", default="cosine")
    parser.add_argument("--random_state", type=int, default=42)
    return parser.parse_args()

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    mlflow.set_experiment("umap-roberta-base")
    with mlflow.start_run(run_name=Path(args.pca_path).stem) as run:
        # Log parameters
        mlflow.log_params({
            "n_neighbors": args.n_neighbors,
            "min_dist": args.min_dist,
            "metric": args.metric,
            "backend": "gpu",
        })

        # Load PCA data
        df = pd.read_parquet(args.pca_path)
        X = np.vstack(df["vector"].values)
        y = df["label"].values
        input_dims = X.shape[1]
        mlflow.log_param("input_dims", input_dims)

        # UMAP projection on GPU
        reducer = GPU_UMAP(
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
            metric=args.metric,
            random_state=args.random_state,
        )
        start = time.perf_counter()
        X_emb = reducer.fit_transform(cp.asarray(X))
        X_2d = cp.asnumpy(X_emb)
        duration = time.perf_counter() - start
        mlflow.log_metric("umap_seconds", duration)

        # Silhouette on 2-D (GPU)
        sil = silhouette_score(X_2d, y, metric="euclidean")
        mlflow.log_metric("silhouette_2d", float(sil))

        # Save UMAP embeddings
        out_umap = Path(args.out_umap)
        out_umap.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_umap, X_2d)
        try:
            mlflow.log_artifact(str(out_umap), artifact_path="umap")
        except Exception:
            print("⚠️ No se pudo loguear el archivo UMAP en MLflow")

        # Scatter plot
        plt.figure(figsize=(6, 4))
        plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap="viridis", s=3, alpha=0.6)
        plt.title(f"UMAP 2-D — SNLI ({input_dims} dims, gpu)")
        plt.xlabel("UMAP1")
        plt.ylabel("UMAP2")
        plt.tight_layout()
        out_plot = Path(args.out_plot)
        plt.savefig(out_plot, dpi=150)
        plt.close()
        try:
            mlflow.log_artifact(str(out_plot), artifact_path="plots")
        except Exception:
            print("⚠️ No se pudo loguear el scatter en MLflow")

        print(f"✅ UMAP (gpu, {input_dims} dims, {duration:.1f}s) → {out_umap}")
        print("   Run ID:", run.info.run_id)


if __name__ == "__main__":
    main()
