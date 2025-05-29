#!/usr/bin/env python
"""
experiments/03_kmeans.py – K-Means GPU-only with local MLflow URI
================================================================
Agrupa los vectores PCA con **K-Means (k = 3)** usando exclusivamente cuML
y fuerza a MLflow a usar un directorio local `mlruns/` para evitar rutas Windows.

Registra en **MLflow**:
• Parámetros: k, input_dims, backend=gpu
• Métricas  : purity, nmi, inertia, kmeans_seconds
• Artefactos: clusters.csv (label, pred)
"""
import argparse
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, normalized_mutual_info_score as nmi_score

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
    p.add_argument("--pca_path", required=True, help="Parquet con vectors y label")
    p.add_argument("--out_csv", required=True, help="CSV de predicciones")
    p.add_argument("--k", type=int, default=3, help="Número de clusters (default 3)")
    p.add_argument("--max_iter", type=int, default=300)
    p.add_argument("--random_state", type=int, default=42)
    return p.parse_args()

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    mlflow.set_experiment("kmeans-roberta-base")

    with mlflow.start_run(run_name=Path(args.pca_path).stem) as run:
        # Log params
        mlflow.log_params({"k": args.k, "backend": BACKEND})

        # Load data
        df = pd.read_parquet(args.pca_path)
        X = np.vstack(df["vector"].values)
        y = df["label"].values
        input_dims = X.shape[1]
        mlflow.log_param("input_dims", input_dims)

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
        mlflow.log_metric("inertia", inertia)
        purity = purity_score(y, preds)
        mlflow.log_metric("purity", purity)
        nmi = nmi_score(y, preds)
        mlflow.log_metric("nmi", nmi)

        # Save predictions CSV
        out_csv = Path(args.out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"label": y, "pred": preds}).to_csv(out_csv, index=False)
        try:
            mlflow.log_artifact(str(out_csv), artifact_path="clusters")
        except Exception:
            print("⚠️ No se pudo loguear el CSV de clusters en MLflow")

        print(f"✅ KMeans GPU k={args.k} → purity {purity:.3f}, NMI {nmi:.3f}")
        print("   Run ID:", run.info.run_id)

if __name__ == "__main__":
    main()
