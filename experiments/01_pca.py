#!/usr/bin/env python
"""
experiments/01_pca.py – PCA GPU-only (cuML) con float32
====================================================
Uso:
-----
python experiments/01_pca.py \
       --in  data/snli_train_embeddings.parquet \
       --out data/snli_train_pca50.parquet \
       --n_components 50

* Ejecuta PCA exclusivamente en GPU usando **cuML**, cargando los datos
  como **float32** para reducir uso de memoria.
* Registra un run en MLflow (experimento `pca-only-gpu`).
* Guarda curva de varianza `plots/evr_curve.png`.
* Artefacto principal: Parquet con vectores reducidos (`pca/`).
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
# Use Agg backend to avoid Qt plugin errors
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mlflow

# ---------------------------------------------------------------------------
# GPU-only PCA
# ---------------------------------------------------------------------------
try:
    import cupy as cp
    from cuml.decomposition import PCA as GPU_PCA
except ImportError:
    raise ImportError(
        "cuML GPU PCA no disponible: instala cuml y verifica tu entorno CUDA/GPU"
    )

# Force MLflow to use local mlruns directory (WSL-safe)
tracking_dir = Path.cwd().joinpath("mlruns")
mlflow.set_tracking_uri(tracking_dir.as_uri())

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="PCA GPU-only con cuML (float32)")
    parser.add_argument("--in", dest="inp", required=True,
                        help="Parquet de entrada con columnas 'vector','label'")
    parser.add_argument("--out", required=True, help="Parquet de salida PCA")
    parser.add_argument("--n_components", type=int, default=50,
                        help="Número de componentes PCA"),
    parser.add_argument("--experiment_name", default="pca-only-gpu-deflated-SNLI")
    return parser.parse_args()

# ---------------------------------------------------------------------------
# Plot EVR curve
# ---------------------------------------------------------------------------
def plot_evr(evr_cum, out_png):
    plt.figure(figsize=(6, 4))
    plt.plot(np.arange(1, len(evr_cum) + 1), evr_cum * 100, marker="o")
    plt.axhline(90, ls="--", color="gray")
    plt.xlabel("Componentes")
    plt.ylabel("Varianza acumulada (%)")
    plt.title("Curva de Varianza PCA (GPU, float32)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=120)
    plt.close()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    mlflow.set_experiment(args.experiment_name)
    with mlflow.start_run(run_name=Path(args.inp).stem) as run:
        mlflow.log_param("n_components", args.n_components)

        # Load data as float32 to save memory
        df = pd.read_parquet(args.inp)
        X_np = np.vstack(df["vector"].values).astype("float32")
        y = df["label"].values

        # Transfer to GPU as float32
        X_gpu = cp.asarray(X_np, dtype=cp.float32)

        # Load data as float32 to save memory
        df = pd.read_parquet(args.inp)
        X_np = np.vstack(df["vector"].values).astype("float32")
        y = df["label"].values

        # Transfer to GPU in chunks to avoid pinned-memory OOM
        from cupy.cuda import PinnedMemoryPool
        pinned_pool = PinnedMemoryPool()
        # set pinned memory allocator to the pool
        cp.cuda.set_pinned_memory_allocator(pinned_pool.malloc)
        # chunk by rows (~10k per chunk)
        n_rows = X_np.shape[0]
        chunk_size = 10000
        parts = []
        for i in range(0, n_rows, chunk_size):
            chunk = X_np[i:i+chunk_size]
            parts.append(cp.asarray(chunk))
            pinned_pool.free_all_blocks()
        # concatenate all chunks on GPU
        X_gpu = cp.concatenate(parts, axis=0)
        # reset pinned memory allocator to default
        cp.cuda.set_pinned_memory_allocator(cp.cuda.PinnedMemoryPool().malloc)

        # GPU PCA
        pca = GPU_PCA(n_components=args.n_components, random_state=42)
        start = time.perf_counter()
        X_red_gpu = pca.fit_transform(X_gpu)
        duration = time.perf_counter() - start
        mlflow.log_metric("pca_seconds", float(duration))

        # Back to numpy
        X_red = cp.asnumpy(X_red_gpu)
        evr_cum = cp.asnumpy(cp.cumsum(pca.explained_variance_ratio_))
        mlflow.log_metric("explained_variance_pct", float(evr_cum[-1] * 100))

        # Save reduced Parquet
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"vector": X_red.tolist(), "label": y}).to_parquet(out_path)
        try:
            mlflow.log_artifact(str(out_path), artifact_path="pca")
        except Exception:
            print("⚠️ No se pudo loguear el artefacto Parquet en MLflow")

        # Save ERV curve
        evr_png = out_path.with_name(f"{out_path.stem}_evr_curve.png")
        plot_evr(evr_cum, evr_png)
        try:
            mlflow.log_artifact(str(evr_png), artifact_path="plots")
        except Exception:
            print("⚠️ No se pudo loguear la curva EVR en MLflow")

        print(f"✅ PCA GPU guardado en {out_path} – varianza acumulada {evr_cum[-1]*100:.2f}% (tiempo {duration:.1f}s)")

if __name__ == "__main__":
    main()
