#!/usr/bin/env python
"""
experiments/04_deflate.py – Remove top-K PCs in batches (GPU-only, chunked)
===========================================================================
Neutraliza las primeras K componentes dominantes (deflate) sin cargar todo en memoria.
Usa cuML en GPU y PyArrow para procesar el Parquet por lotes.

Uso:
-----
python experiments/04_deflate.py \
    --inp      data/snli_train_embeddings.parquet \
    --out      data/snli_train_deflated3.parquet \
    --n_remove 3 \
    [--batch_size 10000]

* Procesa en batches de filas para evitar OOM de CPU/GPU.
* Para cada batch:
  1. Carga subset con PyArrow.dataset
  2. Transforma a array NumPy → CuPy
  3. Ajusta GPU PCA (top-K) solo en la primera llamada
  4. Deflacion: resta la reconstrucción del batch
  5. Vuelca el batch deflacionado a Parquet usando ParquetWriter
* Registra run en MLflow (`deflate-top-pcs`) con params:
  - n_remove, input_dims, batch_size
  - deflate_seconds (total)
"""
import argparse
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pyarrow as pa

# GPU-only PCA
try:
    import cupy as cp
    from cuml.decomposition import PCA as GPU_PCA
except ImportError:
    raise ImportError("cuML GPU PCA no disponible: instala cuml y verifica tu entorno CUDA/GPU")

# Force MLflow to use local mlruns directory
tracking_dir = Path.cwd().joinpath("mlruns")
mlflow.set_tracking_uri(tracking_dir.as_uri())

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Deflate top-K PCs (chunked GPU)")
    p.add_argument("--inp", required=True, help="Parquet embeddings con 'vector','label'")
    p.add_argument("--out", required=True, help="Parquet de salida deflated")
    p.add_argument("--n_remove", type=int, default=3, help="Número de PCs a eliminar")
    p.add_argument("--batch_size", type=int, default=10000,
                   help="Número de filas por lote")
    p.add_argument("--layer_num", type=int, default=12, help="Layer number to use (default: 12)")
    return p.parse_args()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    mlflow.set_experiment("deflate-top-pcs")
    start_total = time.perf_counter()

    dataset = ds.dataset(args.inp, format='parquet')
    total_rows = sum(fragment.count_rows() for fragment in dataset.get_fragments())
    first_batch = True

    writer = None
    comps = None
    input_dims = None

    with mlflow.start_run(run_name=Path(args.out).stem) as run:
        mlflow.log_params({
            "n_remove": args.n_remove,
            "batch_size": args.batch_size
        })
        mlflow.log_param("layer_num", args.layer_num)

        # Iterate over record batches
        for batch in dataset.to_batches(batch_size=args.batch_size):
            table = pa.Table.from_batches([batch])
            vecs = table.column('vector').to_pylist()
            labels = table.column('label').to_pylist()

            X_np = np.array(vecs, dtype=np.float32)
            if first_batch:
                # Fit PCA on first batch only
                X_gpu_full = cp.asarray(X_np)
                pca = GPU_PCA(n_components=args.n_remove, random_state=42)
                pca.fit(X_gpu_full)
                comps = pca.components_  # (n_remove, D)
                input_dims = X_gpu_full.shape[1]
                mlflow.log_param("input_dims", int(input_dims))
                first_batch = False
                del X_gpu_full

            # Deflate current batch
            X_gpu = cp.asarray(X_np)
            scores = X_gpu.dot(comps.T)           # (batch, n_remove)
            recon = scores.dot(comps)             # (batch, D)
            X_def_gpu = X_gpu - recon             # deflated
            X_def_np = cp.asnumpy(X_def_gpu)

            # Build Arrow Table for output
            out_table = pa.Table.from_pydict({
                'vector': X_def_np.tolist(),
                'label': labels
            })

            # Initialize ParquetWriter on first batch
            if writer is None:
                writer = pq.ParquetWriter(args.out, out_table.schema)

            writer.write_table(out_table)
            # free GPU memory
            del X_gpu, X_def_gpu, scores, recon

        # Close writer
        writer.close()

        duration = time.perf_counter() - start_total
        mlflow.log_metric("deflate_seconds", float(duration))
        print(f"✅ Deflated dataset ({total_rows} rows, {input_dims} dims) in {duration:.1f}s -> {args.out}")
        print("   Run ID:", run.info.run_id)

if __name__ == '__main__':
    main()
