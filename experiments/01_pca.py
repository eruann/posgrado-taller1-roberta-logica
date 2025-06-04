#!/usr/bin/env python
"""
experiments/01_pca.py – PCA and ZCA GPU-only (cuML) con float32
====================================================
Uso:
-----
python experiments/01_pca.py \
       --source_path data/snli_train_embeddings.parquet \
       --out data/snli_train_pca50.parquet \
       --n_components 50 \
       --experiment_name "pca-only-gpu-deflated-SNLI" \
       --precision fp32 \
       --batch_size 10000 \
       --dataset snli

Parámetros:
----------
--source_path: Parquet de entrada con columnas 'vector','label'
--out: Parquet de salida PCA/ZCA
--n_components: Número de componentes PCA (default: 50)
--experiment_name: Nombre del experimento en MLflow (default: "pca-only-gpu-deflated-SNLI")
--precision: Precisión de punto flotante (choices: fp32, fp16, default: fp32)
--batch_size: Tamaño del batch para procesamiento en chunks (default: 10000)
--dataset: Nombre del dataset (ej: snli, mnli, etc, default: snli)

Características:
--------------
* Ejecuta PCA exclusivamente en GPU usando **cuML**, cargando los datos
  como **float32** para reducir uso de memoria.
* Registra un run en MLflow con todos los parámetros y métricas.
* Guarda curva de varianza `plots/evr_curve.png` y tabla de varianza.
* Artefacto principal: Parquet con vectores reducidos.
* Optimización automática del tamaño de batch según memoria disponible.
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
from tqdm import tqdm 
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
    parser = argparse.ArgumentParser(description="PCA/ZCA GPU-only con cuML (float32)")
    parser.add_argument("--source_path", dest="inp", required=True,
                        help="Parquet de entrada con columnas 'vector','label'")
    parser.add_argument("--out", required=True, help="Parquet de salida PCA/ZCA")
    parser.add_argument("--n_components", type=int, default=50,
                        help="Número de componentes PCA")
    parser.add_argument("--experiment_name", default="pca-only-gpu-deflated-SNLI")
    parser.add_argument("--precision", choices=["fp32", "fp16"], default="fp32",
                        help="Precisión de punto flotante")
    parser.add_argument("--batch_size", type=int, default=10000,
                        help="Tamaño del batch para procesamiento en chunks")
    parser.add_argument("--dataset", default="snli", 
                        help="Nombre del dataset (ej: snli, mnli, etc)")
    parser.add_argument("--layer_num", type=int, default=12, help="Layer number to use (default: 12)")
    return parser.parse_args()

# ---------------------------------------------------------------------------
# Plot EVR curve
# ---------------------------------------------------------------------------
def plot_evr(evr_cum, out_png, reduction_type="PCA"):
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(1, len(evr_cum) + 1), evr_cum * 100, marker="o")
    plt.axhline(90, ls="--", color="gray", label="90% varianza")
    plt.axhline(80, ls="--", color="gray", alpha=0.5, label="80% varianza")
    plt.axhline(70, ls="--", color="gray", alpha=0.3, label="70% varianza")
    
    # Add annotations for key points
    key_points = [0.5, 0.6, 0.7, 0.8, 0.9]  # 50%, 60%, 70%, 80%, 90%
    for threshold in key_points:
        idxs = np.where(evr_cum >= threshold)[0]
        if len(idxs) > 0:  # Only annotate if threshold is reached
            idx = idxs[0]
            plt.plot(idx + 1, evr_cum[idx] * 100, 'ro')
            plt.annotate(f'{int(evr_cum[idx]*100)}% ({idx+1} dim)',
                        xy=(idx + 1, evr_cum[idx] * 100),
                        xytext=(10, 10), textcoords='offset points')
    
    plt.xlabel("Número de Componentes")
    plt.ylabel("Varianza Acumulada (%)")
    plt.title(f"Varianza Acumulada {reduction_type} (GPU, float32)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=120)
    plt.close()

    # Create and save variance table as CSV with consistent naming
    variance_data = []
    for i, var in enumerate(evr_cum):
        if i < 10 or var >= 0.5 or i % 10 == 0:  # Show first 10, every 10th, and key points
            variance_data.append({
                'componentes': i + 1,
                'varianza_acumulada': var * 100
            })
        if var >= 0.95:  # Stop at 95%
            break
    
    # Save as CSV with matching name
    variance_csv = out_png.with_name(out_png.stem.replace('_evr_curve', '_variance_table') + '.csv')
    variance_df = pd.DataFrame(variance_data)
    variance_df.to_csv(variance_csv, index=False)
    
    # Log to MLflow
    try:
        mlflow.log_artifact(str(variance_csv), artifact_path="variance_tables")
    except Exception:
        print("⚠️ No se pudo loguear la tabla de varianza en MLflow")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def estimate_memory_requirements(n_samples, n_features, n_components, dtype=np.float32):
    # Input data size
    input_size = n_samples * n_features * dtype().itemsize
    # Output data size (reduced dimensions)
    output_size = n_samples * n_components * dtype().itemsize
    # PCA components size
    components_size = n_features * n_components * dtype().itemsize
    # Total GPU memory needed (rough estimate)
    total_gpu = input_size + output_size + components_size
    return {
        'input_size_gb': input_size / (1024**3),
        'output_size_gb': output_size / (1024**3),
        'components_size_gb': components_size / (1024**3),
        'total_gpu_gb': total_gpu / (1024**3)
    }

def get_optimal_batch_size(n_features, n_components, available_memory_gb=56):
    # Estimate memory per sample
    memory_per_sample = (n_features + n_components) * 4  # 4 bytes for float32
    # Leave 20% buffer for other operations
    safe_memory = available_memory_gb * 0.8 * (1024**3)
    # Calculate optimal batch size
    optimal_batch_size = int(safe_memory / memory_per_sample)
    return min(optimal_batch_size, 10000)  # Cap at 10000

def process_in_chunks(X_np, batch_size, pca, is_zca=False):
    n_rows = X_np.shape[0]
    n_chunks = (n_rows + batch_size - 1) // batch_size
    results = []
    
    for i in tqdm(range(0, n_rows, batch_size), 
                 desc="Processing chunks",
                 total=n_chunks):
        chunk = X_np[i:i+batch_size]
        # Process chunk
        chunk_gpu = cp.asarray(chunk)
        chunk_result = pca.transform(chunk_gpu)
        results.append(cp.asnumpy(chunk_result))
        # Clear GPU memory
        del chunk_gpu
        cp.get_default_memory_pool().free_all_blocks()
        
    return np.vstack(results)

def main():
    args = parse_args()
    mlflow.set_experiment(args.experiment_name)
    
    # Load data as float32 to save memory
    df = pd.read_parquet(args.inp)
    X_np = np.vstack(df["vector"].values).astype("float32")
    y = df["label"].values

    n_samples, n_features = X_np.shape
    
    # Store original batch_size if it might be adjusted
    original_batch_size = args.batch_size
    optimal_batch = get_optimal_batch_size(n_features, args.n_components)
    batch_size_adjusted = False
    if args.batch_size != optimal_batch:
        print(f"Adjusting batch size from {args.batch_size} to {optimal_batch} for better memory usage")
        args.batch_size = optimal_batch
        batch_size_adjusted = True

    # Process both PCA and ZCA
    for reduction_type in ["pca", "zca"]:
        print(f"\nProcessing {reduction_type.upper()}...")
        
        # Ensure args.batch_size reflects the potentially adjusted value for run name
        run_name = f"{reduction_type}_{args.dataset}_{args.n_components}_layer{args.layer_num}_batch{args.batch_size}"
        with mlflow.start_run(run_name=run_name) as run:
            # Log all CLI parameters for this specific run
            # We will log original_batch_size separately if it was adjusted
            temp_args_dict = vars(args).copy()
            if batch_size_adjusted:
                # Log the actual batch size used for computation under 'batch_size'
                # And the original one separately
                temp_args_dict['batch_size'] = args.batch_size # The (potentially) adjusted one
            
            for k_arg, v_arg in temp_args_dict.items():
                mlflow.log_param(k_arg, v_arg)
            
            if batch_size_adjusted:
                mlflow.log_param("initial_requested_batch_size", original_batch_size)
            
            mlflow.log_param("effective_batch_size", args.batch_size) # Log effective batch size for this run

            # Log layer_num explicitly as it's key
            mlflow.log_param("layer_num", args.layer_num)
            # Set tags for this run
            mlflow.set_tag("dataset", args.dataset)
            mlflow.set_tag("experiment_name", args.experiment_name)
            mlflow.set_tag("reduction_mode", reduction_type)

            # Log dataset statistics per run
            mlflow.log_param("n_samples", n_samples)
            mlflow.log_param("n_features", n_features)
            # n_components is already in temp_args_dict via vars(args)

            # Calculate and log memory requirements for this specific type of PCA run
            mem_req = estimate_memory_requirements(n_samples, n_features, args.n_components)
            print(f"Memory requirements for {reduction_type.upper()}:")
            for k_mem, v_mem in mem_req.items():
                print(f"  {k_mem}: {v_mem:.2f} GB")
                mlflow.log_metric(f"{reduction_type}_{k_mem}", v_mem)

            # GPU PCA - First fit on a sample to get components
            print("Fitting on sample...")
            sample_size = min(10000, n_samples)
            sample_indices = np.random.choice(n_samples, sample_size, replace=False)
            X_sample = X_np[sample_indices]
            X_sample_gpu = cp.asarray(X_sample)
            if reduction_type == "zca":
                print("Initializing PCA with whiten=True for ZCA.")
                pca = GPU_PCA(n_components=args.n_components, random_state=42, whiten=True)
            else:
                # whiten defaults to False, so explicitly setting it is optional but clear
                pca = GPU_PCA(n_components=args.n_components, random_state=42, whiten=False)
            pca.fit(X_sample_gpu)
            del X_sample_gpu
            cp.get_default_memory_pool().free_all_blocks()

            # Transform all data in chunks
            print("Transforming data in chunks...")
            start = time.perf_counter()
            X_red = process_in_chunks(X_np, args.batch_size, pca, is_zca=(reduction_type=="zca"))
            duration = time.perf_counter() - start
            mlflow.log_metric(f"{reduction_type}_seconds", float(duration))

            # Calculate explained variance
            evr_cum = cp.asnumpy(cp.cumsum(pca.explained_variance_ratio_))
            mlflow.log_metric(f"{reduction_type}_explained_variance_pct", float(evr_cum[-1] * 100))

            # Save reduced Parquet
            out_dir = Path(args.out).parent
            out_dir.mkdir(parents=True, exist_ok=True)
            
            # Modified output path construction
            base_out_path = Path(args.out) 
            # Assumes args.out is like "dataset_ncomps_layer.parquet"
            # Prepend reduction_type to the filename part.
            out_path_name = f"{reduction_type}_{base_out_path.name}"
            out_path = base_out_path.with_name(out_path_name)
            
            pd.DataFrame({
                "vector": list(X_red),
                "label": y
            }).to_parquet(out_path)
            # Don't log parquet to MLflow to avoid copies

            # Save EVR curve
            evr_png_name = f"evr_curve_{reduction_type}_{args.dataset}_{args.n_components}_layer{args.layer_num}.png"
            evr_png = out_dir / evr_png_name
            plot_evr(evr_cum, evr_png, reduction_type=reduction_type.upper())
            mlflow.log_artifact(str(evr_png), artifact_path="plots")

            # Save variance table as CSV
            variance_csv_name = f"variance_table_{reduction_type}_{args.dataset}_{args.n_components}_layer{args.layer_num}.csv"
            variance_csv = out_dir / variance_csv_name
            variance_df = pd.DataFrame({
                'componentes': np.arange(1, len(evr_cum) + 1),
                'varianza_acumulada': evr_cum * 100
            })
            variance_df.to_csv(variance_csv, index=False)
            mlflow.log_artifact(str(variance_csv), artifact_path="variance_tables")

            print(f"✅ {reduction_type.upper()} GPU guardado en {out_path} – varianza acumulada {evr_cum[-1]*100:.2f}% (tiempo {duration:.1f}s)")

if __name__ == "__main__":
    main()
