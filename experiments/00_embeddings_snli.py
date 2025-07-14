#!/usr/bin/env python
"""
Generador de embeddings para datasets de Hugging Face usando RoBERTa-base.
Extrae embeddings de todas las capas del modelo (12 capas + embeddings de entrada).

Ejemplos de uso:
    # Usando dataset SNLI desde Hugging Face
    python experiments/00_embeddings_snli.py \
        --out data/snli/embeddings

    # Usando dataset local
    python experiments/00_embeddings_snli.py \
        --source_path data/snli/dataset \
        --out data/snli/embeddings

    # Usando GPU con precisión FP16
    python experiments/00_embeddings_snli.py \
        --device cuda \
        --precision fp16 \
        --out data/snli/embeddings

    # uso general
    python experiments/00_embeddings_snli.py \
        --source_path data/snli/dataset \
        --out data/snli/embeddings \
        --dataset snli \
        --experiment_name embeddings-roberta-base \
        --device cuda \
        --layer_num 0-12

Parámetros:
    --device: Dispositivo a usar ('cuda' o 'cpu'). Si se usa 'cuda', se verificará
              la disponibilidad de GPU y se mostrará información sobre la misma.
    --precision: Precisión de punto flotante ('fp32' o 'fp16'). fp16 usa menos memoria
                 pero puede ser menos preciso.
    --batch_size: Tamaño del batch para procesamiento. Ajustar según memoria disponible.
    --save_every: Cada cuántos batches guardar resultados parciales.
    --out: Directorio donde se guardarán los embeddings. Se creará si no existe.

Notas:
    - El directorio de salida se creará automáticamente si no existe
    - El dataset local debe estar en formato Hugging Face (usando save_to_disk)
    - Los embeddings se guardan en formato Parquet, un archivo por capa
    - Se generan 13 archivos de embeddings (12 capas + embeddings de entrada)
    - Los archivos se nombran como: embeddings_{dataset}_layer_{N}.parquet
    - MLflow organiza los artefactos por capa para mejor seguimiento
"""

from __future__ import annotations

import argparse
import hashlib
import re
from collections import defaultdict
from pathlib import Path
from typing import List, Union

import mlflow
import pandas as pd
import torch
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
from transformers import RobertaModel, RobertaTokenizer

# Conditionally import cuDF for GPU-accelerated DataFrames
try:
    import cudf
    import cupy
    RAPIDS_AVAILABLE = True
except ImportError:
    RAPIDS_AVAILABLE = False
    cudf = None
    cupy = None

# Define column mappings for different datasets
COLUMN_MAPPINGS = {
    "snli": {
        "src_cols": ["premise", "hypothesis", "label"],
        "tgt_cols": ["premise", "conclusion", "label"],
        "rename_map": {"hypothesis": "conclusion"}
    },
    "folio": {
        "src_cols": ["premises", "conclusion", "label"],
        "tgt_cols": ["premise", "conclusion", "label"],
        "rename_map": {"premises": "premise"}
    }
    # Add more dataset mappings here as needed
}

###############################################################################
# CLI
###############################################################################

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--save_every", type=int, default=100)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    p.add_argument("--source_path")
    p.add_argument("--experiment_name", default="embeddings-generator")
    p.add_argument("--dataset", default="snli", help="Name of the dataset to use from Hugging Face")
    p.add_argument("--layer_num", default="0-12", help="Layer range to process (e.g., '12' or '0-12'). Inclusive.")
    p.add_argument("--combine_only", action="store_true", help="Only combine existing part files, do not generate new embeddings.")
    return p.parse_args()

def parse_layer_range(range_str: str) -> range:
    """Parses a layer range string like '0-12' or '12' into a range object."""
    if not isinstance(range_str, str):
        raise TypeError("layer_num must be a string.")
    try:
        if '-' in range_str:
            start, end = map(int, range_str.split('-'))
            if not (0 <= start <= end <= 12):
                raise ValueError("Layer range must be between 0 and 12, with start <= end.")
            return range(start, end + 1)
        else:
            layer_num = int(range_str)
            if not (0 <= layer_num <= 12):
                raise ValueError("Layer number must be between 0 and 12.")
            return range(layer_num, layer_num + 1)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid layer range format: '{range_str}'. Use 'N' or 'N-M'. Original error: {e}")

###############################################################################
# DATASET
###############################################################################

def ensure_columns(ds, required_cols):
    cols = ds.column_names if hasattr(ds, "column_names") else ds.columns
    missing = set(required_cols) - set(cols)
    if missing:
        raise KeyError(f"Missing columns: {missing}")

def load_dataset_with_mapping(src, dataset_name):
    # Get column mappings for the dataset
    mapping = COLUMN_MAPPINGS.get(dataset_name, COLUMN_MAPPINGS["snli"])  # fallback to SNLI mapping
    src_cols = mapping["src_cols"]
    tgt_cols = mapping["tgt_cols"]
    rename_map = mapping["rename_map"]
    
    # Load dataset
    ds = load_from_disk(src) if src else load_dataset(dataset_name)
    ensure_columns(ds, src_cols)
    
    # Apply column mappings
    if hasattr(ds, "rename_columns"):
        ds = ds.rename_columns(rename_map)
        ds = ds.remove_columns([c for c in ds.column_names if c not in tgt_cols])
    else:
        ds = ds.rename(columns=rename_map)[tgt_cols]
    return ds

###############################################################################
# EMBEDDINGS
###############################################################################

def encode_batch(texts: List[str], tok, model, device):
    enc = tok(texts, padding=True, truncation=True, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        outputs = model(**enc, output_hidden_states=True)
        # Get all layer embeddings (including input embeddings)
        all_layers = outputs.hidden_states
        # Return list of [CLS] token embeddings for each layer
        return [layer[:, 0] for layer in all_layers]  # Keep on device

###############################################################################
# SAVING LOGIC
###############################################################################

def save_dataframe_part(
    vectors_list: list,
    labels: list,
    premise_ids: list,
    hypothesis_ids: list,
    file_path: Path,
    use_cudf: bool
):
    """Saves a list of vector tensors and labels to a Parquet file."""
    if not vectors_list:
        return

    if use_cudf:
        vec_tensor = torch.cat(vectors_list)
        vec_cupy = cupy.from_dlpack(torch.to_dlpack(vec_tensor))
        df = cudf.DataFrame(vec_cupy)
        df.columns = [f'feature_{j}' for j in range(df.shape[1])]
        df['label'] = labels
        df['premise_id'] = premise_ids
        df['hypothesis_id'] = hypothesis_ids
    else:
        vec_tensor = torch.cat(vectors_list).cpu().float().numpy()
        df = pd.DataFrame(vec_tensor, columns=[f'feature_{j}' for j in range(vec_tensor.shape[1])])
        df['label'] = labels
        df['premise_id'] = premise_ids
        df['hypothesis_id'] = hypothesis_ids
    
    df.to_parquet(file_path)

def combine_and_cleanup_parts(
    part_paths: list,
    output_path: Path,
    use_cudf: bool
):
    """Combines partial Parquet files into a single file and deletes the parts."""
    if not part_paths:
        return
    
    print(f"Combining {len(part_paths)} partial files into {output_path.name}...")
    df_lib = cudf if use_cudf else pd
    
    combined_df = df_lib.concat(
        [df_lib.read_parquet(p) for p in part_paths], 
        ignore_index=True
    )
    
    combined_df.to_parquet(output_path)
    
    for p in part_paths:
        p.unlink()

###############################################################################
# MAIN
###############################################################################

def main():
    args = parse_args()

    if args.combine_only:
        out_dir = Path(args.out)
        if not out_dir.is_dir():
            raise SystemExit(f"Output directory not found for combining: {out_dir}")

        use_gpu = args.device == "cuda"
        if use_gpu and not RAPIDS_AVAILABLE:
            print("WARNING: --device cuda was specified, but RAPIDS is not available. Using pandas for combining.")
            use_cudf = False
        else:
            use_cudf = use_gpu
        
        print(f"Combine-only mode. Scanning for parts in {out_dir}...")

        part_paths = defaultdict(list)
        part_paths_delta = defaultdict(list)
        
        pattern = re.compile(f"embeddings_{args.dataset}_layer_(\\d+)(_delta)?_part\\d+\\.parquet")

        found_files = list(out_dir.glob(f"embeddings_{args.dataset}_layer_*_part*.parquet"))
        if not found_files:
            print("No part files found to combine.")
            return

        for part_file in found_files:
            match = pattern.match(part_file.name)
            if match:
                layer_idx, is_delta_str = match.groups()
                layer_idx = int(layer_idx)
                
                if is_delta_str:
                    part_paths_delta[layer_idx].append(part_file)
                else:
                    part_paths[layer_idx].append(part_file)
        
        all_layers = sorted(set(part_paths.keys()) | set(part_paths_delta.keys()))
        print(f"Found parts for layers: {all_layers}")

        for layer_idx in all_layers:
            if layer_parts := sorted(part_paths.get(layer_idx, [])):
                final_path = out_dir / f"embeddings_{args.dataset}_layer_{layer_idx}.parquet"
                combine_and_cleanup_parts(layer_parts, final_path, use_cudf)
                if final_path.exists():
                    print(f"  ✅ Layer {layer_idx} (concatenated) saved to {final_path}")

            if delta_parts := sorted(part_paths_delta.get(layer_idx, [])):
                final_path_delta = out_dir / f"embeddings_{args.dataset}_layer_{layer_idx}_delta.parquet"
                combine_and_cleanup_parts(delta_parts, final_path_delta, use_cudf)
                if final_path_delta.exists():
                    print(f"  ✅ Layer {layer_idx} (delta) saved to {final_path_delta}")
        return

    try:
        layers_to_process = parse_layer_range(args.layer_num)
    except ValueError as e:
        raise SystemExit(e)

    use_gpu = args.device == "cuda"
    if use_gpu:
        if not torch.cuda.is_available():
            raise SystemExit("GPU not detected")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        if not RAPIDS_AVAILABLE:
            raise SystemExit(
                "ERROR: --device cuda requires RAPIDS (cuDF, cupy) to be installed. "
                "Please install RAPIDS or run with --device cpu."
            )
    
    use_cudf = use_gpu and RAPIDS_AVAILABLE
    dtype = torch.float16 if args.precision == "fp16" else torch.float32
    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run(run_name=f"{args.dataset}_all") as run:
        for k, v in vars(args).items():
            mlflow.log_param(k, v)
        mlflow.log_param("layer_num", args.layer_num)

        tok = RobertaTokenizer.from_pretrained("roberta-base")
        model = RobertaModel.from_pretrained("roberta-base", torch_dtype=dtype).to(args.device).eval()
        ds = load_dataset_with_mapping(args.source_path, args.dataset)
        mlflow.log_metric("n_samples", int(len(ds)))

        # Ensure output directory exists
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Initialize storage for each layer to be processed
        tmp_vecs = {layer: [] for layer in layers_to_process}
        tmp_delta_vecs = {layer: [] for layer in layers_to_process}
        tmp_labels = []
        tmp_premise_ids = []
        tmp_hypothesis_ids = []
        
        part_paths = {layer: [] for layer in layers_to_process}
        part_paths_delta = {layer: [] for layer in layers_to_process}
        part_idx = 0

        for i in tqdm(range(0, len(ds), args.batch_size), desc="Encoding", unit="batch"):
            # Get embeddings for all layers
            premises = ds["premise"][i:i+args.batch_size]
            conclusions = ds["conclusion"][i:i+args.batch_size]
            p_vecs = encode_batch(premises, tok, model, args.device)
            c_vecs = encode_batch(conclusions, tok, model, args.device)
            
            # Process only the selected layers
            for layer_idx, (p_vec, c_vec) in enumerate(zip(p_vecs, c_vecs)):
                if layer_idx in layers_to_process:
                    delta_vec = p_vec - c_vec
                    concatenated_vectors = torch.cat([p_vec, c_vec, delta_vec], dim=1)
                    tmp_vecs[layer_idx].append(concatenated_vectors)
                    tmp_delta_vecs[layer_idx].append(delta_vec) # Store delta separately
            
            tmp_labels.extend(list(ds["label"][i:i+args.batch_size]))

            # Generate and store hashes
            tmp_premise_ids.extend([hashlib.sha256(p.encode('utf-8')).hexdigest() for p in premises])
            tmp_hypothesis_ids.extend([hashlib.sha256(c.encode('utf-8')).hexdigest() for c in conclusions])

            finished_batch = (i // args.batch_size) + 1
            is_last = i + args.batch_size >= len(ds)
            if finished_batch % args.save_every == 0 or is_last:
                # Save each layer's concatenated and delta vectors separately
                for layer_idx in layers_to_process:
                    # Save concatenated vectors
                    part_name = f"embeddings_{args.dataset}_layer_{layer_idx}_part{part_idx:04d}.parquet"
                    part_path = out_dir / part_name
                    save_dataframe_part(tmp_vecs[layer_idx], tmp_labels, tmp_premise_ids, tmp_hypothesis_ids, part_path, use_cudf)
                    if tmp_vecs[layer_idx]:
                        part_paths[layer_idx].append(part_path)

                    # Save delta vectors
                    delta_part_name = f"embeddings_{args.dataset}_layer_{layer_idx}_delta_part{part_idx:04d}.parquet"
                    delta_part_path = out_dir / delta_part_name
                    save_dataframe_part(tmp_delta_vecs[layer_idx], tmp_labels, tmp_premise_ids, tmp_hypothesis_ids, delta_part_path, use_cudf)
                    if tmp_delta_vecs[layer_idx]:
                        part_paths_delta[layer_idx].append(delta_part_path)

                # Reset storage
                tmp_vecs = {layer: [] for layer in layers_to_process}
                tmp_delta_vecs = {layer: [] for layer in layers_to_process}
                tmp_labels = []
                tmp_premise_ids = []
                tmp_hypothesis_ids = []
                part_idx += 1

        # Combine parts for each layer and each type of vector
        for layer_idx in layers_to_process:
            # Combine concatenated vectors
            final_path = out_dir / f"embeddings_{args.dataset}_layer_{layer_idx}.parquet"
            combine_and_cleanup_parts(part_paths[layer_idx], final_path, use_cudf)
            if final_path.exists():
                print(f"  ✅ Layer {layer_idx} (concatenated) saved to {final_path}")

            # Combine delta vectors
            final_path_delta = out_dir / f"embeddings_{args.dataset}_layer_{layer_idx}_delta.parquet"
            combine_and_cleanup_parts(part_paths_delta[layer_idx], final_path_delta, use_cudf)
            if final_path_delta.exists():
                print(f"  ✅ Layer {layer_idx} (delta) saved to {final_path_delta}")


if __name__ == "__main__":
    main()
