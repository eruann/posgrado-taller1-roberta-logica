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
from pathlib import Path
from typing import List, Union

import mlflow
import pandas as pd
import torch
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
from transformers import RobertaModel, RobertaTokenizer

# === Set tracking URI for MLflow ===
tracking_dir = Path.cwd().joinpath("mlruns")
mlflow.set_tracking_uri(tracking_dir.as_uri())

# Define column mappings for different datasets
COLUMN_MAPPINGS = {
    "snli": {
        "src_cols": ["premise", "hypothesis", "label"],
        "tgt_cols": ["premise", "conclusion", "label"],
        "rename_map": {"hypothesis": "conclusion"}
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
    p.add_argument("--layer_num", default="0-12", help="Layer range to process (default: 0-12 for all layers)")
    return p.parse_args()

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
        return [layer[:, 0].cpu() for layer in all_layers]

###############################################################################
# MAIN
###############################################################################

def main():
    args = parse_args()
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("GPU not detected")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    dtype = torch.float16 if args.precision == "fp16" else torch.float32
    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run(run_name=f"{args.dataset}_all") as run:
        for k, v in vars(args).items():
            mlflow.log_param(k, v)
        mlflow.log_param("layer_num", args.layer_num)

        tok = RobertaTokenizer.from_pretrained("roberta-base")
        model = RobertaModel.from_pretrained("roberta-base", torch_dtype=dtype).to(args.device).eval()
        ds = load_dataset_with_mapping(args.source_path, args.dataset)
        mlflow.log_metric("n_samples", len(ds))

        # Ensure output directory exists
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Initialize storage for each layer
        tmp_vecs = [[] for _ in range(13)]  # 12 layers + input embeddings
        tmp_labels = []
        part_paths = []
        part_idx = 0

        for i in tqdm(range(0, len(ds), args.batch_size), desc="Encoding", unit="batch"):
            # Get embeddings for all layers
            p_vecs = encode_batch(ds["premise"][i:i+args.batch_size], tok, model, args.device)
            c_vecs = encode_batch(ds["conclusion"][i:i+args.batch_size], tok, model, args.device)
            
            # Process each layer
            for layer_idx, (p_vec, c_vec) in enumerate(zip(p_vecs, c_vecs)):
                diff = p_vec - c_vec
                tmp_vecs[layer_idx].append(torch.cat([p_vec, c_vec, diff], dim=1))
            
            tmp_labels.extend(list(ds["label"][i:i+args.batch_size]))

            finished_batch = (i // args.batch_size) + 1
            is_last = i + args.batch_size >= len(ds)
            if finished_batch % args.save_every == 0 or is_last:
                # Save each layer separately
                for layer_idx in range(13):
                    part_df = pd.DataFrame({
                        "vector": torch.cat(tmp_vecs[layer_idx]).cpu().float().numpy().tolist(),
                        "label": tmp_labels,
                    })
                    part_name = f"embeddings_{args.dataset}_layer_{layer_idx}_part{part_idx:04d}.parquet"
                    part_path = out_dir / part_name
                    part_df.to_parquet(part_path)
                    mlflow.log_artifact(str(part_path), artifact_path=f"partials/layer_{layer_idx}", copy=False)
                    part_paths.append(part_path)
                
                # Reset storage
                tmp_vecs = [[] for _ in range(13)]
                tmp_labels = []
                part_idx += 1

        # Combine parts for each layer
        for layer_idx in range(13):
            layer_parts = [p for p in part_paths if f"_layer_{layer_idx}_" in str(p)]
            # Combine partial files
            print(f"  Combinando {len(layer_parts)} archivos parciales...")
            combined_vectors = []
            for part_path in layer_parts:
                part_df = pd.read_parquet(part_path)
                combined_vectors.append(part_df)
                # Eliminar archivo parcial después de leerlo
                part_path.unlink()
            
            # Combine and save
            combined_df = pd.concat(combined_vectors, ignore_index=True)
            layer_out_path = out_dir / f"embeddings_{args.dataset}_layer_{layer_idx}.parquet"
            combined_df.to_parquet(layer_out_path)
            print(f"  ✅ Guardado en {layer_out_path}")
            
            # Log final file only
            try:
                mlflow.log_artifact(str(layer_out_path), artifact_path="embeddings", copy=False)
            except Exception:
                print("⚠️ No se pudo loguear el archivo final en MLflow")


if __name__ == "__main__":
    main()
