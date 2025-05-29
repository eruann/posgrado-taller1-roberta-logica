#!/usr/bin/env python
"""
experiments/00_embeddings.py – SNLI · RoBERTa-base (guardado incremental fix)
============================================================================
Soluciona el TypeError al concatenar `tmp_vectors` (mezcla de `ndarray`).
Ahora se guardan tensores y se convierten a NumPy **solo al volcar a Parquet**.
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


SRC_COLS = ["premise", "hypothesis", "label"]
TGT_COLS = ["premise", "conclusion", "label"]

###############################################################################
# CLI
###############################################################################

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="train", choices=["train", "validation", "test"])
    p.add_argument("--out", required=True)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--save_every", type=int, default=100)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    p.add_argument("--source_path")
    p.add_argument("--experiment_name", default="embeddings-roberta-base-SNLI")
    return p.parse_args()

###############################################################################
# DATASET
###############################################################################

def ensure_columns(ds):
    cols = ds.column_names if hasattr(ds, "column_names") else ds.columns
    missing = set(SRC_COLS) - set(cols)
    if missing:
        raise KeyError(f"Faltan columnas {missing}")


def load_snli(split, src):
    ds = load_from_disk(src) if src else load_dataset("snli", split=split)
    ensure_columns(ds)
    if hasattr(ds, "rename_columns"):
        ds = ds.rename_columns({"hypothesis": "conclusion"})
        ds = ds.remove_columns([c for c in ds.column_names if c not in TGT_COLS])
    else:
        ds = ds.rename(columns={"hypothesis": "conclusion"})[TGT_COLS]
    return ds

###############################################################################
# EMBEDDINGS
###############################################################################

def encode_batch(texts: List[str], tok, model, device):
    enc = tok(texts, padding=True, truncation=True, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        return model(**enc).last_hidden_state[:, 0].cpu()

###############################################################################
# MAIN
###############################################################################

def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("GPU no detectada")

    dtype = torch.float16 if args.precision == "fp16" else torch.float32
    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run(run_name=f"snli_{args.split}") as run:
        for k, v in vars(args).items():
            mlflow.log_param(k, v)

        tok = RobertaTokenizer.from_pretrained("roberta-base")
        model = RobertaModel.from_pretrained("roberta-base", torch_dtype=dtype).to(args.device).eval()
        ds = load_snli(args.split, args.source_path)
        mlflow.log_metric("n_samples", len(ds))

        out_path = Path(args.out)
        tmp_vecs, tmp_labels = [], []
        part_paths = []
        part_idx = 0

        for i in tqdm(range(0, len(ds), args.batch_size), desc="Encoding", unit="batch"):
            p_vec = encode_batch(ds["premise"][i:i+args.batch_size], tok, model, args.device)
            c_vec = encode_batch(ds["conclusion"][i:i+args.batch_size], tok, model, args.device)
            diff = p_vec - c_vec
            tmp_vecs.append(torch.cat([p_vec, c_vec, diff], dim=1))
            tmp_labels.extend(list(ds["label"][i:i+args.batch_size]))

            finished_batch = (i // args.batch_size) + 1
            is_last = i + args.batch_size >= len(ds)
            if finished_batch % args.save_every == 0 or is_last:
                part_df = pd.DataFrame({
                    "vector": torch.cat(tmp_vecs).cpu().float().numpy().tolist(),
                    "label": tmp_labels,
                })
                part_name = f"{out_path.stem}_part{part_idx:04d}.parquet"
                part_path = out_path.with_name(part_name)
                part_df.to_parquet(part_path)
                mlflow.log_artifact(str(part_path), artifact_path="partials")
                part_paths.append(part_path)
                tmp_vecs, tmp_labels = [], []
                part_idx += 1

        # unir partes
        final_df = pd.concat([pd.read_parquet(p) for p in part_paths], ignore_index=True)
        final_df.to_parquet(out_path)
        mlflow.log_artifact(str(out_path), artifact_path="embeddings")
        print("✅ Embeddings finales:", out_path)


if __name__ == "__main__":
    main()
