#!/usr/bin/env python
"""
scripts/npy_to_parquet.py – Convert NPY files to Parquet
=====================================================
Convierte archivos NPY de UMAP a formato Parquet manteniendo la misma estructura
que los archivos PCA. Preserva las etiquetas originales de los archivos CSV.

Uso:
-----
python scripts/npy_to_parquet.py \
    --npy_dir data/snli/umap \
    --out_dir data/snli/umap
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Convert NPY to Parquet")
    parser.add_argument("--npy_dir", required=True, help="Directorio con archivos NPY")
    parser.add_argument("--out_dir", required=True, help="Directorio de salida para Parquet")
    return parser.parse_args()

def main():
    args = parse_args()
    npy_dir = Path(args.npy_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find all NPY files
    npy_files = list(npy_dir.glob("*.npy"))
    print(f"Found {len(npy_files)} NPY files to convert")

    for npy_file in tqdm(npy_files, desc="Converting files"):
        # Load NPY file
        X = np.load(npy_file)
        
        # Load corresponding CSV file for labels
        csv_file = npy_file.with_suffix('.csv')
        if not csv_file.exists():
            print(f"⚠️ Warning: No CSV file found for {npy_file.name}, skipping...")
            continue
            
        try:
            labels_df = pd.read_csv(csv_file)
            if 'label' not in labels_df.columns:
                print(f"⚠️ Warning: No 'label' column in {csv_file.name}, skipping...")
                continue
            labels = labels_df['label'].values
        except Exception as e:
            print(f"⚠️ Error reading {csv_file.name}: {str(e)}, skipping...")
            continue
        
        # Verify lengths match
        if len(X) != len(labels):
            print(f"⚠️ Warning: Length mismatch between {npy_file.name} ({len(X)}) and {csv_file.name} ({len(labels)}), skipping...")
            continue
        
        # Create DataFrame with same structure as PCA/UMAP parquet files
        df = pd.DataFrame({
            "vector": X.tolist(),
            "label": labels
        })
        
        # Save as parquet with same name but .parquet extension
        out_file = out_dir / f"{npy_file.stem}.parquet"
        df.to_parquet(out_file)
        print(f"✅ Converted {npy_file.name} → {out_file.name}")

if __name__ == "__main__":
    main() 