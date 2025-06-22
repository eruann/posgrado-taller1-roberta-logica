#!/usr/bin/env python
"""
scripts/extract_delta.py
=========================
Extrae únicamente el vector “delta” (diferencia) de un Parquet con vectores
3×d (concatenación + diferencia) de PCA o de embeddings.

Usage:
------
python scripts/extract_delta.py \
    --pca_path data/snli_train_pca60.parquet \
    --out_delta data/snli_train_delta60.parquet
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(
        description="Extrae la parte delta de un vector 3*d (prem|conc|diff)"
    )
    p.add_argument(
        "--pca_path", required=True,
        help="Ruta al Parquet con columnas 'vector' (list[float]) y 'label'"
    )
    p.add_argument(
        "--out_delta", required=True,
        help="Salida Parquet con las columnas 'vector' (solo delta) y 'label'"
    )
    return p.parse_args()


def main():
    args = parse_args()
    df = pd.read_parquet(args.pca_path)

    # Apilamos los vectores en un array (n_samples, D)
    X = np.vstack(df["vector"].values)
    n, D = X.shape

    # Verificar divisibilidad en 3 partes iguales
    if D % 3 != 0:
        sys.exit(f"ERROR: dimensión {D} no divisible por 3 (esperado 3*d).")
    d = D // 3

    # Extraer solo la tercera parte: delta = v_premise - v_conclusion
    delta = X[:, 2 * d :]

    # Construir DataFrame de salida
    df_out = pd.DataFrame({
        "vector": delta.tolist(),
        "label": df["label"].values,
    })

    # Guardar Parquet
    out_path = Path(args.out_delta)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(out_path)

    print(f"✅ Delta vectors saved ({n} samples, {d} dims) → {out_path}")


if __name__ == "__main__":
    main()
