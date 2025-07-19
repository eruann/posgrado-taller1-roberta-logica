#!/usr/bin/env python3
"""
Test simple para verificar paths y datos
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path

def test_paths():
    print("=== TESTING PATHS ===")
    
    # Test paths
    base_path = Path("data/snli/norm_comp_delta_ec_only")
    print(f"Base path exists: {base_path.exists()}")
    
    kmeans_path = base_path / "05_kmeans_delta_all_but_mean/layer_12/skip_30/zca/components_50/neighbors_15/euclidean/kmeans_results.json"
    print(f"Kmeans file exists: {kmeans_path.exists()}")
    
    umap_path = base_path / "04_umap_delta_all_but_mean/layer_12/skip_30/zca/components_50/neighbors_15/euclidean"
    print(f"UMAP dir exists: {umap_path.exists()}")
    
    if umap_path.exists():
        umap_files = list(umap_path.glob("*.parquet"))
        print(f"UMAP files found: {len(umap_files)}")
        if umap_files:
            print(f"First UMAP file: {umap_files[0]}")
    
    # Try to load data
    if kmeans_path.exists():
        try:
            with open(kmeans_path) as f:
                data = json.load(f)
            print(f"Kmeans results: {data}")
        except Exception as e:
            print(f"Error loading kmeans: {e}")
    
    if umap_path.exists() and list(umap_path.glob("*.parquet")):
        try:
            df = pd.read_parquet(umap_files[0])
            print(f"UMAP data shape: {df.shape}")
            print(f"UMAP columns: {list(df.columns)}")
            print(f"Label values: {df['label'].unique()}")
        except Exception as e:
            print(f"Error loading UMAP: {e}")

if __name__ == "__main__":
    test_paths() 