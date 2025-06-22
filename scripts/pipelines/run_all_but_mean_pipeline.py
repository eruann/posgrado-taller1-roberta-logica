#!/usr/bin/env python
"""
scripts/run_all_but_mean_pipeline.py
===================================

The master pipeline script for "Experiment #1: All-but-mean Normalization".
This script orchestrates a full, end-to-end experimental pipeline that
incorporates normalization, dimensionality reduction, slicing, and clustering.

This pipeline is idempotent: it checks for the existence of output files
at each stage and skips steps if the output already exists.

Pipeline Stages & Parameter Matrix:
1.  **Input**: `full` and `delta` embeddings for layers 9-12.
2.  **Normalization**: Applies "All-but-mean" to each input.
3.  **PCA/ZCA**: Reduces normalized embeddings to `[50]` components.
4.  **Slicing**: Takes the 50-component PCA/ZCA output and creates:
    - The original (un-sliced) version.
    - Sliced versions with the top `[3, 4]` components removed.
5.  **UMAP**: For each PCA/ZCA and sliced variant, projects to 2D using:
    - Neighbors: `[15, 100, 150, 200]`
    - Metrics: `euclidean`, `manhattan` (only for PCA)
6.  **K-Means**: Clusters each UMAP output with `k=3`.
7.  **Logging**: A full dictionary of parameters ("provenance") is passed
    down the chain and logged by the final K-Means step for full reproducibility.

Usage:
    python scripts/run_all_but_mean_pipeline.py \\
        --dataset_name snli \\
        --base_data_dir data/snli \\
        --output_dir data/snli/experiments_all_but_mean
"""

import argparse
import subprocess
import shutil
import json
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Run full pipeline with All-but-mean normalization.")
    parser.add_argument("--dataset_name", required=True, help="Name of the dataset (e.g., snli, folio)")
    parser.add_argument("--base_data_dir", required=True, type=Path, help="Base directory of the source embeddings")
    parser.add_argument("--output_dir", required=True, type=Path, help="Base directory for all pipeline outputs")
    parser.add_argument("--clean_output", action='store_true', help="If set, removes the output directory before starting.")
    return parser.parse_args()

def run_command(cmd: list[str]):
    """Executes a command, raising an exception if it fails."""
    print(f"Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"\n--- ERROR ---")
        print(f"COMMAND FAILED: {' '.join(e.cmd)}")
        print(f"STDERR:\n{e.stderr}")
        print(f"STDOUT:\n{e.stdout}")
        print(f"-------------")
        raise e

def main():
    args = parse_args()

    if args.clean_output and args.output_dir.exists():
        print("Performing selective clean of pipeline outputs (preserving normalized files)...")
        dirs_to_clean = [
            args.output_dir / "02_pca",
            args.output_dir / "03_sliced",
            args.output_dir / "04_umap",
            args.output_dir / "05_kmeans"
        ]
        for d in dirs_to_clean:
            if d.exists():
                print(f"  - Removing {d}")
                shutil.rmtree(d)
    
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # --- Parameter Matrix ---
    PCA_COMPONENTS = [1, 5, 50]  # Match original experiments: minimal, low, and high dimensionality
    SLICE_N_VALUES = [0, 3, 5, 10]  # 0 means no slicing
    UMAP_NEIGHBORS = [15, 100, 150, 200, 300]
    UMAP_METRICS_PCA = ["euclidean", "manhattan"]
    UMAP_METRICS_ZCA = ["euclidean"]
    KMEANS_K = 3

    # === Main Loop: Iterate over embedding types and layers ===
    source_dirs = {
        'full': args.base_data_dir / "embeddings", 
        'delta': args.base_data_dir / "difference_embeddings"
    }
    for emb_type, source_dir in source_dirs.items():
        if not source_dir.exists(): continue
        print(f"\n{'='*40}\n=== PROCESSING EMBEDDING: {emb_type.upper()} ===\n{'='*40}")

        for layer_num in range(9, 13):
            source_file = next(source_dir.glob(f"*_layer_{layer_num}.parquet"), None)
            if not source_file: continue
            print(f"\n{'~'*30} Layer {layer_num} {'~'*30}")

            provenance = {"source_embedding": emb_type, "layer": layer_num, "normalization": "all_but_mean"}

            # --- STAGE 1: Normalization ---
            norm_dir = args.output_dir / "01_normalized" / emb_type / f"layer_{layer_num}"
            norm_dir.mkdir(parents=True, exist_ok=True)
            normalized_path = norm_dir / source_file.name.replace(".parquet", "_normalized.parquet")
            
            print("\n[STAGE 1/5] Normalization")
            print(f"  Input:  {source_file}")
            print(f"  Output: {normalized_path}")
            if not normalized_path.exists() or normalized_path.stat().st_size < 100:
                provenance_json = json.dumps(provenance)
                run_command([
                    "python", "experiments/05_all_but_mean.py",
                    "--source_path", str(source_file),
                    "--out_path", str(normalized_path),
                    "--experiment_name", f"norm_{args.dataset_name}_{emb_type}",
                    "--layer_num", str(layer_num),
                    "--provenance", provenance_json
                ])
            if not normalized_path.exists(): continue

            # --- STAGE 2: PCA/ZCA ---
            for n_comp in PCA_COMPONENTS:
                prov_pca = provenance.copy()
                prov_pca["pca_components_initial"] = n_comp
                
                pca_dir = args.output_dir / "02_pca" / emb_type / f"layer_{layer_num}" / f"{n_comp}_components"
                pca_dir.mkdir(parents=True, exist_ok=True)
                pca_out_path = pca_dir / f"{args.dataset_name}_{emb_type}_{n_comp}_layer{layer_num}.parquet"

                print(f"\n[STAGE 2/5] PCA/ZCA (Components: {n_comp})")
                print(f"  Input:  {normalized_path}")
                print(f"  Output Dir: {pca_dir}")
                if not list(pca_dir.glob(f"*_{n_comp}_layer{layer_num}.parquet")):
                    run_command(["python", "experiments/01_pca.py", "--source_path", str(normalized_path), "--out", str(pca_out_path), "--n_components", str(n_comp), "--experiment_name", f"pca_on_norm_{args.dataset_name}", "--dataset", args.dataset_name, "--layer_num", str(layer_num)])

                # --- STAGE 3: Slicing (or not) ---
                for reduction_type in ["pca", "zca"]:
                    pca_variant_path = pca_dir / f"{reduction_type}_{pca_out_path.name}"
                    if not pca_variant_path.exists(): continue
                    prov_reduc = prov_pca.copy()
                    prov_reduc["dim_reduction_initial"] = reduction_type

                    for skip_n in SLICE_N_VALUES:
                        # Skip slicing if skip_n is greater than or equal to available components
                        available_components = n_comp
                        if skip_n >= available_components:
                            print(f"  Skipping slice with skip_n={skip_n} (only {available_components} components available)")
                            continue
                            
                        prov_slice = prov_reduc.copy()
                        prov_slice["pca_components_sliced_off"] = skip_n
                        
                        print(f"\n[STAGE 3/5] Slicing (Source: {reduction_type.upper()}, Skip: {skip_n})")
                        print(f"  Input: {pca_variant_path}")
                        
                        input_for_umap = pca_variant_path
                        if skip_n > 0:
                            slice_dir = args.output_dir / "03_sliced" / emb_type / f"layer_{layer_num}" / f"skip_{skip_n}"
                            slice_dir.mkdir(parents=True, exist_ok=True)
                            sliced_path = slice_dir / f"{pca_variant_path.stem}_skipped{skip_n}.parquet"
                            print(f"  Output: {sliced_path}")
                            if not sliced_path.exists():
                                run_command(["python", "scripts/utilities/slice_parquet_vectors.py", "--input_parquet", str(pca_variant_path), "--output_parquet", str(sliced_path), "--skip_first_n", str(skip_n)])
                            if not sliced_path.exists(): continue
                            input_for_umap = sliced_path
                        else:
                            print("  Action: No slicing performed.")
                        
                        # --- STAGE 4: UMAP ---
                        metrics = UMAP_METRICS_PCA if reduction_type == "pca" else UMAP_METRICS_ZCA
                        for n_neighbors in UMAP_NEIGHBORS:
                            prov_umap = prov_slice.copy()
                            prov_umap["umap_neighbors"] = n_neighbors
                            for metric in metrics:
                                prov_metric = prov_umap.copy()
                                prov_metric["umap_metric"] = metric

                                umap_dir = args.output_dir / "04_umap" / emb_type / f"layer_{layer_num}" / f"skip_{skip_n}" / reduction_type / f"components_{n_comp}" / f"neighbors_{n_neighbors}" / metric
                                umap_dir.mkdir(parents=True, exist_ok=True)
                                
                                umap_out_path = umap_dir / f"umap_{args.dataset_name}_{emb_type}_l{layer_num}_s{skip_n}_{reduction_type}_c{n_comp}_n{n_neighbors}_{metric}.parquet"

                                print(f"\n[STAGE 4/5] UMAP (Neighbors: {n_neighbors}, Metric: {metric})")
                                print(f"  Input:  {input_for_umap}")
                                print(f"  Output: {umap_out_path}")
                                if not umap_out_path.exists():
                                    run_command(["python", "experiments/02_umap.py", "--pca_path", str(input_for_umap), "--out_path", str(umap_out_path), "--n_neighbors", str(n_neighbors), "--metric", metric, "--dataset", args.dataset_name, "--experiment_name", f"umap_on_norm_{args.dataset_name}", "--reduction_type", reduction_type, "--layer_num", str(layer_num), "--input_n_components", str(n_comp - skip_n), "--skipped_n_components", str(skip_n)])
                                if not umap_out_path.exists(): continue

                                # --- STAGE 5: K-Means ---
                                kmeans_dir = args.output_dir / "05_kmeans" / emb_type / f"layer_{layer_num}" / f"skip_{skip_n}" / reduction_type / f"components_{n_comp}" / f"neighbors_{n_neighbors}" / metric
                                kmeans_dir.mkdir(parents=True, exist_ok=True)
                                kmeans_out_path = kmeans_dir / "kmeans_results.json"
                                
                                provenance_json = json.dumps(prov_metric)

                                print(f"\n[STAGE 5/5] K-Means (k={KMEANS_K})")
                                print(f"  Input:  {umap_out_path}")
                                print(f"  Output: {kmeans_out_path}")
                                if not kmeans_out_path.exists():
                                     kmeans_cmd = ["python", "experiments/03_kmeans.py",
                                         "--input_path", str(umap_out_path),
                                         "--out_path", str(kmeans_out_path),
                                         "--k", str(KMEANS_K),
                                         "--dataset", args.dataset_name,
                                         "--experiment_name", f"kmeans_on_norm_{args.dataset_name}",
                                         "--provenance", provenance_json,
                                         "--reduction_type", "umap",
                                         "--layer_num", str(layer_num),
                                         "--input_n_components", str(n_comp - skip_n),
                                         "--umap_n_neighbors", str(n_neighbors),
                                         "--umap_metric", metric,
                                         "--umap_source_reduction_type", reduction_type,
                                         "--skipped_n_components", str(skip_n)
                                     ]
                                     run_command(kmeans_cmd)

    print("\nFull 'All-but-mean' Uber-Pipeline completed.")

if __name__ == "__main__":
    main() 