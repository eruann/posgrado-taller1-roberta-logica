#!/usr/bin/env python
"""
scripts/run_comparison_pipeline.py
=================================

Comparison pipeline script that runs two complete sweeps:
1. Original embeddings (no normalization)
2. All-but-mean normalized embeddings

This allows direct comparison to determine if all-but-mean normalization
is causing the single cluster issue.

Usage:
    python scripts/run_comparison_pipeline.py \\
        --dataset_name snli \\
        --base_data_dir data/snli \\
        --output_dir data/snli/comparison_test
"""

import argparse
import subprocess
import shutil
import json
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Run comparison pipeline: original vs all-but-mean normalization.")
    parser.add_argument("--dataset_name", required=True, help="Name of the dataset (e.g., snli, folio)")
    parser.add_argument("--base_data_dir", required=True, type=Path, help="Base directory of the source embeddings")
    parser.add_argument("--output_dir", required=True, type=Path, help="Base directory for all pipeline outputs")
    parser.add_argument("--clean_output", action='store_true', help="If set, removes the output directory before starting.")
    parser.add_argument("--layer_num", type=int, default=9, help="Layer number to test (default: 9)")
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

def run_sweep(args, sweep_name: str, source_file: Path, use_normalization: bool):
    """Run a complete sweep for either original or normalized data."""
    print(f"\n{'='*60}")
    print(f"=== RUNNING SWEEP: {sweep_name.upper()} ===")
    print(f"=== Source: {source_file} ===")
    print(f"{'='*60}")
    
    # --- Parameter Matrix ---
    PCA_COMPONENTS = [1, 5, 50]
    SLICE_N_VALUES = [0, 3, 5]
    UMAP_NEIGHBORS = [15, 100, 200, 300]
    UMAP_METRICS_PCA = ["euclidean", "manhattan"]
    UMAP_METRICS_ZCA = ["euclidean"]
    KMEANS_K = 3

    sweep_output_dir = args.output_dir / sweep_name
    sweep_output_dir.mkdir(parents=True, exist_ok=True)
    
    provenance = {"source_embedding": "full", "layer": args.layer_num, "normalization": sweep_name}

    # --- STAGE 1: Normalization (or copy original) ---
    norm_dir = sweep_output_dir / "01_source" / f"layer_{args.layer_num}"
    norm_dir.mkdir(parents=True, exist_ok=True)
    
    if use_normalization:
        normalized_path = norm_dir / source_file.name.replace(".parquet", "_normalized.parquet")
        print(f"\n[STAGE 1/5] All-but-mean Normalization")
        print(f"  Input:  {source_file}")
        print(f"  Output: {normalized_path}")
        if not normalized_path.exists():
            provenance_json = json.dumps(provenance)
            run_command([
                "python", "experiments/05_all_but_mean.py",
                "--source_path", str(source_file),
                "--out_path", str(normalized_path),
                "--experiment_name", f"norm_{args.dataset_name}_comparison",
                "--layer_num", str(args.layer_num),
                "--provenance", provenance_json
            ])
        input_for_pca = normalized_path
    else:
        # Just copy the original file
        original_copy_path = norm_dir / source_file.name
        print(f"\n[STAGE 1/5] Using Original Data (No Normalization)")
        print(f"  Input:  {source_file}")
        print(f"  Output: {original_copy_path}")
        if not original_copy_path.exists():
            shutil.copy2(source_file, original_copy_path)
        input_for_pca = original_copy_path

    if not input_for_pca.exists():
        print(f"ERROR: Source file not found: {input_for_pca}")
        return

    # --- STAGE 2: PCA/ZCA ---
    for n_comp in PCA_COMPONENTS:
        prov_pca = provenance.copy()
        prov_pca["pca_components_initial"] = n_comp
        
        pca_dir = sweep_output_dir / "02_pca" / f"layer_{args.layer_num}" / f"{n_comp}_components"
        pca_dir.mkdir(parents=True, exist_ok=True)
        pca_out_path = pca_dir / f"{args.dataset_name}_full_{n_comp}_layer{args.layer_num}.parquet"

        print(f"\n[STAGE 2/5] PCA/ZCA (Components: {n_comp})")
        print(f"  Input:  {input_for_pca}")
        print(f"  Output Dir: {pca_dir}")
        if not list(pca_dir.glob(f"*_{n_comp}_layer{args.layer_num}.parquet")):
            run_command([
                "python", "experiments/01_pca.py", 
                "--source_path", str(input_for_pca), 
                "--out", str(pca_out_path), 
                "--n_components", str(n_comp), 
                "--experiment_name", f"pca_comparison_{sweep_name}", 
                "--dataset", args.dataset_name, 
                "--layer_num", str(args.layer_num)
            ])

        # --- STAGE 3: Slicing (or not) ---
        for reduction_type in ["pca", "zca"]:
            pca_variant_path = pca_dir / f"{reduction_type}_{pca_out_path.name}"
            if not pca_variant_path.exists(): 
                continue
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
                    slice_dir = sweep_output_dir / "03_sliced" / f"layer_{args.layer_num}" / f"skip_{skip_n}"
                    slice_dir.mkdir(parents=True, exist_ok=True)
                    sliced_path = slice_dir / f"{pca_variant_path.stem}_skipped{skip_n}.parquet"
                    print(f"  Output: {sliced_path}")
                    if not sliced_path.exists():
                        run_command([
                            "python", "scripts/utilities/slice_parquet_vectors.py", 
                            "--input_parquet", str(pca_variant_path), 
                            "--output_parquet", str(sliced_path), 
                            "--skip_first_n", str(skip_n)
                        ])
                    if not sliced_path.exists(): 
                        continue
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

                        umap_dir = sweep_output_dir / "04_umap" / f"layer_{args.layer_num}" / f"skip_{skip_n}" / reduction_type / f"components_{n_comp}" / f"neighbors_{n_neighbors}" / metric
                        umap_dir.mkdir(parents=True, exist_ok=True)
                        
                        umap_out_path = umap_dir / f"umap_{args.dataset_name}_full_l{args.layer_num}_s{skip_n}_{reduction_type}_c{n_comp}_n{n_neighbors}_{metric}.parquet"

                        print(f"\n[STAGE 4/5] UMAP (Neighbors: {n_neighbors}, Metric: {metric})")
                        print(f"  Input:  {input_for_umap}")
                        print(f"  Output: {umap_out_path}")
                        if not umap_out_path.exists():
                            run_command([
                                "python", "experiments/02_umap.py", 
                                "--pca_path", str(input_for_umap), 
                                "--out_path", str(umap_out_path), 
                                "--n_neighbors", str(n_neighbors), 
                                "--metric", metric, 
                                "--dataset", args.dataset_name, 
                                "--experiment_name", f"umap_comparison_{sweep_name}", 
                                "--reduction_type", reduction_type, 
                                "--layer_num", str(args.layer_num), 
                                "--input_n_components", str(n_comp - skip_n), 
                                "--skipped_n_components", str(skip_n)
                            ])
                        if not umap_out_path.exists(): 
                            continue

                        # --- STAGE 5: K-Means ---
                        kmeans_dir = sweep_output_dir / "05_kmeans" / f"layer_{args.layer_num}" / f"skip_{skip_n}" / reduction_type / f"components_{n_comp}" / f"neighbors_{n_neighbors}" / metric
                        kmeans_dir.mkdir(parents=True, exist_ok=True)
                        kmeans_out_path = kmeans_dir / "kmeans_results.json"
                        
                        provenance_json = json.dumps(prov_metric)

                        print(f"\n[STAGE 5/5] K-Means (k={KMEANS_K})")
                        print(f"  Input:  {umap_out_path}")
                        print(f"  Output: {kmeans_out_path}")
                        if not kmeans_out_path.exists():
                             kmeans_cmd = [
                                 "python", "experiments/03_kmeans.py",
                                 "--input_path", str(umap_out_path),
                                 "--out_path", str(kmeans_out_path),
                                 "--k", str(KMEANS_K),
                                 "--dataset", args.dataset_name,
                                 "--experiment_name", f"kmeans_comparison_{sweep_name}",
                                 "--provenance", provenance_json,
                                 "--reduction_type", "umap",
                                 "--layer_num", str(args.layer_num),
                                 "--input_n_components", str(n_comp - skip_n),
                                 "--umap_n_neighbors", str(n_neighbors),
                                 "--umap_metric", metric,
                                 "--umap_source_reduction_type", reduction_type,
                                 "--skipped_n_components", str(skip_n)
                             ]
                             run_command(kmeans_cmd)

def main():
    args = parse_args()

    if args.clean_output and args.output_dir.exists():
        print("Cleaning output directory...")
        shutil.rmtree(args.output_dir)
    
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Find the source embedding file
    source_dir = args.base_data_dir / "embeddings"
    source_file = source_dir / f"embeddings_{args.dataset_name}_layer_{args.layer_num}.parquet"
    
    if not source_file.exists():
        print(f"ERROR: Source file not found: {source_file}")
        return

    # Run both sweeps
    run_sweep(args, "original", source_file, use_normalization=False)
    run_sweep(args, "all_but_mean", source_file, use_normalization=True)

    print(f"\n{'='*60}")
    print("=== COMPARISON PIPELINE COMPLETED ===")
    print(f"Results saved to: {args.output_dir}")
    print("Compare the cluster plots between:")
    print(f"  - {args.output_dir}/original/05_kmeans/")
    print(f"  - {args.output_dir}/all_but_mean/05_kmeans/")
    print(f"{'='*60}")

if __name__ == "__main__":
    main() 