#!/usr/bin/env python
"""
scripts/run_normalization_comparison_fixed.py
=============================================

Comprehensive comparison pipeline that tests multiple normalization strategies:
1. 'none' - No normalization (original embeddings)
2. 'per_type' - Separate mean removal for each vector type
3. 'standard' - Standard scaling (mean=0, std=1) per vector type

This runs a focused subset of the full pipeline to quickly compare the effect
of different normalization strategies on clustering quality.

Usage:
    python scripts/run_normalization_comparison_fixed.py \
        --dataset_name snli \
        --base_data_dir data/snli \
        --output_dir data/snli/normalization_comparison_fixed \
        --layer_num 9
"""

import argparse
import subprocess
import shutil
import json
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Compare normalization strategies")
    parser.add_argument("--dataset_name", required=True, help="Name of the dataset (e.g., snli)")
    parser.add_argument("--base_data_dir", required=True, type=Path, help="Base directory of the source embeddings")
    parser.add_argument("--output_dir", required=True, type=Path, help="Base directory for comparison outputs")
    parser.add_argument("--layer_num", type=int, default=9, help="Layer number to test")
    parser.add_argument("--clean_output", action='store_true', help="Clean output directory before starting")
    return parser.parse_args()

def run_command(cmd: list[str], allow_failure=False):
    """Executes a command, optionally allowing failures."""
    print(f"Running: {' '.join(cmd)}")
    print("-" * 60)  # Visual separator
    try:
        # Use real-time output instead of capturing
        result = subprocess.run(cmd, check=True, text=True)
        print("-" * 60)  # Visual separator
        print("✅ Command completed successfully")
        return result, True
    except subprocess.CalledProcessError as e:
        print("-" * 60)  # Visual separator
        print(f"\n❌ COMMAND FAILED: {' '.join(cmd[:3])}...")
        print(f"   Return code: {e.returncode}")
        
        if allow_failure:
            print(f"   (Continuing due to allow_failure=True)")
            return None, False
        else:
            print(f"\n--- CRITICAL ERROR - STOPPING PIPELINE ---")
            print(f"COMMAND: {' '.join(e.cmd)}")
            print(f"RETURN CODE: {e.returncode}")
            print(f"----------------------------------------")
            print("Pipeline stopped to prevent wasting time on broken processing.")
            raise e

def check_file_validity(file_path, min_size=1000):
    """Check if a file exists and has reasonable size."""
    if not file_path.exists():
        return False, f"File does not exist: {file_path}"
    
    if file_path.stat().st_size < min_size:
        return False, f"File too small ({file_path.stat().st_size} bytes): {file_path}"
    
    return True, "OK"

def get_data_dimensions(parquet_file):
    """Get the number of dimensions in a parquet file using GPU acceleration."""
    try:
        import cudf
        gdf = cudf.read_parquet(parquet_file)
        # Count non-label columns
        feature_cols = [col for col in gdf.columns if col != 'label']
        return len(feature_cols)
    except Exception as e:
        print(f"⚠️  Could not determine dimensions of {parquet_file}: {e}")
        return None

def main():
    args = parse_args()
    
    if args.clean_output and args.output_dir.exists():
        print(f"Cleaning output directory: {args.output_dir}")
        shutil.rmtree(args.output_dir)
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find source embedding file
    source_dir = args.base_data_dir / args.dataset_name / "embeddings"
    source_file = source_dir / f"embeddings_{args.dataset_name}_layer_{args.layer_num}.parquet"
    
    if not source_file.exists():
        print(f"Source file not found: {source_file}")
        return
    
    print(f"Using source file: {source_file}")
    
    # Test configurations (reduced for quick comparison)
    normalization_types = ['none', 'per_type', 'standard', 'all_but_mean']  # Re-added all_but_mean with fixed columns
    pca_components = [50]  # Single PCA config for quick comparison
    slice_values = [0, 5]  # Test both unsliced and sliced
    umap_neighbors = [100]  # Single neighbor config
    umap_metrics = ['euclidean']  # Single metric
    
    print(f"\n{'='*60}")
    print(f"NORMALIZATION COMPARISON PIPELINE")
    print(f"Layer: {args.layer_num}")
    print(f"Normalization types: {normalization_types}")
    print(f"{'='*60}\n")
    
    # Run comparison for each normalization type
    for norm_type in normalization_types:
        print(f"\n{'~'*40}")
        print(f"TESTING NORMALIZATION: {norm_type.upper()}")
        print(f"{'~'*40}")
        
        # Stage 1: Apply normalization
        norm_dir = args.output_dir / f"01_normalized_{norm_type}"
        norm_dir.mkdir(parents=True, exist_ok=True)
        
        if norm_type == 'none':
            # For 'none', just copy the original file
            norm_file = norm_dir / f"embeddings_{args.dataset_name}_layer_{args.layer_num}_none.parquet"
            if not norm_file.exists():
                print(f"Copying original file (no normalization)")
                shutil.copy2(source_file, norm_file)
        else:
            # Apply normalization
            norm_file = norm_dir / f"embeddings_{args.dataset_name}_layer_{args.layer_num}_{norm_type}.parquet"
            if not norm_file.exists():
                print(f"[STAGE 1] Applying {norm_type} normalization...")
                result, success = run_command([
                    "python", "experiments/05_all_but_mean_variants.py",
                    "--source_path", str(source_file),
                    "--out_path", str(norm_file),
                    "--normalization_type", norm_type,
                    "--experiment_name", f"norm_comparison_{args.dataset_name}_{norm_type}",
                    "--layer_num", str(args.layer_num)
                ], allow_failure=False)  # Stop on normalization errors
        
        # Validate normalized file
        valid, msg = check_file_validity(norm_file)
        if not valid:
            print(f"❌ SKIPPED: {msg}")
            continue
        
        # Stage 2-5: Run the rest of the pipeline for this normalization
        for n_comp in pca_components:
            # Stage 2: PCA
            pca_dir = args.output_dir / f"02_pca_{norm_type}" / f"{n_comp}_components"
            pca_dir.mkdir(parents=True, exist_ok=True)
            pca_file = pca_dir / f"{args.dataset_name}_layer{args.layer_num}_{norm_type}.parquet"
            
            if not pca_file.exists():
                print(f"[STAGE 2] Running PCA (components: {n_comp})...")
                result, success = run_command([
                    "python", "experiments/01_pca.py",
                    "--source_path", str(norm_file),
                    "--out", str(pca_file),
                    "--n_components", str(n_comp),
                    "--experiment_name", f"pca_comparison_{norm_type}",
                    "--dataset", args.dataset_name,
                    "--layer_num", str(args.layer_num),
                    "--chunk_size_mode", "medium"  # Intelligent chunk sizing
                ], allow_failure=False)  # Stop on PCA errors
            
            # The PCA script actually creates files with pca_ and zca_ prefixes
            # So we need to check for those instead
            pca_variant_files = {
                "pca": pca_dir / f"pca_{pca_file.name}",
                "zca": pca_dir / f"zca_{pca_file.name}"
            }
            
            # Test both PCA variants
            for reduction_type in ["pca", "zca"]:
                pca_variant_file = pca_variant_files[reduction_type]
                
                # Check if variant file exists
                valid, msg = check_file_validity(pca_variant_file)
                if not valid:
                    print(f"⚠️  SKIPPED: {reduction_type.upper()} variant - {msg}")
                    continue
                
                # Check dimensions for slicing validation
                current_dims = get_data_dimensions(pca_variant_file)
                if current_dims is None:
                    print(f"⚠️  SKIPPED: Could not determine dimensions for {pca_variant_file}")
                    continue
                
                for skip_n in slice_values:
                    if skip_n >= current_dims:
                        print(f"⚠️  SKIPPED: Cannot slice {skip_n} components from {current_dims}-dimensional data")
                        continue
                    
                    # Stage 3: Slicing (if needed)
                    if skip_n > 0:
                        slice_dir = args.output_dir / f"03_sliced_{norm_type}" / f"skip_{skip_n}"
                        slice_dir.mkdir(parents=True, exist_ok=True)
                        slice_file = slice_dir / f"{pca_variant_file.stem}_skipped{skip_n}.parquet"
                        
                        if not slice_file.exists():
                            print(f"[STAGE 3] Slicing (skip: {skip_n})...")
                            result, success = run_command([
                                "python", "scripts/utilities/slice_parquet_vectors.py",
                                "--input_parquet", str(pca_variant_file),
                                "--output_parquet", str(slice_file),
                                "--skip_first_n", str(skip_n)
                            ], allow_failure=False)  # Stop on slicing errors
                        
                        input_for_umap = slice_file
                    else:
                        input_for_umap = pca_variant_file
                    
                    # Validate input for UMAP
                    valid, msg = check_file_validity(input_for_umap)
                    if not valid:
                        print(f"❌ SKIPPED: {msg}")
                        continue
                    
                    # Stage 4: UMAP
                    for n_neighbors in umap_neighbors:
                        for metric in umap_metrics:
                            umap_dir = args.output_dir / f"04_umap_{norm_type}" / f"layer_{args.layer_num}" / f"skip_{skip_n}" / reduction_type / f"components_{n_comp}" / f"neighbors_{n_neighbors}" / metric
                            umap_dir.mkdir(parents=True, exist_ok=True)
                            
                            umap_file = umap_dir / f"umap_{args.dataset_name}_l{args.layer_num}_s{skip_n}_{reduction_type}_c{n_comp}_n{n_neighbors}_{metric}_{norm_type}.parquet"
                            
                            if not umap_file.exists():
                                print(f"[STAGE 4] UMAP (neighbors: {n_neighbors}, metric: {metric})...")
                                result, success = run_command([
                                    "python", "experiments/02_umap.py",
                                    "--pca_path", str(input_for_umap),
                                    "--out_path", str(umap_file),
                                    "--n_neighbors", str(n_neighbors),
                                    "--metric", metric,
                                    "--dataset", args.dataset_name,
                                    "--experiment_name", f"umap_comparison_{norm_type}",
                                    "--reduction_type", reduction_type,
                                    "--layer_num", str(args.layer_num),
                                    "--input_n_components", str(n_comp - skip_n),
                                    "--skipped_n_components", str(skip_n)
                                ], allow_failure=False)  # Stop on UMAP errors
                            
                            # Validate UMAP file
                            valid, msg = check_file_validity(umap_file)
                            if not valid:
                                print(f"❌ SKIPPED: {msg}")
                                continue
                            
                            # Stage 5: K-Means
                            kmeans_dir = args.output_dir / f"05_kmeans_{norm_type}" / f"layer_{args.layer_num}" / f"skip_{skip_n}" / reduction_type / f"components_{n_comp}" / f"neighbors_{n_neighbors}" / metric
                            kmeans_dir.mkdir(parents=True, exist_ok=True)
                            kmeans_file = kmeans_dir / "kmeans_results.json"
                            
                            if not kmeans_file.exists():
                                print(f"[STAGE 5] K-Means clustering...")
                                
                                provenance = {
                                    "normalization_type": norm_type,
                                    "source_embedding": "full",
                                    "layer": args.layer_num,
                                    "pca_components_initial": n_comp,
                                    "dim_reduction_initial": reduction_type,
                                    "pca_components_sliced_off": skip_n,
                                    "umap_neighbors": n_neighbors,
                                    "umap_metric": metric
                                }
                                
                                result, success = run_command([
                                    "python", "experiments/03_kmeans.py",
                                    "--input_path", str(umap_file),
                                    "--out_path", str(kmeans_file),
                                    "--k", "3",
                                    "--dataset", args.dataset_name,
                                    "--experiment_name", f"kmeans_comparison_{norm_type}",
                                    "--provenance", json.dumps(provenance),
                                    "--reduction_type", "umap",
                                    "--layer_num", str(args.layer_num),
                                    "--input_n_components", str(n_comp - skip_n),
                                    "--umap_n_neighbors", str(n_neighbors),
                                    "--umap_metric", metric,
                                    "--umap_source_reduction_type", reduction_type,
                                    "--skipped_n_components", str(skip_n)
                                ], allow_failure=False)  # Stop on K-Means errors
    
    print(f"\n{'='*60}")
    print("NORMALIZATION COMPARISON COMPLETED!")
    print(f"Results saved in: {args.output_dir}")
    print("\nTo compare results, check the K-Means plots and metrics in:")
    for norm_type in normalization_types:
        kmeans_dir = args.output_dir / f"05_kmeans_{norm_type}"
        if kmeans_dir.exists():
            print(f"  - {norm_type}: {kmeans_dir}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main() 