#!/usr/bin/env python
"""
Orchestrates a comprehensive anisotropy analysis pipeline for SNLI embeddings.

This pipeline performs the following steps:
1.  Generates base embeddings (full and delta) from a filtered SNLI dataset
    for specified layers (e.g., 9-12).
2.  Calculates baseline S_intra and S_inter anisotropy for these embeddings.
3.  Applies various normalization techniques ('per_type', 'all_but_mean',
    'standard') to the base embeddings.
4.  Calculates anisotropy for each set of normalized embeddings to measure
    the impact of normalization.
5.  Runs a contrastive analysis using the 'cross_differences' method.
6.  Calculates S_inter anisotropy for the resulting contrastive vectors.
7.  Logs all parameters, metrics, and configurations to a single MLflow
    experiment named 'anisotropy_analysis'.
8.  Generates a final summary CSV with all anisotropy results.

Usage:
    python scripts/pipelines/run_anisotropy_pipeline.py \\
        --layers 9 10 11 12 \\
        --output_dir data/snli/anisotropy_analysis
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import mlflow
import pandas as pd

# --- Configuration ---
SOURCE_DATA_PATH = "data/snli/dataset/snli_filtered"
EXPERIMENT_NAME = "anisotropy_analysis"
LAYERS_TO_PROCESS = [9, 10, 11, 12]
NORMALIZATION_TYPES = ['per_type', 'all_but_mean', 'standard']

def parse_args():
    """Parses command-line arguments."""
    p = argparse.ArgumentParser(description="Run the full anisotropy analysis pipeline.")
    p.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/snli/anisotropy_analysis"),
        help="Base directory to store all generated files and results."
    )
    p.add_argument(
        "--layers",
        nargs='+',
        type=int,
        default=LAYERS_TO_PROCESS,
        help="List of model layers to process."
    )
    p.add_argument(
        "--skip_embeddings",
        action="store_true",
        help="Skip the initial embedding generation step."
    )
    p.add_argument(
        "--skip_base_anisotropy",
        action="store_true",
        help="Skip the baseline anisotropy calculation steps."
    )
    p.add_argument(
        "--skip_normalization",
        action="store_true",
        help="Skip normalization and its subsequent anisotropy calculation steps."
    )
    return p.parse_args()

def run_command(cmd: list, cwd: Path = None) -> str:
    """Executes a command and returns its stdout, raising an error on failure."""
    print(f"\n▶️ RUNNING: {' '.join(map(str, cmd))}")
    try:
        process = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            cwd=cwd,
        )
        print("✅ Command successful.")
        # Print stdout for logging, but only return the part needed (e.g., JSON)
        print("--- CAPTURED STDOUT ---")
        print(process.stdout)
        print("-----------------------")
        return process.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ COMMAND FAILED: {' '.join(map(str, cmd))}", file=sys.stderr)
        print(f"--- STDERR ---", file=sys.stderr)
        print(e.stderr, file=sys.stderr)
        print("--------------", file=sys.stderr)
        raise e

def calculate_anisotropy(file_path: Path, emb_type: str, calculations: list) -> dict:
    """Invokes the anisotropy calculation script and returns the results."""
    print(f"  -> Calculating anisotropy for: {file_path.name}...")
    cmd = [
        "python", "scripts/utilities/calculate_anisotropy.py",
        "--input_path", file_path,
        "--embedding_type", emb_type,
        "--calculations", *calculations
    ]
    output = run_command(cmd)
    # Extract the JSON part from the output
    try:
        json_str = output.split("--- Results ---")[1].split("---------------")[0].strip()
        return json.loads(json_str)
    except (IndexError, json.JSONDecodeError) as e:
        print(f"Could not parse JSON from anisotropy script output: {e}", file=sys.stderr)
        return {}

def main():
    """Main pipeline execution function."""
    args = parse_args()
    
    # --- Setup Directories ---
    base_out_dir = args.output_dir
    full_emb_dir = base_out_dir / "embeddings"
    delta_emb_dir = base_out_dir / "difference_embeddings"
    norm_dir = base_out_dir / "normalized"
    contrastive_dir = base_out_dir / "contrastive_analysis"
    
    for d in [full_emb_dir, delta_emb_dir, norm_dir, contrastive_dir]:
        d.mkdir(parents=True, exist_ok=True)
        
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    summary_results = []

    # --- 1. Generate Base Embeddings ---
    if not args.skip_embeddings:
        print("\n--- STEP 1: Generating Base Embeddings ---")
        with mlflow.start_run(run_name="00_generate_base_embeddings") as parent_run:
            mlflow.log_params({"layers": args.layers, "output_dir": str(base_out_dir)})
            cmd = [
                "python", "experiments/00_embeddings_snli.py",
                "--source_path", SOURCE_DATA_PATH,
                "--out", base_out_dir,
                "--dataset", "snli",
                "--layer_num", f"{min(args.layers)}-{max(args.layers)}",
                "--device", "cuda"
            ]
            run_command(cmd)
            # The script saves full and delta embeddings in the same base directory.
            # We need to move them to their dedicated subfolders.
            for layer in args.layers:
                # Move full embeddings
                src_full = base_out_dir / f"embeddings_snli_layer_{layer}.parquet"
                dest_full = full_emb_dir / f"embeddings_snli_layer_{layer}.parquet"
                if src_full.exists():
                    src_full.rename(dest_full)
                
                # Move delta embeddings
                src_delta = base_out_dir / f"embeddings_snli_layer_{layer}_delta.parquet"
                dest_delta = delta_emb_dir / f"embeddings_snli_layer_{layer}_delta.parquet"
                if src_delta.exists():
                    src_delta.rename(dest_delta)
    else:
        print("⏩ SKIPPING Embedding Generation (Step 1)")

    # --- Loop through layers for analysis ---
    for layer in args.layers:
        print(f"\n--- Analyzing Layer {layer} ---")

        # --- 2. Calculate Base Anisotropy ---
        if not args.skip_base_anisotropy:
            print(f"\n--- STEP 2: Baseline Anisotropy for Layer {layer} ---")
            # For FULL embeddings
            full_emb_path = full_emb_dir / f"embeddings_snli_layer_{layer}.parquet"
            if full_emb_path.exists():
                with mlflow.start_run(run_name=f"anisotropy_base_full_layer_{layer}", nested=True):
                    mlflow.log_params({"layer": layer, "type": "full", "normalization": "none"})
                    metrics = calculate_anisotropy(full_emb_path, "full", ["s_intra", "s_inter"])
                    mlflow.log_metrics(metrics)
                    summary_results.append({"layer": layer, "type": "full", "normalization": "none", **metrics})
            
            # For DELTA embeddings
            delta_emb_path = delta_emb_dir / f"embeddings_snli_layer_{layer}_delta.parquet"
            if delta_emb_path.exists():
                with mlflow.start_run(run_name=f"anisotropy_base_delta_layer_{layer}", nested=True):
                    mlflow.log_params({"layer": layer, "type": "delta", "normalization": "none"})
                    metrics = calculate_anisotropy(delta_emb_path, "delta", ["s_intra", "s_inter"])
                    mlflow.log_metrics(metrics)
                    summary_results.append({"layer": layer, "type": "delta", "normalization": "none", **metrics})
        else:
            print("⏩ SKIPPING Baseline Anisotropy Calculation (Step 2)")

        # --- 3 & 4. Normalization and Normalized Anisotropy ---
        if not args.skip_normalization:
            print(f"\n--- STEPS 3 & 4: Normalization and Anisotropy for Layer {layer} ---")
            for emb_type, source_dir in [("full", full_emb_dir), ("delta", delta_emb_dir)]:
                source_file = source_dir / f"embeddings_snli_layer_{layer}.parquet"
                if not source_file.exists():
                    continue

                for norm_type in NORMALIZATION_TYPES:
                    run_name = f"anisotropy_norm_{emb_type}_{norm_type}_layer_{layer}"
                    with mlflow.start_run(run_name=run_name, nested=True):
                        mlflow.log_params({"layer": layer, "type": emb_type, "normalization": norm_type})
                        
                        # Step 3: Run normalization
                        out_norm_dir = norm_dir / emb_type / norm_type
                        out_norm_dir.mkdir(parents=True, exist_ok=True)
                        norm_file = out_norm_dir / f"layer_{layer}.parquet"
                        
                        if not norm_file.exists():
                            cmd_norm = [
                                "python", "experiments/05_all_but_mean_variants.py",
                                "--source_path", source_file,
                                "--out_path", norm_file,
                                "--normalization_type", norm_type,
                                "--experiment_name", EXPERIMENT_NAME, # for internal logging
                                "--layer_num", str(layer)
                            ]
                            run_command(cmd_norm)
                        else:
                            print(f"⏩ SKIPPING normalization, file exists: {norm_file}")
                            
                        mlflow.log_artifact(str(norm_file))
                        
                        # Step 4: Calculate anisotropy on normalized file
                        print(f"  -> Calculating anisotropy for: {norm_file.name}...")
                        metrics = calculate_anisotropy(norm_file, "delta", ["s_intra", "s_inter"])
                        mlflow.log_metrics(metrics)
                        summary_results.append({"layer": layer, "type": emb_type, "normalization": norm_type, **metrics})
        else:
            print("⏩ SKIPPING Normalization and Anisotropy (Steps 3 & 4)")

    # --- 5 & 6. Contrastive Analysis and Anisotropy ---
    print("\n--- STEPS 5 & 6: Contrastive Analysis and Anisotropy ---")
    
    # Check if the last expected file from this step exists, to see if we can skip it.
    last_contrastive_file = contrastive_dir / f"contrastive_cross_differences_ecn_layer_{max(args.layers)}.parquet"
    if not last_contrastive_file.exists():
        with mlflow.start_run(run_name="02_contrastive_analysis", nested=True):
            cmd_contrastive = [
                "python", "experiments/02_contrastive_analysis.py",
                "--input_dir", str(full_emb_dir), # Uses full embeddings
                "--output_dir", str(contrastive_dir),
                "--layers", *map(str, args.layers),
                "--methods", "cross_differences",
                "--experiment_name", EXPERIMENT_NAME
            ]
            run_command(cmd_contrastive)
    else:
        print(f"⏩ SKIPPING contrastive analysis, output file exists: {last_contrastive_file}")

    # Calculate anisotropy for the outputs
    for layer in args.layers:
        for mode in ["ec", "ecn"]:
            contrastive_file = contrastive_dir / f"contrastive_cross_differences_{mode}_layer_{layer}.parquet"
            if contrastive_file.exists():
                run_name = f"anisotropy_contrastive_cross_diff_{mode}_layer_{layer}"
                with mlflow.start_run(run_name=run_name, nested=True):
                    mlflow.log_params({"layer": layer, "type": "contrastive", "method": "cross_differences", "mode": mode})
                    # S_intra makes no sense for cross-differences
                    metrics = calculate_anisotropy(contrastive_file, "contrastive", ["s_inter"])
                    mlflow.log_metrics(metrics)
                    summary_results.append({"layer": layer, "type": f"contrastive_{mode}", "normalization": "cross_differences", **metrics})

    # --- Final Summary ---
    print("\n--- PIPELINE COMPLETED ---")
    if summary_results:
        summary_df = pd.DataFrame(summary_results)
        summary_csv_path = base_out_dir / "anisotropy_summary.csv"
        summary_df.to_csv(summary_csv_path, index=False)
        
        print("Final Anisotropy Summary:")
        print(summary_df.to_string())
        print(f"\nSummary saved to: {summary_csv_path}")
        with mlflow.start_run(run_name="pipeline_summary", nested=True):
             mlflow.log_artifact(str(summary_csv_path))

if __name__ == "__main__":
    main() 