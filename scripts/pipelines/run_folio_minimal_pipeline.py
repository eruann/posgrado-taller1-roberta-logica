#!/usr/bin/env python
"""
Runs a minimal, targeted analysis pipeline on the FOLIO dataset.

This pipeline is designed to be a faster, more focused version of the
comprehensive SNLI analysis, targeting specific combinations of data views,
layers, and normalization techniques.

The process is as follows:
1.  Pre-processes the raw FOLIO data to create cleaned 'imbalanced' and
    'balanced' versions.
2.  Generates embeddings (full and delta) for specified layers (9, 11, 12)
    on both dataset versions.
3.  Runs a specific list of experiments:
    -   FULL embeddings (layers 9, 12) with 'none' and 'ABTT' normalization.
    -   DELTA embeddings (layers 9, 11) with 'none' normalization.
    -   CROSS-DIFFERENCE embeddings (layer 9) with 'none' normalization.
4.  For each experiment, it runs dimensionality reduction and clustering via
    `run_normalization_comparison_fixed.py` or contrastive analysis via
    `02_contrastive_analysis.py`.
5.  Calculates anisotropy metrics (S_intra, S_inter) on the generated
    embedding files.
6.  Logs all parameters and metrics to a dedicated MLflow experiment and
    compiles a final summary CSV.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import mlflow
import pandas as pd

# --- Global Configuration ---
EXPERIMENT_NAME = "folio_minimal_analysis"

# Define the specific set of experiments to run
# ABTT = All But The Mean. In our case, this corresponds to the 'all_but_mean'
# normalization strategy, which is then sliced by the comparison script.
EXPERIMENT_CONFIGS = [
    # View, Layer, Normalization, Analysis Script, Dataset Type
    {"view": "full", "layer": 9, "norm": "none", "dataset": "balanced"},
    {"view": "full", "layer": 12, "norm": "none", "dataset": "balanced"},
    {"view": "full", "layer": 9, "norm": "all_but_mean", "dataset": "balanced"},
    {"view": "full", "layer": 12, "norm": "all_but_mean", "dataset": "balanced"},
    {"view": "delta", "layer": 9, "norm": "none", "dataset": "balanced"},
    {"view": "delta", "layer": 11, "norm": "none", "dataset": "balanced"},
    {"view": "cross_diff", "layer": 9, "norm": "none", "dataset": "balanced"},
]

def parse_args():
    p = argparse.ArgumentParser(description="Run the minimal FOLIO analysis pipeline.")
    p.add_argument("--folio_raw_dir", type=Path, default=Path("data/folio/dataset"), help="Dir with raw FOLIO data.")
    p.add_argument("--output_dir", type=Path, default=Path("data/folio/minimal_analysis"), help="Base output directory.")
    p.add_argument("--force_preprocessing", action="store_true", help="Force re-running the preprocessing step.")
    p.add_argument("--force_embeddings", action="store_true", help="Force re-running the embedding generation step.")
    return p.parse_args()

def run_command(cmd: list, cwd: Path = None) -> str:
    """Executes a command and returns its stdout."""
    cmd_str = " ".join(map(str, cmd))
    print(f"▶️ RUNNING: {cmd_str}")
    try:
        process = subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=cwd)
        print(f"✅ COMMAND SUCCEEDED: {cmd_str}")
        if process.stdout:
            print(f"--- STDOUT ---\n{process.stdout.strip()}\n--------------")
        return process.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ COMMAND FAILED: {cmd_str}")
        print("--- STDOUT ---")
        print(e.stdout)
        print("--- STDERR ---")
        print(e.stderr)
        raise e

def calculate_anisotropy(file_path: Path, emb_type: str, calculations: list) -> dict:
    """Invokes the anisotropy calculation script and returns results."""
    if not file_path.exists():
        print(f"⏩ SKIPPING anisotropy, file not found: {file_path}")
        return {}
    
    print(f"  -> Calculating anisotropy for: {file_path.name}...")
    cmd = [
        "python", "scripts/utilities/calculate_anisotropy.py",
        "--input_path", file_path,
        "--embedding_type", emb_type,
        "--calculations", *calculations,
        "--sample_size", "25000" # As requested
    ]
    output = run_command(cmd)
    try:
        json_str = output.split("--- Results ---")[1].split("---------------")[0].strip()
        return json.loads(json_str)
    except (IndexError, json.JSONDecodeError) as e:
        print(f"Could not parse JSON from anisotropy script output: {e}", file=sys.stderr)
        return {}

def main():
    args = parse_args()
    
    # --- 0. Setup Directories ---
    processed_data_dir = args.output_dir / "processed_data"
    emb_dir = args.output_dir / "embeddings"
    norm_dir = args.output_dir / "normalized"
    contrastive_dir = args.output_dir / "contrastive"
    results_dir = args.output_dir / "results"
    for d in [processed_data_dir, emb_dir, norm_dir, contrastive_dir, results_dir]:
        d.mkdir(parents=True, exist_ok=True)
        
    mlflow.set_experiment(EXPERIMENT_NAME)
    summary_results = []

    # --- 1. Pre-process FOLIO dataset ---
    print("\n--- STEP 1: Pre-processing FOLIO Dataset ---")
    imbalanced_cleaned_path = processed_data_dir / "folio_imbalanced_cleaned"
    balanced_cleaned_path = processed_data_dir / "folio_balanced_cleaned"
    
    if not args.force_preprocessing and imbalanced_cleaned_path.exists() and balanced_cleaned_path.exists():
        print("⏩ SKIPPING preprocessing, output files already exist.")
    else:
        cmd_preprocess = [
            "python", "scripts/pipelines/preprocess_folio.py",
            "--input_dir", args.folio_raw_dir,
            "--output_dir", processed_data_dir,
        ]
        run_command(cmd_preprocess)

    # --- 2. Generate required embeddings ---
    print("\n--- STEP 2: Generating Embeddings ---")
    layers_to_gen = sorted(list(set(c['layer'] for c in EXPERIMENT_CONFIGS)))
    for dataset_type in ["imbalanced", "balanced"]:
        source_path = processed_data_dir / f"folio_{dataset_type}_cleaned"
        out_emb_dir = emb_dir / dataset_type
        out_emb_dir.mkdir(parents=True, exist_ok=True)

        # Check for a representative file from the last layer to see if work is done
        last_layer_file = out_emb_dir / f"embeddings_folio_layer_{max(layers_to_gen)}.parquet"

        if not args.force_embeddings and last_layer_file.exists():
            print(f"⏩ SKIPPING embedding generation for '{dataset_type}', files exist.")
        else:
            cmd_embed = [
                "python", "experiments/00_embeddings_snli.py",
                "--source_path", str(source_path),
                "--out", str(out_emb_dir),
                "--dataset", "folio",
                "--layer_num", f"{min(layers_to_gen)}-{max(layers_to_gen)}",
                "--device", "cuda"
            ]
            run_command(cmd_embed)

    # --- 3, 4, 5. Run Targeted Experiments and Analyses ---
    print("\n--- STEPS 3, 4, 5: Running Targeted Experiments ---")
    for config in EXPERIMENT_CONFIGS:
        view, layer, norm, dataset = config["view"], config["layer"], config["norm"], config["dataset"]
        run_name = f"{dataset}_{view}_layer{layer}_{norm}"
        
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(config)
            
            # Define input and output paths
            source_emb_dir = emb_dir / dataset
            
            # --- Cross-Difference Path ---
            if view == "cross_diff":
                print(f"\nProcessing: {dataset} / cross_diff / layer {layer}")
                analysis_out_dir = contrastive_dir / dataset
                analysis_out_dir.mkdir(exist_ok=True)
                
                # We need the full embeddings as input for this script
                full_emb_path = source_emb_dir / f"embeddings_folio_layer_{layer}.parquet"
                
                cmd_contrastive = [
                    "python", "experiments/02_contrastive_analysis.py",
                    "--input_dir", str(full_emb_path.parent),
                    "--output_dir", str(analysis_out_dir),
                    "--layers", str(layer),
                    "--methods", "cross_differences",
                    "--experiment_name", EXPERIMENT_NAME
                ]
                run_command(cmd_contrastive)
                
                # --- Decision Tree Probe for Cross-Difference ---
                cd_file = analysis_out_dir / f"contrastive_cross_differences_ec_layer_{layer}.parquet"
                if cd_file.exists():
                    tree_out_dir = analysis_out_dir / "decision_tree_probe"
                    tree_out_dir.mkdir(parents=True, exist_ok=True)
                    cmd_tree = [
                        "python", "scripts/run_decision_tree_probe.py",
                        "--input_path", cd_file,
                        "--output_dir", tree_out_dir,
                        "--experiment_name", EXPERIMENT_NAME,
                        "--dataset_name", "folio",
                        "--embedding_type", view,
                        "--layer_num", str(layer)
                    ]
                    run_command(cmd_tree)
                else:
                    print(f"⏩ SKIPPING Decision Tree Probe, no cross-difference file found: {cd_file}")

                # Anisotropy for cross-diff (S_inter only)
                metrics = calculate_anisotropy(cd_file, "contrastive", ["s_inter"])
                mlflow.log_metrics({f"anisotropy_{k}": v for k, v in metrics.items()})
                summary_results.append({"run_name": run_name, **config, **metrics})
                continue # End of this experiment config

            # --- FULL and DELTA Path ---
            emb_suffix = "" if view == "full" else "_delta"
            source_file = source_emb_dir / f"embeddings_folio_layer_{layer}{emb_suffix}.parquet"

            if norm != "none":
                print(f"\nProcessing: {dataset} / {view} / layer {layer} / norm {norm}")
                # Apply normalization
                out_norm_dir = norm_dir / dataset / view / norm
                out_norm_dir.mkdir(parents=True, exist_ok=True)
                norm_file = out_norm_dir / f"layer_{layer}.parquet"
                
                cmd_norm = [
                    "python", "experiments/05_all_but_mean_variants.py",
                    "--source_path", source_file, "--out_path", norm_file,
                    "--normalization_type", norm, "--layer_num", str(layer),
                    "--experiment_name", EXPERIMENT_NAME,
                ]
                run_command(cmd_norm)
                input_for_analysis = norm_file
            else:
                print(f"\nProcessing: {dataset} / {view} / layer {layer} / no normalization")
                input_for_analysis = source_file
            
            # Run clustering and get purity/NMI via the comparison fixed script
            analysis_results_dir = results_dir / dataset / view / f"layer_{layer}" / norm
            analysis_results_dir.mkdir(parents=True, exist_ok=True)

            cmd_analysis = [
                "python", "scripts/run_normalization_comparison_fixed.py",
                "--dataset_name", "folio",
                "--embedding_type", view,
                f"--{view}_embeddings_dir", str(input_for_analysis.parent),
                "--output_dir", str(analysis_results_dir),
                "--layer_num", str(layer),
                "--normalization_types", norm, # Pass the correct norm type
                "--filter_to_ec", # Always filter to E/C for clustering
                "--clean_output",
                "--experiment_name", EXPERIMENT_NAME
            ]
            run_command(cmd_analysis)
            
            # --- Decision Tree Probe ---
            # Find the PCA output file to use as input for the tree probe
            pca_output_dir = analysis_results_dir / f"02_pca_{view}_{norm}" / "50_components"
            pca_files = list(pca_output_dir.glob("*.parquet"))
            
            if pca_files:
                pca_file = pca_files[0]
                tree_out_dir = analysis_results_dir / "03_decision_tree_probe"
                tree_out_dir.mkdir(parents=True, exist_ok=True)
                
                cmd_tree = [
                    "python", "scripts/run_decision_tree_probe.py",
                    "--input_path", pca_file,
                    "--output_dir", tree_out_dir,
                    "--experiment_name", EXPERIMENT_NAME,
                    "--dataset_name", "folio",
                    "--embedding_type", view,
                    "--layer_num", str(layer)
                ]
                run_command(cmd_tree)
            else:
                print(f"⏩ SKIPPING Decision Tree Probe, no PCA file found in {pca_output_dir}")

            # Anisotropy for this view
            calc_list = ["s_inter"]
            if view == "full" or (view == "delta" and norm == "none"):
                calc_list.append("s_intra")
            
            metrics = calculate_anisotropy(input_for_analysis, view, calc_list)
            mlflow.log_metrics({f"anisotropy_{k}": v for k, v in metrics.items()})
            summary_results.append({"run_name": run_name, **config, **metrics})


    # --- Final Summary ---
    print("\n--- PIPELINE COMPLETED ---")
    if summary_results:
        summary_df = pd.DataFrame(summary_results)
        summary_csv_path = args.output_dir / "folio_minimal_analysis_summary.csv"
        summary_df.to_csv(summary_csv_path, index=False)
        
        print("Final Analysis Summary:")
        print(summary_df.to_string())
        print(f"\nSummary saved to: {summary_csv_path}")
        mlflow.log_artifact(str(summary_csv_path))

if __name__ == "__main__":
    main() 