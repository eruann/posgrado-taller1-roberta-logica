#!/usr/bin/env python3
"""
scripts/pipelines/10_run_pipeline.py - Pipeline Principal de Análisis Lógico
==========================================================================
Pipeline principal que ejecuta toda la metodología del PDF:
1. Extracción de embeddings
2. Normalización (7 métodos incluyendo contrastivos)
3. Medición de anisotropía por método
4. Configuración EC/ECN
5. Reducción dimensional (PCA + UMAP)
6. Clustering (K-means)
7. Probing (Decision trees)
8. Validación estadística
9. Comparación SNLI vs FOLIO

Usage:
    python scripts/pipelines/10_run_pipeline.py \
        --dataset snli \
        --layers 9 10 11 12 \
        --output_dir data/snli/unified_pipeline
"""

import argparse
import json
import subprocess
import sys
import uuid
from pathlib import Path
from typing import List, Dict, Any

import mlflow

# Configuration from PDF and previous pipelines
NORMALIZATION_METHODS = [
    "none",              # Baseline (sin normalización)
    "all_but_mean",      # Normalización global
    "per_type",          # Normalización por tipo
    "standard"           # Estandarización Z-score
]

DATASET_CONFIGURATIONS = ["EC", "ECN"]  # Entailment-Contradiction vs Entailment-Contradiction-Neutral

# Configuraciones de parámetros basadas en pipelines anteriores
PCA_COMPONENTS = [1, 5, 50]  # Múltiples configuraciones de PCA
UMAP_NEIGHBORS = [15, 100, 200]  # Múltiples valores de neighbors
UMAP_MIN_DIST_VALUES = [0.1, 0.5]  # Múltiples valores de min_dist
UMAP_METRICS_PCA = ["euclidean", "manhattan"]  # Métricas para PCA
UMAP_METRICS_ZCA = ["euclidean"]  # Métricas para ZCA
SLICE_N_VALUES = [0, 3, 5]  # Valores de slicing (0 = sin slicing)
KMEANS_K_VALUES = [2, 3]  # Valores de k para clustering

def generate_run_id() -> str:
    """Generate a unique run ID for this pipeline execution."""
    return f"id{uuid.uuid4().hex[:8]}"

def get_existing_normalizations(output_dir: Path, dataset: str, layer: int) -> List[str]:
    """Get list of normalization methods that already exist for a specific layer."""
    normalized_dir = output_dir / "normalized"
    if not normalized_dir.exists():
        return []
    
    existing_methods = []
    for method_dir in normalized_dir.iterdir():
        if method_dir.is_dir():
            expected_file = method_dir / f"normalized_{dataset}_layer_{layer}_{method_dir.name}.parquet"
            if expected_file.exists():
                existing_methods.append(method_dir.name)
    
    return existing_methods

def parse_args():
    parser = argparse.ArgumentParser(description="Main pipeline for logical analysis")
    parser.add_argument("--dataset", required=True, choices=["snli", "folio", "both"], 
                       help="Dataset to process")
    parser.add_argument("--layers", nargs='+', type=int, default=[9, 10, 11, 12],
                       help="Model layers to process")
    parser.add_argument("--output_dir", required=True, type=Path,
                       help="Base output directory")
    
    # Configuraciones de parámetros
    parser.add_argument("--pca_components", nargs='+', type=int, default=PCA_COMPONENTS,
                       help="PCA components to test")
    parser.add_argument("--umap_neighbors", nargs='+', type=int, default=UMAP_NEIGHBORS,
                       help="UMAP n_neighbors values to test")
    parser.add_argument("--umap_min_dist", nargs='+', type=float, default=UMAP_MIN_DIST_VALUES,
                       help="UMAP min_dist values to test")
    parser.add_argument("--umap_metrics_pca", nargs='+', default=UMAP_METRICS_PCA,
                       help="UMAP metrics for PCA")
    parser.add_argument("--umap_metrics_zca", nargs='+', default=UMAP_METRICS_ZCA,
                       help="UMAP metrics for ZCA")
    parser.add_argument("--slice_n_values", nargs='+', type=int, default=SLICE_N_VALUES,
                       help="Slice N values (0 = no slicing)")
    parser.add_argument("--kmeans_k_values", nargs='+', type=int, default=KMEANS_K_VALUES,
                       help="K-means k values to test")
    
    parser.add_argument("--probe_max_depth", type=int, default=4,
                       help="Maximum depth for decision tree probes")
    parser.add_argument("--skip_embeddings", action="store_true",
                       help="Skip embedding extraction if already done")
    parser.add_argument("--skip_normalization", action="store_true",
                       help="Skip normalization step")
    parser.add_argument("--skip_anisotropy", action="store_true",
                       help="Skip anisotropy measurement")
    parser.add_argument("--normalization_methods", nargs='+', default=NORMALIZATION_METHODS,
                       help="Specific normalization methods to run (default: all)")
    parser.add_argument("--experiment_name", default="unified_pipeline",
                       help="MLflow experiment name")
    
    # Configuraciones de reducción dimensional
    parser.add_argument("--reduction_types", nargs='+', default=["pca", "zca"],
                       help="Dimensionality reduction types to test")
    
    return parser.parse_args()

def run_command(cmd: List[str], cwd: Path = None) -> str:
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
        return process.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ COMMAND FAILED: {' '.join(map(str, cmd))}", file=sys.stderr)
        print(f"--- STDERR ---", file=sys.stderr)
        print(e.stderr, file=sys.stderr)
        print("--------------", file=sys.stderr)
        raise e

def step_1_extract_embeddings(args, layer: int, run_id: str) -> Path:
    """Step 1: Extract embeddings for a specific layer."""
    print(f"\n📊 Step 1: Extracting embeddings for layer {layer}")
    
    output_path = args.output_dir / "embeddings" / f"embeddings_{args.dataset}_layer_{layer}.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Log step information
    step_name = f"{run_id}_layer_{layer}_10_embeddings"
    mlflow.log_param("dataset", args.dataset)
    
    # Create simple provenance
    provenance = {"step": "embeddings", "layer": layer}
    
    cmd = [
        "python3", "experiments/10_embeddings.py",
        "--out", str(output_path.parent),  # Directorio, no archivo
        "--dataset", args.dataset,
        "--layer_num", str(layer),
        "--source_path", f"data/{args.dataset}/dataset",
        "--experiment_name", args.experiment_name,
        "--run_id", run_id,
        "--provenance", json.dumps(provenance)
    ]
    
    # Check if embeddings already exist
    if output_path.exists() and not args.skip_embeddings:
        print(f"⏩ Embeddings already exist: {output_path}")
        mlflow.log_metric("embeddings_existed", 1)
    elif not args.skip_embeddings:
        run_command(cmd)
        mlflow.log_metric("embeddings_generated", 1)
    else:
        mlflow.log_metric("embeddings_skipped", 1)
    
    return output_path

def step_2_normalize_embeddings(args, embedding_path: Path, method: str, layer: int, run_id: str) -> Path:
    """Step 2: Normalize embeddings using specified method."""
    print(f"\n🔧 Step 2: Normalizing embeddings using {method}")
    
    output_path = args.output_dir / "normalized" / method / f"normalized_{args.dataset}_layer_{layer}_{method}.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Log step information
    step_name = f"{run_id}_layer_{layer}_20_normalization_{method}"
    mlflow.log_param("dataset", args.dataset)
    
    # Create simple provenance
    provenance = {"step": "normalization", "method": method, "layer": layer}
    
    cmd = [
        "python3", "experiments/20_normalization.py",
        "--source_path", str(embedding_path),
        "--out_path", str(output_path),
        "--normalization_type", method,
        "--layer_num", str(layer),
        "--experiment_name", args.experiment_name,
        "--run_id", run_id,
        "--provenance", json.dumps(provenance)
    ]
    
    # Check if normalized embeddings already exist
    if output_path.exists():
        print(f"⏩ Normalized embeddings already exist: {output_path}")
        mlflow.log_metric("normalization_existed", 1)
    elif not args.skip_normalization:
        run_command(cmd)
        mlflow.log_metric("normalization_completed", 1)
    else:
        mlflow.log_metric("normalization_skipped", 1)
    
    return output_path

def step_3_measure_anisotropy(args, normalized_path: Path, method: str, layer: int, run_id: str) -> Dict[str, Any]:
    """Step 3: Measure anisotropy for the normalized embeddings."""
    print(f"\n📐 Step 3: Measuring anisotropy for {method}")
    
    output_dir = args.output_dir / "anisotropy" / method
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Log step information
    step_name = f"{run_id}_layer_{layer}_70_anisotropy_{method}"
    mlflow.log_param("dataset", args.dataset)
    
    # Create simple provenance
    provenance = {"step": "anisotropy", "method": method, "layer": layer}
    
    cmd = [
        "python3", "experiments/70_anisotropy_analysis.py",
        "--input_path", str(normalized_path),
        "--output_dir", str(output_dir),
        "--dataset", args.dataset,
        "--layer_num", str(layer),
        "--embedding_type", "full",  # o "delta" según el caso
        "--normalization_type", method,
        "--experiment_name", args.experiment_name,
        "--run_id", run_id,
        "--provenance", json.dumps(provenance)
    ]
    
    if not args.skip_anisotropy:
        run_command(cmd)
        mlflow.log_metric("anisotropy_measured", 1)
    else:
        mlflow.log_metric("anisotropy_skipped", 1)
    
    return {"method": method, "layer": layer, "output_dir": output_dir}

def step_4_dimensionality_reduction(args, normalized_path: Path, method: str, layer: int, run_id: str) -> List[Path]:
    """Step 4: Apply dimensionality reduction (PCA + UMAP) with multiple configurations."""
    print(f"\n📉 Step 4: Dimensionality reduction for {method}")
    
    output_dir = args.output_dir / "dimensionality" / method
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Log step information
    step_name = f"{run_id}_layer_{layer}_30_dimensionality_{method}"
    
    # Create simple provenance
    provenance = {"step": "dimensionality_reduction", "method": method, "layer": layer}
    
    # Iterate through all PCA components
    for pca_comp in args.pca_components:
            pca_output_dir = output_dir / f"pca_{pca_comp}"
            pca_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Run PCA
            pca_cmd = [
                "python3", "experiments/31_pca.py",
                "--source_path", str(normalized_path),
                "--out", str(pca_output_dir / f"pca_{args.dataset}_layer{layer}_{pca_comp}.parquet"),
                "--n_components", str(pca_comp),
                "--dataset", args.dataset,
                "--layer_num", str(layer),
                "--normalization_type", method,
                "--experiment_name", args.experiment_name,
                "--run_id", run_id,
                "--provenance", json.dumps(provenance)
            ]
            run_command(pca_cmd)
            
                        # For each reduction type (PCA/ZCA)
            for reduction_type in args.reduction_types:
                # Try different possible file patterns
                possible_files = [
                    pca_output_dir / f"{reduction_type}_{args.dataset}_layer{layer}_{pca_comp}.parquet",
                    pca_output_dir / f"{reduction_type}_{reduction_type}_{args.dataset}_layer{layer}_{pca_comp}.parquet",
                    pca_output_dir / f"{reduction_type}_pca_{args.dataset}_layer{layer}_{pca_comp}.parquet",
                    # Add more patterns to catch all possible file names
                    pca_output_dir / f"{reduction_type}_{args.dataset}_layer{layer}_{pca_comp}.parquet",
                    pca_output_dir / f"{reduction_type}_{args.dataset}_layer{layer}_{pca_comp}.parquet",
                    # Direct file names without prefixes
                    pca_output_dir / f"{reduction_type}_{args.dataset}_layer{layer}_{pca_comp}.parquet"
                ]
                
                pca_file = None
                print(f"🔍 Searching for {reduction_type} files in {pca_output_dir}")
                for file_path in possible_files:
                    print(f"  Checking: {file_path}")
                    if file_path.exists():
                        pca_file = file_path
                        print(f"✅ Found: {file_path}")
                        break
                    else:
                        print(f"  ❌ Not found: {file_path}")
                
                if not pca_file:
                    print(f"⚠️  No PCA file found for {reduction_type} in {pca_output_dir}")
                    print(f"Available files in directory:")
                    for f in pca_output_dir.glob("*.parquet"):
                        print(f"  - {f.name}")
                    continue
                    
                # For each slice configuration
                for slice_n in args.slice_n_values:
                    if slice_n >= pca_comp:
                        continue  # Skip if slice_n >= available components
                        
                    # Apply slicing if needed
                    umap_input = pca_file
                    if slice_n > 0:
                        # Apply slicing (implement if needed)
                        pass
                    
                    # For each UMAP configuration
                    for neighbors in args.umap_neighbors:
                        for min_dist in args.umap_min_dist:
                            metrics = args.umap_metrics_pca if reduction_type == "pca" else args.umap_metrics_zca
                            for metric in metrics:
                                umap_output = output_dir / f"umap_{args.dataset}_layer{layer}_{pca_comp}_{neighbors}_{min_dist}_{metric}.parquet"
                                
                                umap_cmd = [
                                    "python3", "experiments/32_umap.py",
                                    "--pca_path", str(umap_input),
                                    "--out_path", str(umap_output),
                                    "--n_neighbors", str(neighbors),
                                    "--min_dist", str(min_dist),
                                    "--metric", metric,
                                    "--n_components", "2",
                                    "--dataset", args.dataset,
                                    "--reduction_type", reduction_type,
                                    "--layer_num", str(layer),
                                    "--input_n_components", str(pca_comp),
                                    "--skipped_n_components", "0",
                                    "--normalization_type", method,
                                    "--experiment_name", args.experiment_name,
                                    "--run_id", run_id,
                                    "--provenance", json.dumps(provenance)
                                ]
                                run_command(umap_cmd)
        
    mlflow.log_metric("dimensionality_reduction_completed", 1)
    
    # Return list of generated UMAP files
    umap_files = list(output_dir.rglob("umap_*.parquet"))
    print(f"📊 Generated {len(umap_files)} UMAP files:")
    for umap_file in umap_files:
        print(f"  - {umap_file}")
    
    return umap_files

def step_5_clustering(args, umap_path: Path, method: str, layer: int, config: str, run_id: str) -> Dict[str, Any]:
    """Step 5: Apply clustering analysis with multiple k values."""
    print(f"\n🎯 Step 5: Clustering analysis for {method} - {config}")
    
    output_dir = args.output_dir / "clustering" / method / config
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Log step information
    step_name = f"{run_id}_layer_{layer}_40_clustering_{method}_{config}"
    
    # Create simple provenance
    provenance = {"step": "clustering", "method": method, "config": config, "layer": layer}
    
    # Determine k values based on configuration
    k_values = [2] if config == "EC" else args.kmeans_k_values
    
    results = []
    for k in k_values:
        k_output_dir = output_dir / f"k{k}"
        k_output_dir.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            "python3", "experiments/40_clustering_analysis.py",
            "--input_path", str(umap_path),
            "--out_path", str(k_output_dir / "clustering_results.json"),
            "--k", str(k),
            "--dataset", args.dataset,
            "--layer_num", str(layer),
            "--reduction_type", "umap",
            "--normalization_type", method,
            "--experiment_name", args.experiment_name,
            "--run_id", run_id,
            "--provenance", json.dumps(provenance)
        ]
        
        run_command(cmd)
        results.append({"method": method, "config": config, "layer": layer, "k": k, "output_dir": k_output_dir})
        
    mlflow.log_metric("clustering_completed", len(results))
    
    return results

def step_6_contrastive_analysis(args, normalized_path: Path, method: str, layer: int, run_id: str) -> Path:
    """Step 6: Apply contrastive analysis."""
    print(f"\n🔄 Step 6: Contrastive analysis for {method}")
    
    output_dir = args.output_dir / "contrastive" / method
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Log step information
    step_name = f"{run_id}_layer_{layer}_50_contrastive_{method}"
    
    # Create simple provenance
    provenance = {"step": "contrastive_analysis", "method": method, "layer": layer}
    
    # Crear directorio temporal con el archivo normalizado para el script
    temp_input_dir = output_dir / "temp_input"
    temp_input_dir.mkdir(parents=True, exist_ok=True)
    
    # Copiar el archivo normalizado al directorio temporal
    import shutil
    temp_file = temp_input_dir / f"embeddings_{args.dataset}_layer_{layer}.parquet"
    shutil.copy2(normalized_path, temp_file)
    
    cmd = [
        "python3", "experiments/50_contrastive_analysis.py",
        "--input_dir", str(temp_input_dir),
        "--output_dir", str(output_dir),
        "--layers", str(layer),
        "--methods", "arithmetic_mean", "geometric_median", "cross_differences",
        "--normalization_type", method,
        "--experiment_name", args.experiment_name,
        "--run_id", run_id,
        "--provenance", json.dumps(provenance)
    ]
    
    run_command(cmd)
    mlflow.log_metric("contrastive_analysis_completed", 1)
    
    return output_dir

def step_7_probing(args, normalized_path: Path, method: str, layer: int, run_id: str) -> Dict[str, Any]:
    """Step 7: Apply decision tree probing."""
    print(f"\n🔍 Step 7: Decision tree probing for {method}")
    
    output_dir = args.output_dir / "probes" / method
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Log step information
    step_name = f"{run_id}_layer_{layer}_60_probing_{method}"
    
    # Create simple provenance
    provenance = {"step": "probing", "method": method, "layer": layer}
    
    cmd = [
        "python3", "experiments/60_decision_tree_probe.py",
        "--input_path", str(normalized_path),
        "--output_dir", str(output_dir),
        "--dataset_name", args.dataset,  # Corregido: dataset_name en lugar de dataset
        "--normalization_type", method,
        "--embedding_type", "full",  # Agregado: embedding_type requerido
        "--layer_num", str(layer),
        "--max_depth", str(args.probe_max_depth),
        "--experiment_name", args.experiment_name,
        "--run_id", run_id,
        "--provenance", json.dumps(provenance)
    ]
    
    run_command(cmd)
    mlflow.log_metric("probing_completed", 1)
    
    return {"method": method, "layer": layer, "output_dir": output_dir}

def step_8_statistical_validation(args, results_dir: Path, run_id: str) -> Dict[str, Any]:
    """Step 8: Statistical validation of results."""
    print(f"\n📊 Step 8: Statistical validation")
    
    output_dir = args.output_dir / "statistical"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Log step information
    step_name = f"{run_id}_80_statistical_validation"
    
    # Create simple provenance
    provenance = {"step": "statistical_validation"}
    
    cmd = [
        "python3", "experiments/90_statistical_validation.py",
        "--results_dir", str(results_dir),
        "--output_dir", str(output_dir),
        "--dataset", args.dataset,
        "--experiment_name", args.experiment_name,
        "--run_id", run_id,
        "--provenance", json.dumps(provenance)
    ]
    
    run_command(cmd)
    mlflow.log_metric("statistical_validation_completed", 1)
    
    return {"output_dir": output_dir}



def main():
    print("🚀 Starting pipeline...")
    args = parse_args()
    print(f"📊 Dataset: {args.dataset}")
    print(f"📁 Output dir: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate unique run ID for this pipeline execution
    run_id = generate_run_id()
    print(f"🆔 Generated Run ID: {run_id}")
    
    # Configure experiment based on experiment_name parameter or dataset
    experiment_name = args.experiment_name if args.experiment_name else f"logical_analysis_{args.dataset}"
    mlflow.set_experiment(experiment_name)
    
    print(f"🚀 Starting Main Pipeline for {args.dataset}")
    print(f"📊 MLflow Experiment: {experiment_name}")
    print(f"🆔 Common Run ID: {run_id}")
    
    # Log all parameters at experiment level
    mlflow.log_params({
        "run_id": run_id,
        "dataset": args.dataset,
        "layers": str(args.layers),
        "normalization_methods": str(args.normalization_methods),
        "dataset_configurations": str(DATASET_CONFIGURATIONS),
        "pca_components": str(args.pca_components),
        "umap_neighbors": str(args.umap_neighbors),
        "umap_min_dist": str(args.umap_min_dist),
        "umap_metrics_pca": str(args.umap_metrics_pca),
        "umap_metrics_zca": str(args.umap_metrics_zca),
        "slice_n_values": str(args.slice_n_values),
        "kmeans_k_values": str(args.kmeans_k_values),
        "reduction_types": str(args.reduction_types),
        "probe_max_depth": args.probe_max_depth,
        "skip_embeddings": args.skip_embeddings,
        "skip_normalization": args.skip_normalization,
        "skip_anisotropy": args.skip_anisotropy
    })
    
    results = {}
    
    for layer in args.layers:
        print(f"\n{'='*60}")
        print(f"PROCESSING LAYER {layer}")
        print(f"{'='*60}")
        
        # Step 1: Extract embeddings
        embedding_path = step_1_extract_embeddings(args, layer, run_id)
        
        # Get existing normalizations for this layer
        existing_normalizations = get_existing_normalizations(args.output_dir, args.dataset, layer)
        print(f"📋 Existing normalizations for layer {layer}: {existing_normalizations}")
        
        # Process only missing normalizations
        missing_normalizations = [m for m in args.normalization_methods if m not in existing_normalizations]
        if missing_normalizations:
            print(f"🆕 Missing normalizations for layer {layer}: {missing_normalizations}")
        else:
            print(f"✅ All normalizations already exist for layer {layer}")
        
        for method in args.normalization_methods:
            if method in existing_normalizations:
                print(f"\n{'~'*40}")
                print(f"SKIPPING NORMALIZATION: {method.upper()} (already exists)")
                print(f"{'~'*40}")
                # Create the path for existing file
                normalized_path = args.output_dir / "normalized" / method / f"normalized_{args.dataset}_layer_{layer}_{method}.parquet"
            else:
                print(f"\n{'~'*40}")
                print(f"PROCESSING NORMALIZATION: {method.upper()}")
                print(f"{'~'*40}")
                
                # Step 2: Normalize embeddings
                normalized_path = step_2_normalize_embeddings(args, embedding_path, method, layer, run_id)
            
            # Step 3: Measure anisotropy
            anisotropy_results = step_3_measure_anisotropy(args, normalized_path, method, layer, run_id)
            
            # Step 4: Dimensionality reduction
            umap_files = step_4_dimensionality_reduction(args, normalized_path, method, layer, run_id)
            
            # Step 5: Clustering for each configuration
            for config in DATASET_CONFIGURATIONS:
                # Use the specific UMAP files generated in step 4
                for umap_file in umap_files:
                    clustering_results = step_5_clustering(args, umap_file, method, layer, config, run_id)
            
            # Step 6: Contrastive analysis
            contrastive_dir = step_6_contrastive_analysis(args, normalized_path, method, layer, run_id)
            
            # Step 7: Probing
            probing_results = step_7_probing(args, normalized_path, method, layer, run_id)
    
    # Step 8: Statistical validation
    statistical_results = step_8_statistical_validation(args, args.output_dir, run_id)
    
    # Log final artifacts
    print(f"\n✅ Main Pipeline completed successfully!")
    print(f"📊 Results saved to: {args.output_dir}")
    print(f"🔗 MLflow UI: mlflow ui")
    print(f"🆔 Run ID: {run_id}")

if __name__ == "__main__":
    main() 