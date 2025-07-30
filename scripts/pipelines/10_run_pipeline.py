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
from pathlib import Path
from typing import List, Dict, Any

import mlflow

# Configuration from PDF and previous pipelines
NORMALIZATION_METHODS = [
    "none",              # Baseline (sin normalización)
    "all_but_mean",      # Normalización global
    "per_type",          # Normalización por tipo
    "standard",          # Estandarización Z-score
    "cross_differences", # Análisis contrastivo
    "arithmetic_mean",   # Análisis contrastivo
    "geometric_median"   # Análisis contrastivo
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

def step_1_extract_embeddings(args, layer: int) -> Path:
    """Step 1: Extract embeddings for a specific layer."""
    print(f"\n📊 Step 1: Extracting embeddings for layer {layer}")
    
    output_path = args.output_dir / "embeddings" / f"embeddings_{args.dataset}_layer_{layer}.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        "python", "experiments/10_embeddings.py",
        "--out", str(output_path.parent),  # Directorio, no archivo
        "--dataset", args.dataset,
        "--layer_num", str(layer),
        "--experiment_name", f"{args.experiment_name}_embeddings"
    ]
    
    if not args.skip_embeddings:
        run_command(cmd)
    
    return output_path

def step_2_normalize_embeddings(args, embedding_path: Path, method: str, layer: int) -> Path:
    """Step 2: Normalize embeddings using specified method."""
    print(f"\n🔧 Step 2: Normalizing embeddings using {method}")
    
    output_path = args.output_dir / "normalized" / method / f"normalized_{args.dataset}_layer_{layer}_{method}.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        "python", "experiments/20_normalization.py",
        "--source_path", str(embedding_path),
        "--out_path", str(output_path),
        "--normalization_type", method,
        "--experiment_name", f"{args.experiment_name}_normalization",
        "--layer_num", str(layer)
    ]
    
    if not args.skip_normalization:
        run_command(cmd)
    
    return output_path

def step_3_measure_anisotropy(args, normalized_path: Path, method: str, layer: int) -> Dict[str, Any]:
    """Step 3: Measure anisotropy for the normalized embeddings."""
    print(f"\n📐 Step 3: Measuring anisotropy for {method}")
    
    output_dir = args.output_dir / "anisotropy" / method
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        "python", "experiments/70_anisotropy_analysis.py",
        "--input_path", str(normalized_path),
        "--output_dir", str(output_dir),
        "--dataset", args.dataset,
        "--layer_num", str(layer),
        "--experiment_name", f"{args.experiment_name}_anisotropy",
        "--embedding_type", "full"  # o "delta" según el caso
    ]
    
    if not args.skip_anisotropy:
        run_command(cmd)
    
    return {"method": method, "layer": layer, "output_dir": output_dir}

def step_4_dimensionality_reduction(args, normalized_path: Path, method: str, layer: int) -> Path:
    """Step 4: Apply dimensionality reduction (PCA + UMAP) with multiple configurations."""
    print(f"\n📉 Step 4: Dimensionality reduction for {method}")
    
    output_dir = args.output_dir / "dimensionality" / method
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Iterate through all PCA components
    for pca_comp in args.pca_components:
        pca_output_dir = output_dir / f"pca_{pca_comp}"
        pca_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Run PCA
        pca_cmd = [
            "python", "experiments/31_pca.py",
            "--source_path", str(normalized_path),
            "--out", str(pca_output_dir / f"pca_{args.dataset}_layer{layer}_{pca_comp}.parquet"),
            "--n_components", str(pca_comp),
            "--dataset", args.dataset,
            "--layer_num", str(layer),
            "--experiment_name", f"{args.experiment_name}_pca"
        ]
        run_command(pca_cmd)
        
        # For each reduction type (PCA/ZCA)
        for reduction_type in args.reduction_types:
            pca_file = pca_output_dir / f"{reduction_type}_{args.dataset}_layer{layer}_{pca_comp}.parquet"
            if not pca_file.exists():
                continue
                
            # For each slice configuration
            for slice_n in args.slice_n_values:
                if slice_n >= pca_comp:
                    continue  # Skip if slice_n >= available components
                    
                # Apply slicing if needed
                umap_input = pca_file
                if slice_n > 0:
                    slice_output = pca_output_dir / f"{reduction_type}_{args.dataset}_layer{layer}_{pca_comp}_slice{slice_n}.parquet"
                    slice_cmd = [
                        "python", "scripts/utilities/slice_parquet_vectors.py",
                        "--input_parquet", str(pca_file),
                        "--output_parquet", str(slice_output),
                        "--skip_first_n", str(slice_n)
                    ]
                    run_command(slice_cmd)
                    umap_input = slice_output
                
                # For each UMAP configuration
                metrics = args.umap_metrics_pca if reduction_type == "pca" else args.umap_metrics_zca
                for n_neighbors in args.umap_neighbors:
                    for min_dist in args.umap_min_dist:
                        for metric in metrics:
                            umap_output_dir = output_dir / f"umap_{pca_comp}_{reduction_type}_slice{slice_n}_n{n_neighbors}_d{min_dist}_{metric}"
                            umap_output_dir.mkdir(parents=True, exist_ok=True)
                            
                            umap_cmd = [
                                "python", "experiments/32_umap.py",
                                "--pca_path", str(umap_input),
                                "--out_path", str(umap_output_dir / f"umap_{args.dataset}_layer{layer}.parquet"),
                                "--n_neighbors", str(n_neighbors),
                                "--min_dist", str(min_dist),
                                "--metric", metric,
                                "--n_components", "2",
                                "--dataset", args.dataset,
                                "--experiment_name", f"{args.experiment_name}_umap",
                                "--reduction_type", reduction_type,
                                "--layer_num", str(layer),
                                "--input_n_components", str(pca_comp - slice_n),
                                "--skipped_n_components", str(slice_n)
                            ]
                            run_command(umap_cmd)
    
    return output_dir

def step_5_clustering(args, umap_path: Path, method: str, layer: int, config: str) -> Dict[str, Any]:
    """Step 5: Apply clustering analysis with multiple k values."""
    print(f"\n🎯 Step 5: Clustering analysis for {method} - {config}")
    
    output_dir = args.output_dir / "clustering" / method / config
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine k values based on configuration
    k_values = [2] if config == "EC" else args.kmeans_k_values
    
    results = []
    for k in k_values:
        k_output_dir = output_dir / f"k{k}"
        k_output_dir.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            "python", "experiments/40_clustering_analysis.py",
            "--input_path", str(umap_path),
            "--out_path", str(k_output_dir / "clustering_results.json"),
            "--k", str(k),
            "--dataset", args.dataset,
            "--layer_num", str(layer),
            "--experiment_name", f"{args.experiment_name}_clustering",
            "--reduction_type", "umap"
        ]
        
        run_command(cmd)
        results.append({"method": method, "config": config, "layer": layer, "k": k, "output_dir": k_output_dir})
    
    return results

def step_6_contrastive_analysis(args, normalized_path: Path, method: str, layer: int) -> Path:
    """Step 6: Apply contrastive analysis."""
    print(f"\n🔄 Step 6: Contrastive analysis for {method}")
    
    output_dir = args.output_dir / "contrastive" / method
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Crear directorio temporal con el archivo normalizado para el script
    temp_input_dir = output_dir / "temp_input"
    temp_input_dir.mkdir(parents=True, exist_ok=True)
    
    # Copiar el archivo normalizado al directorio temporal
    import shutil
    temp_file = temp_input_dir / f"embeddings_{args.dataset}_layer_{layer}.parquet"
    shutil.copy2(normalized_path, temp_file)
    
    cmd = [
        "python", "experiments/50_contrastive_analysis.py",
        "--input_dir", str(temp_input_dir),
        "--output_dir", str(output_dir),
        "--layers", str(layer),
        "--experiment_name", f"{args.experiment_name}_contrastive",
        "--methods", "arithmetic_mean", "geometric_median", "cross_differences"
    ]
    
    run_command(cmd)
    return output_dir

def step_7_probing(args, normalized_path: Path, method: str, layer: int) -> Dict[str, Any]:
    """Step 7: Apply decision tree probing."""
    print(f"\n🔍 Step 7: Decision tree probing for {method}")
    
    output_dir = args.output_dir / "probes" / method
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        "python", "experiments/60_decision_tree_probe.py",
        "--input_path", str(normalized_path),
        "--output_dir", str(output_dir),
        "--dataset_name", args.dataset,  # Corregido: dataset_name en lugar de dataset
        "--embedding_type", "full",  # Agregado: embedding_type requerido
        "--layer_num", str(layer),
        "--max_depth", str(args.probe_max_depth),
        "--experiment_name", f"{args.experiment_name}_probing"
    ]
    
    run_command(cmd)
    
    return {"method": method, "layer": layer, "output_dir": output_dir}

def step_8_statistical_validation(args, results_dir: Path) -> Dict[str, Any]:
    """Step 8: Statistical validation of results."""
    print(f"\n📊 Step 8: Statistical validation")
    
    output_dir = args.output_dir / "statistical"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        "python", "experiments/90_statistical_validation.py",
        "--results_dir", str(results_dir),
        "--output_dir", str(output_dir),
        "--dataset", args.dataset,
        "--experiment_name", f"{args.experiment_name}_statistical"
    ]
    
    run_command(cmd)
    return {"output_dir": output_dir}

def step_9_comparative_analysis(args) -> Dict[str, Any]:
    """Step 9: Comparative analysis between datasets."""
    print(f"\n🔬 Step 9: Comparative analysis")
    
    output_dir = args.output_dir / "comparative"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        "python", "experiments/95_comparative_analysis.py",
        "--dataset_name", args.dataset,  # Corregido: dataset_name en lugar de dataset
        "--base_data_dir", str(args.output_dir),  # Agregado: base_data_dir requerido
        "--output_dir", str(output_dir),
        "--experiment_name", f"{args.experiment_name}_comparative"
    ]
    
    run_command(cmd)
    return {"output_dir": output_dir}

def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    mlflow.set_experiment(args.experiment_name)
    
    with mlflow.start_run(run_name=f"unified_pipeline_{args.dataset}") as run:
        print(f"🚀 Starting Main Pipeline for {args.dataset}")
        
        # Log all parameters
        mlflow.log_params({
            "dataset": args.dataset,
            "layers": args.layers,
            "normalization_methods": NORMALIZATION_METHODS,
            "dataset_configurations": DATASET_CONFIGURATIONS,
            "pca_components": args.pca_components,
            "umap_neighbors": args.umap_neighbors,
            "umap_min_dist": args.umap_min_dist,
            "umap_metrics_pca": args.umap_metrics_pca,
            "umap_metrics_zca": args.umap_metrics_zca,
            "slice_n_values": args.slice_n_values,
            "kmeans_k_values": args.kmeans_k_values,
            "reduction_types": args.reduction_types,
            "probe_max_depth": args.probe_max_depth
        })
        
        results = {}
        
        for layer in args.layers:
            print(f"\n{'='*60}")
            print(f"PROCESSING LAYER {layer}")
            print(f"{'='*60}")
            
            # Step 1: Extract embeddings
            embedding_path = step_1_extract_embeddings(args, layer)
            
            for method in NORMALIZATION_METHODS:
                print(f"\n{'~'*40}")
                print(f"PROCESSING NORMALIZATION: {method.upper()}")
                print(f"{'~'*40}")
                
                # Step 2: Normalize embeddings
                normalized_path = step_2_normalize_embeddings(args, embedding_path, method, layer)
                
                # Step 3: Measure anisotropy
                anisotropy_results = step_3_measure_anisotropy(args, normalized_path, method, layer)
                
                # Step 4: Dimensionality reduction
                dim_reduction_dir = step_4_dimensionality_reduction(args, normalized_path, method, layer)
                
                # Step 5: Clustering for each configuration
                for config in DATASET_CONFIGURATIONS:
                    # Find UMAP files for clustering
                    umap_files = list(dim_reduction_dir.rglob("umap_*.parquet"))
                    for umap_file in umap_files:
                        clustering_results = step_5_clustering(args, umap_file, method, layer, config)
                
                # Step 6: Contrastive analysis
                contrastive_dir = step_6_contrastive_analysis(args, normalized_path, method, layer)
                
                # Step 7: Probing
                probing_results = step_7_probing(args, normalized_path, method, layer)
        
        # Step 8: Statistical validation
        statistical_results = step_8_statistical_validation(args, args.output_dir)
        
        # Step 9: Comparative analysis (only for both datasets)
        if args.dataset == "both":
            comparison_results = step_9_comparative_analysis(args)
        
        mlflow.log_artifact(str(args.output_dir))
        print(f"\n✅ Main Pipeline completed successfully!")

if __name__ == "__main__":
    main() 