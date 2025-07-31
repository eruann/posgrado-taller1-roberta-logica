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
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import mlflow
import mlflow.exceptions

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

def get_or_create_run_id(output_dir: Path, experiment_name: str = None) -> str:
    """Get existing run_id from checkpoint file or create new one."""
    # Use experiment_name in checkpoint filename if provided
    if experiment_name:
        checkpoint_file = output_dir / f".pipeline_checkpoint_{experiment_name}"
    else:
        checkpoint_file = output_dir / ".pipeline_checkpoint"
    
    if checkpoint_file.exists():
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
                existing_run_id = checkpoint_data.get('run_id')
                if existing_run_id:
                    print(f"🔄 Found existing checkpoint: {existing_run_id}")
                    print(f"📁 Continuing from: {checkpoint_file}")
                    return existing_run_id
        except (json.JSONDecodeError, KeyError) as e:
            print(f"⚠️  Corrupted checkpoint file, creating new run_id: {e}")
    
    # Create new run_id and save checkpoint
    new_run_id = generate_run_id()
    checkpoint_data = {
        'run_id': new_run_id,
        'created_at': str(datetime.now()),
        'dataset': None,  # Will be set in main()
        'layers': None,   # Will be set in main()
        'experiment_name': experiment_name
    }
    
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)
    
    print(f"🆔 Created new checkpoint: {new_run_id}")
    return new_run_id

def update_checkpoint(output_dir: Path, experiment_name: str = None, **kwargs):
    """Update checkpoint file with additional information."""
    # Use experiment_name in checkpoint filename if provided
    if experiment_name:
        checkpoint_file = output_dir / f".pipeline_checkpoint_{experiment_name}"
    else:
        checkpoint_file = output_dir / ".pipeline_checkpoint"
    
    if checkpoint_file.exists():
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            checkpoint_data = {}
    else:
        checkpoint_data = {}
    
    # Update with new data
    checkpoint_data.update(kwargs)
    checkpoint_data['last_updated'] = str(datetime.now())
    
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)

def check_step_completed(output_dir: Path, step_name: str, layer: int, config: str = None, method: str = None) -> bool:
    """Check if a specific step has been completed."""
    step_file = output_dir / ".completed_steps" / f"{step_name}_{layer}_{config}_{method}.done"
    return step_file.exists()

def mark_step_completed(output_dir: Path, step_name: str, layer: int, config: str = None, method: str = None):
    """Mark a step as completed."""
    step_file = output_dir / ".completed_steps" / f"{step_name}_{layer}_{config}_{method}.done"
    step_file.parent.mkdir(parents=True, exist_ok=True)
    step_file.touch()
    
    # Update checkpoint with step completion
    update_checkpoint(output_dir, 
                     last_completed_step=step_name,
                     last_completed_layer=layer,
                     last_completed_config=config,
                     last_completed_method=method)

def get_existing_normalizations(output_dir: Path, dataset: str, layer: int, config: str = None) -> List[str]:
    """Get list of normalization methods that already exist for a specific layer and config."""
    if config:
        normalized_dir = output_dir / "normalized" / config
    else:
        normalized_dir = output_dir / "normalized"
    
    if not normalized_dir.exists():
        return []
    
    existing_methods = []
    for method_dir in normalized_dir.iterdir():
        if method_dir.is_dir():
            if config:
                expected_file = method_dir / f"normalized_{dataset}_layer_{layer}_{config}_{method_dir.name}.parquet"
            else:
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
    
    # Configuraciones de dataset
    parser.add_argument("--configurations", nargs='+', default=DATASET_CONFIGURATIONS,
                       choices=["EC", "ECN"],
                       help="Dataset configurations to process (EC, ECN, or both)")
    
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
    elif not args.skip_embeddings:
        run_command(cmd)
    else:
        print(f"⏩ Skipping embeddings generation")
    
    return output_path

def step_2_normalize_embeddings(args, embedding_path: Path, method: str, layer: int, run_id: str) -> Path:
    """Step 2: Normalize embeddings using specified method."""
    print(f"\n🔧 Step 2: Normalizing embeddings using {method}")
    
    output_path = args.output_dir / "normalized" / method / f"normalized_{args.dataset}_layer_{layer}_{method}.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Log step information
    step_name = f"{run_id}_layer_{layer}_20_normalization_{method}"
    
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
    elif not args.skip_normalization:
        run_command(cmd)
    else:
        print(f"⏩ Skipping normalization")
    
    return output_path

def step_3_measure_anisotropy(args, normalized_path: Path, method: str, layer: int, run_id: str) -> Dict[str, Any]:
    """Step 3: Measure anisotropy for the normalized embeddings."""
    print(f"\n📐 Step 3: Measuring anisotropy for {method}")
    
    output_dir = args.output_dir / "anisotropy" / method
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Log step information
    step_name = f"{run_id}_layer_{layer}_70_anisotropy_{method}"
    
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
    else:
        print(f"⏩ Skipping anisotropy measurement")
    
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
        
    print(f"✅ Step completed")
    
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
        
    print(f"✅ Step completed")
    
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
    temp_file = temp_input_dir / f"embeddings_{args.dataset}_layer_{layer}_{config}.parquet"
    shutil.copy2(normalized_path, temp_file)
    
    cmd = [
        "python3", "experiments/50_contrastive_analysis.py",
        "--input_dir", str(temp_input_dir),
        "--output_dir", str(output_dir),
        "--layers", str(layer),
        "--methods", "arithmetic_mean", "geometric_median", "cross_differences",
        "--dataset_name", args.dataset,
        "--normalization_type", method,
        "--config", config,
        "--experiment_name", args.experiment_name,
        "--run_id", run_id,
        "--provenance", json.dumps(provenance)
    ]
    
    run_command(cmd)
    print(f"✅ Step completed")
    
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
    print(f"✅ Step completed")
    
    return {"method": method, "layer": layer, "output_dir": output_dir}

def step_2_separate_configurations(args, embedding_path: Path, layer: int, run_id: str) -> Dict[str, Path]:
    """Step 2: Separate embeddings into EC and ECN configurations."""
    print(f"\n🔀 Step 2: Separating embeddings into EC and ECN configurations")
    
    output_dir = args.output_dir / "separated"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if separation is already completed
    if check_step_completed(args.output_dir, "separation", layer):
        print(f"⏩ Separation already completed for layer {layer}")
        # Return existing paths
        return {
            "EC": output_dir / "EC" / f"embeddings_{args.dataset}_layer_{layer}_EC.parquet",
            "ECN": output_dir / "ECN" / f"embeddings_{args.dataset}_layer_{layer}_ECN.parquet"
        }
    
    # Create simple provenance
    provenance = {"step": "data_separation", "layer": layer}
    
    # Separate EC (entailment + contradiction only)
    ec_output = output_dir / "EC" / f"embeddings_{args.dataset}_layer_{layer}_EC.parquet"
    ec_output.parent.mkdir(parents=True, exist_ok=True)
    
    ec_cmd = [
        "python3", "experiments/15_separate_configurations.py",
        "--input_path", str(embedding_path),
        "--output_path", str(ec_output),
        "--config", "EC",
        "--dataset_name", args.dataset,
        "--layer_num", str(layer),
        "--experiment_name", args.experiment_name,
        "--run_id", run_id,
        "--provenance", json.dumps(provenance)
    ]
    
    # Separate ECN (entailment + contradiction + neutral)
    ecn_output = output_dir / "ECN" / f"embeddings_{args.dataset}_layer_{layer}_ECN.parquet"
    ecn_output.parent.mkdir(parents=True, exist_ok=True)
    
    ecn_cmd = [
        "python3", "experiments/15_separate_configurations.py",
        "--input_path", str(embedding_path),
        "--output_path", str(ecn_output),
        "--config", "ECN",
        "--dataset_name", args.dataset,
        "--layer_num", str(layer),
        "--experiment_name", args.experiment_name,
        "--run_id", run_id,
        "--provenance", json.dumps(provenance)
    ]
    
    run_command(ec_cmd)
    run_command(ecn_cmd)
    
    # Mark step as completed
    mark_step_completed(args.output_dir, "separation", layer)
    
    print(f"✅ Step completed")
    
    return {
        "EC": ec_output,
        "ECN": ecn_output
    }

def step_9_statistical_validation(args, results_dir: Path, run_id: str) -> Dict[str, Any]:
    """Step 9: Statistical validation of results."""
    print(f"\n📊 Step 9: Statistical validation")
    
    output_dir = args.output_dir / "statistical"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Log step information
    step_name = f"{run_id}_90_statistical_validation"
    
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
    print(f"✅ Step completed")
    
    return {"output_dir": output_dir}

def step_3_normalize_embeddings_config(args, embedding_path: Path, method: str, layer: int, config: str, run_id: str) -> Path:
    """Step 3: Normalize embeddings for a specific configuration."""
    print(f"\n🔧 Step 3: Normalizing embeddings for {config} - {method}")
    
    output_dir = args.output_dir / "normalized" / config / method
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / f"normalized_{args.dataset}_layer_{layer}_{config}_{method}.parquet"
    
    # Check if normalization is already completed
    if check_step_completed(args.output_dir, "normalization", layer, config, method):
        print(f"⏩ Normalization already completed for {config} - {method}")
        return output_path
    
    # Log step information
    step_name = f"{run_id}_layer_{layer}_{config}_20_normalization_{method}"
    
    # Create simple provenance
    provenance = {"step": "normalization", "method": method, "config": config, "layer": layer}
    
    cmd = [
        "python3", "experiments/20_normalization.py",
        "--source_path", str(embedding_path),
        "--out_path", str(output_path),
        "--normalization_type", method,
        "--layer_num", str(layer),
        "--dataset_name", args.dataset,
        "--config", config,
        "--experiment_name", args.experiment_name,
        "--run_id", run_id,
        "--provenance", json.dumps(provenance)
    ]
    
    run_command(cmd)
    
    # Mark step as completed
    mark_step_completed(args.output_dir, "normalization", layer, config, method)
    
    print(f"✅ Step completed")
    
    return output_path

def step_4_measure_anisotropy_config(args, normalized_path: Path, method: str, layer: int, config: str, run_id: str) -> Dict[str, Any]:
    """Step 4: Measure anisotropy for a specific configuration."""
    print(f"\n📐 Step 4: Measuring anisotropy for {config} - {method}")
    
    output_dir = args.output_dir / "anisotropy" / config / method
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Log step information
    step_name = f"{run_id}_layer_{layer}_{config}_70_anisotropy_{method}"
    
    # Create simple provenance
    provenance = {"step": "anisotropy", "method": method, "config": config, "layer": layer}
    
    cmd = [
        "python3", "experiments/70_anisotropy_analysis.py",
        "--input_path", str(normalized_path),
        "--output_dir", str(output_dir),
        "--dataset_name", args.dataset,
        "--layer_num", str(layer),
        "--embedding_type", "full",
        "--config", config,
        "--experiment_name", args.experiment_name,
        "--run_id", run_id,
        "--provenance", json.dumps(provenance)
    ]
    
    run_command(cmd)
    print(f"✅ Step completed")
    
    return {"method": method, "config": config, "layer": layer, "output_dir": output_dir}

def step_5_dimensionality_reduction_config(args, normalized_path: Path, method: str, layer: int, config: str, run_id: str) -> List[Path]:
    """Step 5: Apply dimensionality reduction for a specific configuration."""
    print(f"\n📉 Step 5: Dimensionality reduction for {config} - {method}")
    
    output_dir = args.output_dir / "dimensionality" / config / method
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Log step information
    step_name = f"{run_id}_layer_{layer}_{config}_30_dimensionality_{method}"
    
    # Create simple provenance
    provenance = {"step": "dimensionality_reduction", "method": method, "config": config, "layer": layer}
    
    # Iterate through all PCA components
    for pca_comp in args.pca_components:
        pca_output_dir = output_dir / f"pca_{pca_comp}"
        pca_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Run PCA
        pca_cmd = [
            "python3", "experiments/31_pca.py",
            "--source_path", str(normalized_path),
            "--out", str(pca_output_dir / f"pca_{args.dataset}_layer{layer}_{config}_{pca_comp}.parquet"),
            "--n_components", str(pca_comp),
            "--dataset_name", args.dataset,
            "--layer_num", str(layer),
            "--normalization_type", method,
            "--config", config,
            "--experiment_name", args.experiment_name,
            "--run_id", run_id,
            "--provenance", json.dumps(provenance)
        ]
        run_command(pca_cmd)
        
        # For each reduction type (PCA/ZCA)
        for reduction_type in args.reduction_types:
            # Try different possible file patterns
            possible_files = [
                pca_output_dir / f"{reduction_type}_{args.dataset}_layer{layer}_{config}_{pca_comp}.parquet",
                pca_output_dir / f"{reduction_type}_{reduction_type}_{args.dataset}_layer{layer}_{config}_{pca_comp}.parquet",
                pca_output_dir / f"{reduction_type}_pca_{args.dataset}_layer{layer}_{config}_{pca_comp}.parquet",
                # Add more patterns to catch all possible file names
                pca_output_dir / f"{reduction_type}_{args.dataset}_layer{layer}_{config}_{pca_comp}.parquet",
                pca_output_dir / f"{reduction_type}_{args.dataset}_layer{layer}_{config}_{pca_comp}.parquet",
                # Direct file names without prefixes
                pca_output_dir / f"{reduction_type}_{args.dataset}_layer{layer}_{config}_{pca_comp}.parquet"
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
                print(f"🔍 Processing slice_n={slice_n} with pca_comp={pca_comp}")
                if slice_n >= pca_comp:
                    print(f"⏭️  Skipping slice_n={slice_n} (>= pca_comp={pca_comp})")
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
                            umap_output = output_dir / f"umap_{args.dataset}_layer{layer}_{config}_{pca_comp}_{neighbors}_{min_dist}_{metric}.parquet"
                            
                            umap_cmd = [
                                "python3", "experiments/32_umap.py",
                                "--pca_path", str(umap_input),
                                "--out_path", str(umap_output),
                                "--n_neighbors", str(neighbors),
                                "--min_dist", str(min_dist),
                                "--metric", metric,
                                "--n_components", "2",
                                "--dataset_name", args.dataset,
                                "--reduction_type", reduction_type,
                                "--layer_num", str(layer),
                                "--input_n_components", str(pca_comp),
                                "--skipped_n_components", "0",
                                "--normalization_type", method,
                                "--config", config,
                                "--experiment_name", args.experiment_name,
                                "--run_id", run_id,
                                "--provenance", json.dumps(provenance)
                            ]
                            run_command(umap_cmd)
    
    print(f"✅ Step completed")
    
    # Return list of generated UMAP files
    umap_files = list(output_dir.rglob("umap_*.parquet"))
    print(f"📊 Generated {len(umap_files)} UMAP files:")
    for umap_file in umap_files:
        print(f"  - {umap_file}")
    
    return umap_files

def step_6_clustering_config(args, umap_path: Path, method: str, layer: int, config: str, run_id: str) -> Dict[str, Any]:
    """Step 6: Apply clustering analysis for a specific configuration."""
    print(f"\n🎯 Step 6: Clustering analysis for {config} - {method}")
    
    output_dir = args.output_dir / "clustering" / config / method
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Log step information
    step_name = f"{run_id}_layer_{layer}_{config}_40_clustering_{method}"
    
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
            "--dataset_name", args.dataset,
            "--layer_num", str(layer),
            "--reduction_type", "umap",
            "--normalization_type", method,
            "--config", config,
            "--experiment_name", args.experiment_name,
            "--run_id", run_id,
            "--provenance", json.dumps(provenance)
        ]
        
        run_command(cmd)
        results.append({"method": method, "config": config, "layer": layer, "k": k, "output_dir": k_output_dir})
        
    print(f"✅ Step completed")
    
    return results

def step_7_contrastive_analysis_config(args, normalized_path: Path, method: str, layer: int, config: str, run_id: str) -> Path:
    """Step 7: Apply contrastive analysis for a specific configuration."""
    print(f"\n🔄 Step 7: Contrastive analysis for {config} - {method}")
    
    output_dir = args.output_dir / "contrastive" / config / method
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Log step information
    step_name = f"{run_id}_layer_{layer}_{config}_50_contrastive_{method}"
    
    # Create simple provenance
    provenance = {"step": "contrastive_analysis", "method": method, "config": config, "layer": layer}
    
    # Crear directorio temporal con el archivo normalizado para el script
    temp_input_dir = output_dir / "temp_input"
    temp_input_dir.mkdir(parents=True, exist_ok=True)
    
    # Copiar el archivo normalizado al directorio temporal
    import shutil
    temp_file = temp_input_dir / f"embeddings_{args.dataset}_layer_{layer}_{config}.parquet"
    shutil.copy2(normalized_path, temp_file)
    
    cmd = [
        "python3", "experiments/50_contrastive_analysis.py",
        "--input_dir", str(temp_input_dir),
        "--output_dir", str(output_dir),
        "--layers", str(layer),
        "--methods", "arithmetic_mean", "geometric_median", "cross_differences",
        "--dataset_name", args.dataset,
        "--normalization_type", method,
        "--config", config,
        "--experiment_name", args.experiment_name,
        "--run_id", run_id,
        "--provenance", json.dumps(provenance)
    ]
    
    run_command(cmd)
    print(f"✅ Step completed")
    
    return output_dir

def step_8_probing_config(args, normalized_path: Path, method: str, layer: int, config: str, run_id: str) -> Dict[str, Any]:
    """Step 8: Apply decision tree probing for a specific configuration."""
    print(f"\n🔍 Step 8: Decision tree probing for {config} - {method}")
    
    output_dir = args.output_dir / "probes" / config / method
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Log step information
    step_name = f"{run_id}_layer_{layer}_{config}_60_probing_{method}"
    
    # Create simple provenance
    provenance = {"step": "probing", "method": method, "config": config, "layer": layer}
    
    cmd = [
        "python3", "experiments/60_decision_tree_probe.py",
        "--input_path", str(normalized_path),
        "--output_dir", str(output_dir),
        "--dataset_name", args.dataset,  # Corregido: dataset_name en lugar de dataset
        "--normalization_type", method,
        "--embedding_type", method,
        "--layer_num", str(layer),
        "--max_depth", str(args.probe_max_depth),
        "--min_samples_split", "2",
        "--scale_features",
        "--config", config,
        "--experiment_name", args.experiment_name,
        "--run_id", run_id,
        "--provenance", json.dumps(provenance)
    ]
    
    run_command(cmd)
    print(f"✅ Step completed")
    
    return {"method": method, "config": config, "layer": layer, "output_dir": output_dir}



def main():
    print("🚀 Starting pipeline...")
    args = parse_args()
    print(f"📊 Dataset: {args.dataset}")
    print(f"📁 Output dir: {args.output_dir}")
    print(f"⚙️  Configurations: {args.configurations}")
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure experiment based on experiment_name parameter or dataset
    experiment_name = args.experiment_name if args.experiment_name else f"logical_analysis_{args.dataset}"
    
    # Get or create run ID from checkpoint (using experiment_name)
    run_id = get_or_create_run_id(args.output_dir, experiment_name)
    print(f"🆔 Run ID: {run_id}")
    
    # No MLflow logging for the main pipeline - only individual scripts log
    
    print(f"🚀 Starting Main Pipeline for {args.dataset}")
    print(f"📊 MLflow Experiment: {experiment_name}")
    print(f"🆔 Common Run ID: {run_id}")
    
    # Update checkpoint with current execution info
    update_checkpoint(args.output_dir, experiment_name,
                     dataset=args.dataset,
                     layers=args.layers,
                     configurations=args.configurations)
    
    results = {}
    
    for layer in args.layers:
        print(f"\n{'='*60}")
        print(f"PROCESSING LAYER {layer}")
        print(f"{'='*60}")
        
        # Step 1: Extract embeddings
        embedding_path = step_1_extract_embeddings(args, layer, run_id)
        
        # Step 2: Separate into EC and ECN configurations
        separated_paths = step_2_separate_configurations(args, embedding_path, layer, run_id)
        
        # Process each configuration separately
        for config in args.configurations:
            print(f"\n{'#'*50}")
            print(f"PROCESSING CONFIGURATION: {config}")
            print(f"{'#'*50}")
            
            config_embedding_path = separated_paths[config]
            
            # Get existing normalizations for this layer and config
            existing_normalizations = get_existing_normalizations(args.output_dir, args.dataset, layer, config)
            print(f"📋 Existing normalizations for layer {layer} config {config}: {existing_normalizations}")
            
            # Process only missing normalizations
            missing_normalizations = [m for m in args.normalization_methods if m not in existing_normalizations]
            if missing_normalizations:
                print(f"🆕 Missing normalizations for layer {layer} config {config}: {missing_normalizations}")
            else:
                print(f"✅ All normalizations already exist for layer {layer} config {config}")
            
            for method in args.normalization_methods:
                if method in existing_normalizations:
                    print(f"\n{'~'*40}")
                    print(f"SKIPPING NORMALIZATION: {method.upper()} for {config} (already exists)")
                    print(f"{'~'*40}")
                    # Create the path for existing file
                    normalized_path = args.output_dir / "normalized" / config / method / f"normalized_{args.dataset}_layer_{layer}_{config}_{method}.parquet"
                else:
                    print(f"\n{'~'*40}")
                    print(f"PROCESSING NORMALIZATION: {method.upper()} for {config}")
                    print(f"{'~'*40}")
                    
                    # Step 3: Normalize embeddings for this config
                    normalized_path = step_3_normalize_embeddings_config(args, config_embedding_path, method, layer, config, run_id)
                
                # Step 4: Measure anisotropy for this config
                anisotropy_results = step_4_measure_anisotropy_config(args, normalized_path, method, layer, config, run_id)
                
                # Step 5: Dimensionality reduction for this config
                umap_files = step_5_dimensionality_reduction_config(args, normalized_path, method, layer, config, run_id)
                
                # Step 6: Clustering for this config
                for umap_file in umap_files:
                    clustering_results = step_6_clustering_config(args, umap_file, method, layer, config, run_id)
                
                # Step 7: Contrastive analysis for this config
                contrastive_dir = step_7_contrastive_analysis_config(args, normalized_path, method, layer, config, run_id)
                
                # Step 8: Probing for this config
                probing_results = step_8_probing_config(args, normalized_path, method, layer, config, run_id)
    
    # Step 9: Statistical validation
    statistical_results = step_9_statistical_validation(args, args.output_dir, run_id)
    
    # Log final artifacts
    print(f"\n✅ Main Pipeline completed successfully!")
    print(f"📊 Results saved to: {args.output_dir}")
    print(f"🔗 MLflow UI: mlflow ui")
    print(f"🆔 Run ID: {run_id}")

if __name__ == "__main__":
    main() 