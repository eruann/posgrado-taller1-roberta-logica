#!/usr/bin/env python3
"""
experiments/90_statistical_validation.py - Validación Estadística Individual
==========================================================================
Script individual para validación estadística de resultados de clustering.

Usage:
    python experiments/90_statistical_validation.py \
        --results_dir data/snli/unified_pipeline \
        --output_dir data/snli/statistical \
        --dataset snli \
        --experiment_name statistical_validation
"""

import argparse
import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import chi2_contingency
from sklearn.metrics import adjusted_rand_score, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from scipy.stats import ttest_ind, mannwhitneyu, ks_2samp
from numpy import std, mean, sqrt
from sklearn.decomposition import PCA
from statsmodels.stats.multitest import multipletests

import mlflow

# --- Configure Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

class NpEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super(NpEncoder, self).default(obj)

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Individual statistical validation for clustering results")
    parser.add_argument("--results_dir", required=True, type=Path, help="Directory containing clustering results")
    parser.add_argument("--output_dir", required=True, type=Path, help="Output directory for statistical results")
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g., snli, folio)")
    parser.add_argument("--experiment_name", default="statistical_validation", help="MLflow experiment name")
    parser.add_argument("--bootstrap_iterations", type=int, default=50, help="Number of bootstrap iterations")
    parser.add_argument("--permutation_iterations", type=int, default=100, help="Number of permutation test iterations")
    parser.add_argument("--provenance", default="{}", help="Provenance JSON string")
    parser.add_argument("--run_id", default="", help="MLflow run ID")
    return parser.parse_args()

def test_clustering_significance(y_true, y_pred):
    """Test if clustering is significantly better than random"""
    contingency = confusion_matrix(y_true, y_pred)
    chi2, p_value = chi2_contingency(contingency)[:2]
    ari = adjusted_rand_score(y_true, y_pred)
    
    return {
        'chi2_statistic': float(chi2),
        'chi2_p_value': float(p_value),
        'adjusted_rand_index': float(ari),
        'significant': p_value < 0.05
    }

def compute_purity(y_true, y_pred):
    """Compute cluster purity"""
    contingency_matrix = confusion_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

def bootstrap_clustering_metrics(X, y_true, k_clusters, n_bootstrap=50):
    """Bootstrap confidence intervals for clustering metrics"""
    np.random.seed(42)
    metrics = []
    
    logging.info(f"Running {n_bootstrap} bootstrap iterations...")
    for i in range(n_bootstrap):
        if i % 10 == 0:
            logging.info(f"  Bootstrap iteration {i+1}/{n_bootstrap}")
        
        # Resample with replacement
        indices = np.random.choice(len(X), size=len(X), replace=True)
        X_boot = X[indices]
        y_boot = y_true[indices]
        
        # Run clustering and compute metrics
        kmeans = KMeans(n_clusters=k_clusters, random_state=42)
        pred_boot = kmeans.fit_predict(X_boot)
        
        # Compute purity
        purity = compute_purity(y_boot, pred_boot)
        nmi = normalized_mutual_info_score(y_boot, pred_boot)
        
        metrics.append({'purity': purity, 'nmi': nmi})
    
    # Compute confidence intervals (95%)
    return {
        'purity': {
            'mean': np.mean([m['purity'] for m in metrics]),
            'lower': np.percentile([m['purity'] for m in metrics], 2.5),
            'upper': np.percentile([m['purity'] for m in metrics], 97.5)
        },
        'nmi': {
            'mean': np.mean([m['nmi'] for m in metrics]),
            'lower': np.percentile([m['nmi'] for m in metrics], 2.5),
            'upper': np.percentile([m['nmi'] for m in metrics], 97.5)
        }
    }

def permutation_test_clustering(X, y_true, k_clusters, n_permutations=100):
    """Permutation test to assess clustering significance"""
    np.random.seed(42)
    
    # Run clustering on original data
    kmeans = KMeans(n_clusters=k_clusters, random_state=42)
    pred_original = kmeans.fit_predict(X)
    nmi_original = normalized_mutual_info_score(y_true, pred_original)
    
    # Run permutation tests
    nmi_permuted = []
    for i in range(n_permutations):
        if i % 20 == 0:
            logging.info(f"  Permutation test iteration {i+1}/{n_permutations}")
        
        # Permute labels
        y_permuted = np.random.permutation(y_true)
        pred_permuted = kmeans.fit_predict(X)
        nmi_perm = normalized_mutual_info_score(y_permuted, pred_permuted)
        nmi_permuted.append(nmi_perm)
    
    # Calculate p-value
    p_value = np.mean(np.array(nmi_permuted) >= nmi_original)
    
    return {
        'original_nmi': float(nmi_original),
        'permutation_p_value': float(p_value),
        'significant': p_value < 0.05
    }

def analyze_clustering_results(results_dir: Path, dataset: str):
    """Analyze clustering results from the pipeline output"""
    logging.info(f"Analyzing clustering results in: {results_dir}")
    
    # Find clustering result files
    clustering_files = list(results_dir.rglob("clustering_results.json"))
    
    if not clustering_files:
        logging.warning(f"No clustering results found in {results_dir}")
        return {}
    
    all_results = {}
    
    for result_file in clustering_files:
        try:
            with open(result_file, 'r') as f:
                clustering_data = json.load(f)
            
            # Extract information from file path
            path_parts = result_file.parts
            method_idx = path_parts.index('clustering') + 1
            config_idx = method_idx + 1
            k_idx = config_idx + 1
            
            if len(path_parts) > k_idx:
                method = path_parts[method_idx]
                config = path_parts[config_idx]
                k = int(path_parts[k_idx].replace('k', ''))
                
                logging.info(f"Analyzing: method={method}, config={config}, k={k}")
                
                # Perform statistical analysis
                if 'X' in clustering_data and 'y_true' in clustering_data and 'y_pred' in clustering_data:
                    X = np.array(clustering_data['X'])
                    y_true = np.array(clustering_data['y_true'])
                    y_pred = np.array(clustering_data['y_pred'])
                    
                    # Basic metrics
                    purity = compute_purity(y_true, y_pred)
                    nmi = normalized_mutual_info_score(y_true, y_pred)
                    
                    # Statistical tests
                    significance_test = test_clustering_significance(y_true, y_pred)
                    bootstrap_results = bootstrap_clustering_metrics(X, y_true, k, n_bootstrap=50)
                    permutation_results = permutation_test_clustering(X, y_true, k, n_permutations=100)
                    
                    # Store results
                    key = f"{method}_{config}_k{k}"
                    all_results[key] = {
                        'purity': purity,
                        'nmi': nmi,
                        'significance_test': significance_test,
                        'bootstrap_results': bootstrap_results,
                        'permutation_results': permutation_results
                    }
                    
        except Exception as e:
            logging.error(f"Error analyzing {result_file}: {e}")
            continue
    
    return all_results

def main():
    """Main execution function."""
    args = parse_args()
    
    # Setup output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Handle MLflow run creation - Flat structure
    if hasattr(args, 'experiment_name') and args.experiment_name:
        mlflow.set_experiment(args.experiment_name)
    
    # Create run with consistent naming pattern
    run_name = f"{args.run_id}_80_statistical_validation" if hasattr(args, 'run_id') and args.run_id else f"{args.dataset}_80_statistical_validation"
    
    with mlflow.start_run(run_name=run_name) as run:
        print(f"--- Starting Statistical Validation ---")
        
        # Log all parameters automatically
        mlflow.log_params(vars(args))
        
        # Log provenance if provided
        if hasattr(args, 'provenance') and args.provenance:
            try:
                provenance = json.loads(args.provenance)
                mlflow.log_params(provenance)
            except json.JSONDecodeError:
                print("Warning: Could not decode provenance JSON")
        
        # Analyze clustering results
        if args.results_dir.exists():
            results = analyze_clustering_results(args.results_dir, args.dataset)
            
            if results:
                # Save results to file
                results_file = args.output_dir / f"statistical_validation_{args.dataset}.json"
                with open(results_file, 'w') as f:
                    json.dump(results, f, indent=2, cls=NpEncoder)
                
                # Log artifact
                mlflow.log_artifact(str(results_file))
                
                # Log summary metrics
                for key, data in results.items():
                    mlflow.log_metric(f"{key}_purity", data['purity'])
                    mlflow.log_metric(f"{key}_nmi", data['nmi'])
                    mlflow.log_metric(f"{key}_significant", data['significance_test']['significant'])
                
                print(f"✅ Statistical validation completed successfully!")
                print(f"Results saved to: {results_file}")
                print(f"Analyzed {len(results)} clustering configurations")
            else:
                print(f"❌ No valid clustering results found to analyze")
        else:
            print(f"❌ Results directory not found: {args.results_dir}")
            return

if __name__ == "__main__":
    main() 