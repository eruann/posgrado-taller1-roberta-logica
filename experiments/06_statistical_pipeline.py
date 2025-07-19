#!/usr/bin/env python3
"""
experiments/06_statistical_pipeline_simple.py
Pipeline estadístico simplificado y funcional
"""
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

# --- Configure Logging ---
log_path = Path(__file__).resolve().parent
log_file = log_path / "06_statistical_pipeline.log"

# Remove existing log file
if log_file.exists():
    log_file.unlink()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
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
    """Bootstrap confidence intervals for clustering metrics (reduced iterations)"""
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
    """Perform a permutation test for clustering significance."""
    np.random.seed(42)
    
    # Calculate the observed NMI
    kmeans = KMeans(n_clusters=k_clusters, random_state=42)
    y_pred_observed = kmeans.fit_predict(X)
    observed_nmi = normalized_mutual_info_score(y_true, y_pred_observed)
    
    # Calculate NMI for permuted labels
    permuted_nmis = []
    logging.info(f"\nRunning {n_permutations} permutation test iterations (Corrected Logic)...")
    for i in range(n_permutations):
        if (i + 1) % 10 == 0:
            logging.info(f"  Permutation iteration {i + 1}/{n_permutations}")
        y_permuted = np.random.permutation(y_true)
        # Clustering is always on the data X. The labels are permuted for evaluation.
        # The predictions y_pred_observed do not change.
        permuted_nmis.append(normalized_mutual_info_score(y_permuted, y_pred_observed))

    p_value = (np.sum(np.array(permuted_nmis) >= observed_nmi) + 1) / (n_permutations + 1)
    
    return {
        'observed_nmi': observed_nmi,
        'permuted_nmi_mean': np.mean(permuted_nmis),
        'p_value': p_value,
        'significant': p_value < 0.05
    }

def get_bootstrap_samples(X, y_true, k_clusters, n_bootstrap=500):
    """Just get the bootstrap scores for a given metric."""
    np.random.seed(42)
    metrics = {'purity': [], 'nmi': []}
    
    for _ in range(n_bootstrap):
        indices = np.random.choice(len(X), size=len(X), replace=True)
        X_boot, y_boot = X[indices], y_true[indices]
        
        kmeans = KMeans(n_clusters=k_clusters, random_state=42)
        pred_boot = kmeans.fit_predict(X_boot)
        
        metrics['purity'].append(compute_purity(y_boot, pred_boot))
        metrics['nmi'].append(normalized_mutual_info_score(y_boot, pred_boot))
        
    return metrics

def compare_metrics_ttest(samples1, samples2):
    """Perform an independent t-test on two sets of metric samples."""
    t_stat, p_value = ttest_ind(samples1, samples2, equal_var=False) # Welch's t-test
    return {'t_statistic': t_stat, 'p_value': p_value}

def cohen_d(x,y):
    """Calculate Cohen's d for independent samples"""
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (mean(x) - mean(y)) / sqrt(((nx-1)*std(x, ddof=1) ** 2 + (ny-1)*std(y, ddof=1) ** 2) / dof)

def compare_datasets(snli_embedding_path, folio_embedding_path):
    """Load SNLI and FOLIO embeddings and compare their distributions."""
    logging.info("\n=== Comparing Datasets (SNLI vs. FOLIO) ===")
    
    if not snli_embedding_path.exists() or not folio_embedding_path.exists():
        logging.error("ERROR: Embedding files not found. Skipping dataset comparison.")
        return None
        
    logging.info("Loading SNLI embeddings...")
    df_snli = pd.read_parquet(snli_embedding_path)
    
    logging.info("Loading FOLIO embeddings...")
    df_folio = pd.read_parquet(folio_embedding_path)
    
    # Assuming embeddings are stored in columns, not as a single vector column
    # Let's find the feature columns (e.g., 'emb_0', 'emb_1', ...)
    snli_feature_cols = [c for c in df_snli.columns if c.startswith('feature_')]
    folio_feature_cols = [c for c in df_folio.columns if c.startswith('feature_')]
    
    if not snli_feature_cols or not folio_feature_cols:
        logging.error("ERROR: Could not find embedding feature columns. Skipping dataset comparison.")
        return None

    # Calculate L2 norms
    snli_norms = np.linalg.norm(df_snli[snli_feature_cols].values, axis=1)
    folio_norms = np.linalg.norm(df_folio[folio_feature_cols].values, axis=1)
    
    logging.info(f"Comparing L2 norm distributions: SNLI (n={len(snli_norms)}) vs. FOLIO (n={len(folio_norms)})")
    
    # Perform statistical tests
    mwu_stat, mwu_p = mannwhitneyu(snli_norms, folio_norms, alternative='two-sided')
    ks_stat, ks_p = ks_2samp(snli_norms, folio_norms)
    cohen_d_score = cohen_d(snli_norms, folio_norms)
    
    logging.info(f"Mann-Whitney U: statistic={mwu_stat:.3f}, p-value={mwu_p:.4f}")
    logging.info(f"Kolmogorov-Smirnov: statistic={ks_stat:.3f}, p-value={ks_p:.4f}")
    logging.info(f"Cohen's d: {cohen_d_score:.3f}")
    
    return {
        "mann_whitney_u": {"statistic": mwu_stat, "p_value": mwu_p},
        "kolmogorov_smirnov": {"statistic": ks_stat, "p_value": ks_p},
        "cohens_d": cohen_d_score
    }

def parallel_analysis(data, n_reps=5, percentile=95):
    """Perform parallel analysis to determine the number of significant PCA components."""
    logging.info(f"\n--- Running Parallel Analysis ({n_reps} reps) ---")
    n_samples, n_features = data.shape
    
    # Get eigenvalues from real data
    pca_real = PCA()
    pca_real.fit(data)
    real_eigenvalues = pca_real.explained_variance_
    
    # Get eigenvalues from random data
    random_eigenvalues = np.zeros((n_reps, n_features))
    for i in range(n_reps):
        logging.info(f"  Rep {i+1}/{n_reps}...")
        random_data = np.random.normal(size=(n_samples, n_features))
        pca_random = PCA()
        pca_random.fit(random_data)
        random_eigenvalues[i, :] = pca_random.explained_variance_
        
    # Get the percentile of random eigenvalues
    percentile_eigenvalues = np.percentile(random_eigenvalues, percentile, axis=0)
    
    # Determine the number of significant components
    n_significant_components = np.sum(real_eigenvalues > percentile_eigenvalues)
    
    return {
        "real_eigenvalues": real_eigenvalues.tolist(),
        "percentile_eigenvalues": percentile_eigenvalues.tolist(),
        "n_significant_components": n_significant_components
    }

def analyze_pca(pre_pca_data_path):
    """Load pre-PCA data and perform Kaiser and Parallel Analysis."""
    logging.info("\n=== Performing PCA Analysis ===")
    
    if not pre_pca_data_path.exists():
        logging.error("ERROR: Pre-PCA data file not found. Skipping PCA analysis.")
        return None
        
    logging.info(f"Loading pre-PCA data from {pre_pca_data_path}...")
    df = pd.read_parquet(pre_pca_data_path)
    
    feature_cols = [c for c in df.columns if c.startswith('delta_')]
    if not feature_cols:
        logging.error("ERROR: Could not find embedding feature columns. Skipping PCA analysis.")
        return None
        
    data = df[feature_cols].values
    
    # Kaiser Criterion (requires eigenvalues, so we need to fit PCA)
    pca = PCA()
    pca.fit(data)
    eigenvalues = pca.explained_variance_
    kaiser_components = np.sum(eigenvalues > 1)
    logging.info(f"Kaiser Criterion: {kaiser_components} components with eigenvalue > 1.")
    
    # Parallel Analysis
    # NOTE: This is computationally expensive. Using a low number of reps for this example.
    pa_results = parallel_analysis(data, n_reps=5)
    logging.info(f"Parallel Analysis: {pa_results['n_significant_components']} components are significant.")
    
    return {
        "kaiser_criterion": {"n_components": int(kaiser_components)},
        "parallel_analysis": pa_results
    }


def run_analysis_for_condition(condition_config):
    """Runs the full statistical analysis for a given experimental condition."""
    
    logging.info(f"\n=== Running Analysis for: {condition_config['name']} ===")
    
    kmeans_path = condition_config['kmeans_path']
    umap_path = condition_config['umap_path']

    if not kmeans_path.exists():
        logging.error(f"ERROR: File not found: {kmeans_path}")
        return None

    with open(kmeans_path) as f:
        results = json.load(f)
    
    logging.info(f"Loaded results: Purity={results['purity']:.3f}, NMI={results['nmi']:.6f}")

    umap_files = list(umap_path.glob("*.parquet"))
    if not umap_files:
        logging.error(f"ERROR: No UMAP files found in {umap_path}")
        return None
        
    df = pd.read_parquet(umap_files[0])
    
    feature_cols = [c for c in df.columns if c.startswith('UMAP_')]
    if not feature_cols:
        if 'UMAP_0' in df.columns and 'UMAP_1' in df.columns:
            feature_cols = ['UMAP_0', 'UMAP_1']
        else:
            raise ValueError("Could not find UMAP feature columns in the Parquet file.")
            
    X = df[feature_cols].values
    y_true = df['label'].values
    
    unique_labels = np.unique(y_true)
    k_clusters = len(unique_labels)
    label_mapping = {label: i for i, label in enumerate(unique_labels)}
    y_true_mapped = np.array([label_mapping[label] for label in y_true])

    logging.info("Running K-means clustering...")
    kmeans = KMeans(n_clusters=k_clusters, random_state=42)
    y_pred = kmeans.fit_predict(X)

    logging.info("\n--- Test 1: Clustering Significance ---")
    sig_results = test_clustering_significance(y_true_mapped, y_pred)

    logging.info("\n--- Test 2: Bootstrap Confidence Intervals ---")
    # We need the raw samples for comparison, so let's get them here.
    bootstrap_samples = get_bootstrap_samples(X, y_true_mapped, k_clusters)
    bootstrap_results = {
        'purity': {
            'mean': np.mean(bootstrap_samples['purity']),
            'lower': np.percentile(bootstrap_samples['purity'], 2.5),
            'upper': np.percentile(bootstrap_samples['purity'], 97.5)
        },
        'nmi': {
            'mean': np.mean(bootstrap_samples['nmi']),
            'lower': np.percentile(bootstrap_samples['nmi'], 2.5),
            'upper': np.percentile(bootstrap_samples['nmi'], 97.5)
        }
    }
    
    logging.info("\n--- Test 3: Permutation Test ---")
    perm_results = permutation_test_clustering(X, y_true_mapped, k_clusters, n_permutations=100)

    return {
        'dataset_info': {
            'n_samples': len(X),
            'n_features': len(feature_cols),
            'k_clusters': k_clusters,
        },
        'original_results': results,
        'statistical_tests': {
            'clustering_significance': sig_results,
            'bootstrap_ci': bootstrap_results,
            'permutation_test': perm_results
        },
        'bootstrap_samples': bootstrap_samples # Store for comparison
    }

def apply_benjamini_hochberg_correction(results):
    """
    Find all p-values in the results dictionary, apply FDR correction,
    and add the adjusted p-values back into the dictionary.
    """
    logging.info("\n=== Applying Benjamini-Hochberg FDR Correction ===")
    
    p_value_paths = []
    p_values = []

    # --- Collect all p-values and their locations ---
    for condition in ['all_but_mean', 'none']:
        if condition in results:
            # Chi-squared p-value
            p_value_paths.append([condition, 'statistical_tests', 'clustering_significance', 'chi2_p_value'])
            p_values.append(results[condition]['statistical_tests']['clustering_significance']['chi2_p_value'])
            # Permutation test p-value
            p_value_paths.append([condition, 'statistical_tests', 'permutation_test', 'p_value'])
            p_values.append(results[condition]['statistical_tests']['permutation_test']['p_value'])

    # T-test p-values
    if 'comparison_tests' in results:
        p_value_paths.append(['comparison_tests', 'nmi_ttest', 'p_value'])
        p_values.append(results['comparison_tests']['nmi_ttest']['p_value'])
        p_value_paths.append(['comparison_tests', 'purity_ttest', 'p_value'])
        p_values.append(results['comparison_tests']['purity_ttest']['p_value'])

    # Dataset comparison p-values
    if 'dataset_comparison' in results:
        p_value_paths.append(['dataset_comparison', 'mann_whitney_u', 'p_value'])
        p_values.append(results['dataset_comparison']['mann_whitney_u']['p_value'])
        p_value_paths.append(['dataset_comparison', 'kolmogorov_smirnov', 'p_value'])
        p_values.append(results['dataset_comparison']['kolmogorov_smirnov']['p_value'])
        
    if not p_values:
        logging.info("No p-values found to correct.")
        return results

    # --- Apply Benjamini-Hochberg correction ---
    reject, pvals_corrected, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')

    # --- Add adjusted p-values back to the results dictionary ---
    for i, path in enumerate(p_value_paths):
        # Create a new key for the adjusted p-value
        adjusted_key = path[-1] + '_adjusted'
        # Navigate to the correct nested dictionary
        sub_dict = results
        for key in path[:-1]:
            sub_dict = sub_dict[key]
        # Add the adjusted p-value
        sub_dict[adjusted_key] = pvals_corrected[i]
        
    logging.info(f"Corrected {len(p_values)} p-values.")
    return results


def main():
    # Common paths
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent
    base_path = project_root / "data/snli/norm_comp_delta_ec_only"
    
    # Configuration for different experimental conditions
    conditions = {
        "all_but_mean": {
            "name": "All-But-Mean Normalization",
            "kmeans_path": base_path / "05_kmeans_delta_all_but_mean/layer_12/skip_30/zca/components_50/neighbors_15/euclidean/kmeans_results.json",
            "umap_path": base_path / "04_umap_delta_all_but_mean/layer_12/skip_30/zca/components_50/neighbors_15/euclidean"
        },
        "none": {
            "name": "No Normalization",
            "kmeans_path": base_path / "05_kmeans_delta_none/layer_12/skip_30/zca/components_50/neighbors_15/euclidean/kmeans_results.json",
            "umap_path": base_path / "04_umap_delta_none/layer_12/skip_30/zca/components_50/neighbors_15/euclidean"
        }
    }

    # Run analysis for each condition
    all_results = {}
    for key, config in conditions.items():
        condition_results = run_analysis_for_condition(config)
        if condition_results:
            all_results[key] = condition_results
            
    #--- Comparison Phase ---
    if "all_but_mean" in all_results and "none" in all_results:
        logging.info("\n=== Comparing Conditions (All-But-Mean vs. None) ===")
        
        # Extract bootstrap samples
        samples_abm = all_results['all_but_mean']['bootstrap_samples']
        samples_none = all_results['none']['bootstrap_samples']
        
        # Compare NMI
        nmi_comparison = compare_metrics_ttest(samples_abm['nmi'], samples_none['nmi'])
        logging.info(f"NMI T-test: t={nmi_comparison['t_statistic']:.3f}, p={nmi_comparison['p_value']:.4f}")

        # Compare Purity
        purity_comparison = compare_metrics_ttest(samples_abm['purity'], samples_none['purity'])
        logging.info(f"Purity T-test: t={purity_comparison['t_statistic']:.3f}, p={purity_comparison['p_value']:.4f}")
        
        all_results['comparison_tests'] = {
            'nmi_ttest': nmi_comparison,
            'purity_ttest': purity_comparison
        }

    #--- Dataset Comparison Phase ---
    snli_embeddings_path = project_root / "data/snli/embeddings/filtered/embeddings/embeddings_snli_layer_12.parquet"
    folio_embeddings_path = project_root / "data/folio/minimal_analysis/embeddings/balanced/embeddings_folio_layer_12.parquet"
    dataset_comparison_results = compare_datasets(snli_embeddings_path, folio_embeddings_path)
    if dataset_comparison_results:
        all_results['dataset_comparison'] = dataset_comparison_results

    #--- PCA Analysis Phase ---
    pre_pca_path = project_root / "data/snli/norm_comp_delta_ec_only/01_normalized_delta_all_but_mean/embeddings_delta_snli_layer_12_all_but_mean.parquet"
    pca_analysis_results = analyze_pca(pre_pca_path)
    if pca_analysis_results:
        all_results['pca_analysis'] = pca_analysis_results

    #--- Final Correction Phase ---
    all_results = apply_benjamini_hochberg_correction(all_results)

    # Save combined results
    output_path = project_root / "experiments"
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / "snli_statistical_comparison_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, cls=NpEncoder)
    
    logging.info(f"\nCombined results saved to: {output_file}")

if __name__ == "__main__":
    main() 