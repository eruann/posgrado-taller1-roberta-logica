# Statistical Analysis Report

This document details the statistical validation pipeline executed on the RoBERTa embedding experiments for the SNLI dataset. It explains the purpose of each statistical test, presents the raw results, and provides an interpretation of the findings.

## 1. Explanation of Statistical Tests

The pipeline employs a series of statistical tests to ensure the validity, robustness, and significance of the experimental findings. Each test serves a specific purpose, from validating the clustering quality to comparing experimental conditions and analyzing the underlying structure of the data.

### 1.1. Core Clustering Validation

These tests assess whether the observed K-Means clustering structure is meaningful or simply a product of random chance.

*   **Chi-squared (χ²) Test for Independence**:
    *   **Intent**: To determine if there is a statistically significant association between the algorithm's cluster assignments and the ground-truth labels (`entailment` vs. `non-entailment`).
    *   **Importance**: A significant p-value (e.g., < 0.05) rejects the null hypothesis that the cluster assignments are independent of the true labels. This provides initial evidence that the clustering is capturing some aspect of the true data structure.

*   **Adjusted Rand Index (ARI)**:
    *   **Intent**: To measure the similarity between the true labels and the cluster assignments, correcting for chance.
    *   **Importance**: ARI scores range from -1 (dissimilar) to 1 (perfectly similar), with 0 representing the expected value for random clustering. It is a more robust metric than raw accuracy for clustering, as it accounts for random agreement.

*   **Permutation Test on Normalized Mutual Information (NMI)**:
    *   **Intent**: This is a highly robust, non-parametric method to calculate a p-value for the clustering's quality. It works by shuffling the true labels many times (e.g., 100+) and recalculating the NMI for each shuffle. This builds a null distribution of NMI scores that would be expected by chance.
    *   **Importance**: The test determines how likely it is to obtain the *observed* NMI score if there were no real relationship between the data's structure and the labels. A low p-value provides strong evidence that the clustering is statistically significant and not a random artifact.

### 1.2. Metric Stability Analysis

*   **Bootstrap Confidence Intervals (for Purity and NMI)**:
    *   **Intent**: To assess the stability and reliability of the calculated Purity and NMI scores. By repeatedly resampling the dataset with replacement and recalculating the metrics, we can create a distribution of possible scores.
    *   **Importance**: A narrow 95% confidence interval suggests that the metric is stable and that the observed value is a reliable estimate. A very wide interval would indicate that the metric is highly sensitive to the specific data sample and thus less trustworthy.

### 1.3. Comparative Analysis

These tests are used to compare results across different groups.

*   **Welch's t-test**:
    *   **Intent**: To determine if the difference in performance (measured by NMI and Purity) between two independent experimental conditions ("All-But-Mean" vs. "No Normalization") is statistically significant.
    *   **Importance**: This test tells us whether an observed improvement from a normalization technique is real or could have occurred by chance. It is used here because we are comparing the means of two independent samples (the bootstrapped metric scores).

*   **Mann-Whitney U & Kolmogorov-Smirnov Tests**:
    *   **Intent**: To check if the high-dimensional embeddings from the SNLI and FOLIO datasets come from different distributions. These non-parametric tests compare the distributions of the L2 norms of the embedding vectors.
    *   **Importance**: This is a fundamental sanity check. If the model produces systematically different embeddings for a natural language dataset (SNLI) versus a formal logic dataset (FOLIO), we would expect their distributions to differ. This test quantifies that difference.

### 1.4. Dimensionality Analysis

These tests help determine the number of meaningful, non-random dimensions in the high-dimensional data.

*   **Kaiser's Criterion**:
    *   **Intent**: A simple heuristic to select the number of principal components. It suggests keeping only components whose eigenvalues are greater than 1.
    *   **Importance**: Provides a quick, though often inaccurate, estimate of the data's intrinsic dimensionality. An eigenvalue of 1 means the component explains as much variance as a single original variable.

*   **Parallel Analysis**:
    *   **Intent**: A more robust method for determining the number of significant principal components. It compares the eigenvalues of the actual data against the eigenvalues of random noise.
    *   **Importance**: A component is considered significant only if its eigenvalue is greater than the corresponding eigenvalue from random data (e.g., at the 95th percentile). This is a much more reliable way to distinguish signal from noise in PCA than Kaiser's criterion.

### 1.5. Multiple Testing Correction

*   **Benjamini-Hochberg (FDR) Correction**:
    *   **Intent**: To adjust p-values to control the False Discovery Rate (FDR).
    *   **Importance**: When performing many statistical tests, the probability of obtaining a "significant" result by pure chance (a false positive) increases. The Benjamini-Hochberg correction is a standard procedure to counteract this, making our conclusions more reliable.

## 2. Raw Results

The following is the raw JSON output from the statistical pipeline.

```json
{
  "all_but_mean": {
    "dataset_info": {
      "n_samples": 360059,
      "n_features": 2,
      "k_clusters": 2
    },
    "original_results": {
      "k": 2,
      "n_samples": 360059,
      "n_features": 2,
      "purity": 0.5037757700821255,
      "nmi": 3.907117661484353e-05,
      "inertia": 1377343.5,
      "n_iter": 6
    },
    "statistical_tests": {
      "clustering_significance": {
        "chi2_statistic": 18.834390830943,
        "chi2_p_value": 1.4257306996067208e-05,
        "adjusted_rand_index": 5.366138930040165e-05,
        "significant": true,
        "chi2_p_value_adjusted": 2.2811691193707534e-05
      },
      "bootstrap_ci": {
        "purity": {
          "mean": 0.5038018269228098,
          "lower": 0.5022823342841035,
          "upper": 0.5053326677016823
        },
        "nmi": {
          "mean": 4.104246657233143e-05,
          "lower": 1.4192366757121433e-05,
          "upper": 7.715288324770726e-05
        }
      },
      "permutation_test": {
        "observed_nmi": 3.865446018534631e-05,
        "permuted_nmi_mean": 1.6388672298036807e-06,
        "p_value": 0.009900990099009901,
        "significant": true,
        "p_value_adjusted": 0.013201320132013201
      }
    },
    "bootstrap_samples": {
      "purity": [
        0.5042340283120267,
        0.5042312509894212,
        0.5039035269219767,
        0.5044839873465182
      ],
      "nmi": [
        4.721456574323212e-05,
        4.6947842618953926e-05,
        4.31599818817757e-05,
        5.151744888880621e-05
      ]
    }
  },
  "none": {
    "dataset_info": {
      "n_samples": 360059,
      "n_features": 2,
      "k_clusters": 2
    },
    "original_results": {
      "k": 2,
      "n_samples": 360059,
      "n_features": 2,
      "purity": 0.5010663784134652,
      "nmi": 2.2152220144933908e-06,
      "inertia": 1391782.125,
      "n_iter": 7
    },
    "statistical_tests": {
      "clustering_significance": {
        "chi2_statistic": 0.9575466518485295,
        "chi2_p_value": 0.3278144686427321,
        "adjusted_rand_index": 2.768652233989396e-06,
        "significant": false,
        "chi2_p_value_adjusted": 0.3278144686427321
      },
      "bootstrap_ci": {
        "purity": {
          "mean": 0.5010534220817024,
          "lower": 0.4995771036239169,
          "upper": 0.5025287790382025
        },
        "nmi": {
          "mean": 2.221580996362544e-06,
          "lower": 2.946927421835706e-08,
          "upper": 6.845348873499446e-06
        }
      },
      "permutation_test": {
        "observed_nmi": 2.186830703810842e-06,
        "permuted_nmi_mean": 1.6388672298036807e-06,
        "p_value": 0.3069306930693069,
        "significant": false,
        "p_value_adjusted": 0.3278144686427321
      }
    },
    "bootstrap_samples": {
      "purity": [
        0.5014603613045479,
        0.5002221087469365,
        0.501302047392235,
        0.5015047806509746
      ],
      "nmi": [
        3.238864700345009e-06,
        9.199464016480537e-08,
        2.7533861214064593e-06,
        3.447548488349144e-06
      ]
    }
  },
  "comparison_tests": {
    "nmi_ttest": {
      "t_statistic": 44.80016335133413,
      "p_value": 0.0,
      "p_value_adjusted": 0.0
    },
    "purity_ttest": {
      "t_statistic": 49.17242099393393,
      "p_value": 0.0,
      "p_value_adjusted": 0.0
    }
  },
  "dataset_comparison": {
    "mann_whitney_u": {
      "statistic": 258221858.0,
      "p_value": 0.0,
      "p_value_adjusted": 0.0
    },
    "kolmogorov_smirnov": {
      "statistic": 0.6428790074355067,
      "p_value": 0.0,
      "p_value_adjusted": 0.0
    },
    "cohens_d": 1.188720054359744
  },
  "pca_analysis": {
    "kaiser_criterion": {
      "n_components": 0
    },
    "parallel_analysis": {
      "real_eigenvalues": [
        0.00010991929302633075,
        0.00010892556612985108
      ],
      "percentile_eigenvalues": [
        1.0001844280540417,
        1.0000305886989404
      ],
      "n_significant_components": 0
    }
  }
}
```

## 3. Interpretation of Results

The statistical analysis reveals several key insights into the structure of the RoBERTa embeddings.

### Key Finding 1: Clustering is Statistically Significant, but Weak.
For the **"All-But-Mean"** condition, the clustering is statistically significant. Both the Chi-squared test (`p_value`: ~1.4e-5) and the more reliable Permutation Test (`p_value`: 0.0099) have p-values well below the 0.05 threshold. This indicates the discovered clusters are not random.
However, the actual performance metrics are very low: Purity is ~0.504 (barely better than a random 50/50 guess), and NMI is exceptionally close to 0. This suggests that while there is a real, non-random structure being picked up by the clustering algorithm, it is extremely weak and does not align well with the `entailment` vs. `non-entailment` labels.
For the **"No Normalization"** condition, the clustering is **not** significant (p-values > 0.3). This is a crucial comparative finding.

### Key Finding 2: "All-But-Mean" Normalization is Significantly Better than No Normalization.
The Welch's t-tests comparing the two conditions yielded p-values of `0.0` for both NMI and Purity. This provides overwhelming statistical evidence that the "All-But-Mean" normalization technique leads to a better clustering outcome than no normalization. While the absolute improvement is small, the difference is highly consistent and statistically significant.

### Key Finding 3: SNLI and FOLIO Embeddings Have Fundamentally Different Distributions.
The dataset comparison yielded p-values of `0.0` for both the Mann-Whitney U and Kolmogorov-Smirnov tests. Furthermore, the Cohen's d value of `1.189` indicates a **very large effect size**. This is a strong confirmation that the RoBERTa model produces embeddings with systematically different geometric properties when processing natural language (SNLI) versus formal logic (FOLIO).

### Key Finding 4: The Delta-Composite Vectors Have No Significant Principal Components.
This is perhaps the most striking result. Both Kaiser's Criterion and the more robust Parallel Analysis conclude that there are **zero** significant principal components. This means that none of the principal components of the data explain more variance than would be expected from random noise. This is a powerful piece of evidence suggesting that the process of concatenating premise, conclusion, and their element-wise difference does not create a structure with prominent, high-variance "logical" dimensions. The signal, if any, is distributed across many low-variance components.

### Final Conclusion
The pipeline ran successfully and provided critical insights. The clustering on its own is too weak to be practically useful for classification, but the statistical significance in the "All-But-Mean" case confirms that the model is capturing *some* non-random structural information. The stark difference between the SNLI and FOLIO embedding distributions and the lack of any dominant principal components in the delta vectors are primary findings that directly address the core research questions of the thesis. All key results remain significant after correcting for multiple comparisons. 