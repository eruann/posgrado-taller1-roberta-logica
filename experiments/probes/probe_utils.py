#!/usr/bin/env python
"""
experiments/probes/probe_utils.py
=================================
Core utilities for training a decision tree probe on embeddings.
Uses cuML for GPU acceleration.
"""
import cudf
import cupy as cp
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from pathlib import Path

def train_decision_tree_probe(
    data_path: Path,
    max_depth: int = 10,
    min_samples_split: int = 50,
    scale_features: bool = False,
    test_size: float = 0.2,
    random_state: int = 42
) -> dict:
    """
    Loads data and trains a single Decision Tree classifier probe on the GPU.

    Args:
        data_path: Path to the Parquet file with embeddings and labels.
        max_depth: Maximum depth of the decision tree.
        min_samples_split: The minimum number of samples required to split an internal node.
        scale_features: Whether to apply StandardScaler to the features.
        test_size: Proportion of the dataset to include in the test split.
        random_state: Seed for reproducibility.

    Returns:
        A dictionary containing the trained model, metrics, and data splits.
    """
    print(f"--- [1/5] Loading data from: {data_path} ---")
    try:
        gdf = cudf.read_parquet(data_path)
    except Exception as e:
        raise IOError(f"Could not read Parquet file at {data_path}") from e

    # Exclude non-feature columns (IDs and labels)
    exclude_cols = ['label', 'premise_id', 'hypothesis_id']
    feature_cols = [col for col in gdf.columns if col not in exclude_cols]
    X = gdf[feature_cols]
    y = gdf['label']
    
    # If labels are [0, 2] (Entailment, Contradiction), remap to [0, 1] for binary metrics.
    unique_labels = y.unique()
    if sorted(unique_labels.to_numpy().tolist()) == [0, 2]:
        print("Remapping labels from {0, 2} to {0, 1} for binary classification.")
        y = y.replace({2: 1})

    # Convert to pandas/numpy for scikit-learn
    X_pd = X.to_pandas()
    y_pd = y.to_pandas()

    print(f"Loaded {len(gdf)} samples with {len(feature_cols)} features.")
    
    # Stratified train-test split
    print(f"--- [2/5] Splitting data ({1-test_size:.0%}/{test_size:.0%} ratio) ---")
    X_train, X_test, y_train, y_test = train_test_split(
        X_pd, y_pd, 
        test_size=test_size, 
        random_state=42,
        stratify=y_pd
    )
    print(f"Train/Test split: {len(X_train)} / {len(X_test)}")
    
    # Optional feature scaling
    if scale_features:
        print("--- [3/5] Applying StandardScaler to features... ---")
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        print("✓ Scaling complete.")
    else:
        print("--- [3/5] Skipping feature scaling. ---")

    # Train model
    print(f"--- [4/5] Training Decision Tree (max_depth={max_depth})... ---")
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42
    )
    model.fit(X_train, y_train)
    print("✓ Training complete.")

    # Make predictions
    print("--- [5/5] Making predictions on the test set... ---")
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    acc = accuracy_score(y_test, y_pred)
    print(f"✓ Probe Accuracy: {acc:.4f}")

    return {
        "model": model,
        "accuracy": acc,
        "feature_names": feature_cols,
        "class_names": [str(c) for c in model.classes_],
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred": y_pred,
        "feature_importances": model.feature_importances_,
    } 