#!/usr/bin/env python
"""
experiments/probes/analysis_utils.py
====================================
Utilities for analyzing the results of a decision tree probe.
"""
import pandas as pd
from sklearn.tree import _tree
from sklearn.metrics import confusion_matrix, precision_score, recall_score

def get_classification_metrics(y_true, y_pred) -> dict:
    """Calculates precision, recall, and confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    # Determine if binary or multiclass
    n_classes = len(set(y_true) | set(y_pred))
    if n_classes == 2:
        # For binary classification, use 'weighted' to avoid pos_label issues
        average = 'weighted'
    else:
        average = 'weighted'  # Use weighted average for multiclass
    
    metrics = {
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "confusion_matrix": cm.tolist() # Convert to list for JSON serialization
    }
    return metrics

def get_top_features(importances, feature_names, top_n=10) -> pd.DataFrame:
    """Gets the top N most important features from the decision tree."""
    imp_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    return imp_df.head(top_n)

def get_printable_rules(tree, feature_names, class_names, max_depth=3) -> str:
    """
    Extracts and formats human-readable rules from a trained scikit-learn Tree.
    """
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    
    rules_str = ""

    def recurse(node, depth):
        nonlocal rules_str
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:  # Not a leaf
            name = feature_name[node]
            threshold = tree_.threshold[node]
            rules_str += f"{indent}if {name} <= {threshold:.3f}:\n"
            if depth < max_depth:
                recurse(tree_.children_left[node], depth + 1)
            else:
                 rules_str += f"{indent}  [... deeper ...]\n"

            rules_str += f"{indent}else:  # if {name} > {threshold:.3f}\n"
            if depth < max_depth:
                recurse(tree_.children_right[node], depth + 1)
            else:
                 rules_str += f"{indent}  [... deeper ...]\n"
        else:  # Leaf node
            values = tree_.value[node][0]
            class_idx = values.argmax()
            confidence = values[class_idx] / values.sum()
            rules_str += f"{indent}--> predict: {class_names[class_idx]} ({confidence:.2%} confidence, {int(values.sum())} samples)\n"

    rules_str += "Decision Tree Rules (Top Levels):\n"
    recurse(0, 1)
    return rules_str 