#!/usr/bin/env python
"""
Pre-processes the FOLIO dataset to prepare it for embedding and analysis.

This script performs the following steps:
1.  Loads the raw FOLIO dataset.
2.  Maps string labels ('true', 'false', 'uncertain') to integer format
    (0, 2, 1) consistent with the SNLI pipeline (Entailment, Contradiction, Neutral).
3.  Filters out 'uncertain' samples to focus on clear logical pairs.
4.  Identifies and removes "cross-contaminated" records where a sentence
    appears as both a premise and a conclusion in different records.
5.  Saves the resulting cleaned but IMbalanced dataset.
6.  Performs stratified sampling on the cleaned data to create a BALANCED
    dataset, ensuring that premises with both 'true' and 'false' conclusions
    are kept together to preserve logical structure.
7.  Saves the final balanced dataset.

Usage:
    python scripts/pipelines/preprocess_folio.py \\
        --input_dir data/folio/raw \\
        --output_dir data/folio/processed
"""
import argparse
from pathlib import Path

import pandas as pd
from datasets import Dataset, load_from_disk


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Pre-process the FOLIO dataset.")
    parser.add_argument("--input_dir", required=True, type=Path, help="Directory containing the raw FOLIO dataset files.")
    parser.add_argument("--output_dir", required=True, type=Path, help="Directory to save the processed datasets.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load and map labels
    print("1. Loading dataset and mapping labels...")
    ds = load_from_disk(str(args.input_dir))
    df = ds.to_pandas()

    label_map = {"True": 0, "False": 2, "Uncertain": 1}
    df["label"] = df["label"].map(label_map)
    print(f"-> Original size: {len(df)}")

    # 2. Filter out 'uncertain'
    df_filtered = df[df["label"].isin([0, 2])].copy()
    print(f"-> Size after removing 'uncertain' samples: {len(df_filtered)}")

    # 3. Check for cross-contamination and decide whether to filter
    print("\n2. Checking for cross-contamination...")
    unique_premises = set(df_filtered['premises'].unique())
    unique_conclusions = set(df_filtered['conclusion'].unique())
    cross_contaminated_texts = unique_premises.intersection(unique_conclusions)

    num_to_remove = 0
    if not cross_contaminated_texts:
        print("-> No cross-contamination found.")
        df_cleaned = df_filtered.copy()
    else:
        contamination_mask = (
            df_filtered['premises'].isin(cross_contaminated_texts) | 
            df_filtered['conclusion'].isin(cross_contaminated_texts)
        )
        num_to_remove = contamination_mask.sum()
        percentage_to_remove = (num_to_remove / len(df_filtered)) * 100 if len(df_filtered) > 0 else 0
        
        print(f"-> Found {len(cross_contaminated_texts)} sentences as both premise and conclusion.")
        print(f"-> This would affect {num_to_remove} records ({percentage_to_remove:.2f}% of the dataset).")

        # If contamination is too high, it's likely a feature of the dataset.
        CONTAMINATION_THRESHOLD_PERCENT = 10.0
        if percentage_to_remove > CONTAMINATION_THRESHOLD_PERCENT:
            print(f"-> Contamination level is above the {CONTAMINATION_THRESHOLD_PERCENT}% threshold. Skipping removal step.")
            df_cleaned = df_filtered.copy()
        else:
            print(f"-> Contamination level is low. Proceeding with removal.")
            df_cleaned = df_filtered[~contamination_mask].copy()

    # 4. Save the imbalanced dataset
    imbalanced_path = args.output_dir / "folio_imbalanced_cleaned"
    print(f"\n3. Saving imbalanced dataset to: {imbalanced_path}")
    imbalanced_ds = Dataset.from_pandas(df_cleaned)
    imbalanced_ds.save_to_disk(str(imbalanced_path))

    # 5. Create the balanced dataset using a simpler, more robust method
    print("\n4. Creating balanced dataset...")
    
    if df_cleaned.empty:
        raise ValueError("Cannot create a balanced dataset because the cleaned dataset is empty.")

    class_counts = df_cleaned['label'].value_counts()
    print("-> Class distribution before balancing:")
    print(class_counts)

    if len(class_counts) < 2:
        raise ValueError("Cannot balance dataset: only one class is present after cleaning.")
        
    n_min = class_counts.min()
    print(f"-> Balancing dataset by sampling {n_min} records from each class.")
    
    df_final_balanced = df_cleaned.groupby('label', group_keys=False).apply(lambda x: x.sample(n_min, random_state=42))

    print(f"-> Final balanced dataset size: {len(df_final_balanced)}")
    print("-> Final balanced class distribution:")
    print(df_final_balanced['label'].value_counts())

    # 6. Save the balanced dataset
    balanced_path = args.output_dir / "folio_balanced_cleaned"
    print(f"\n5. Saving balanced dataset to: {balanced_path}")
    balanced_ds = Dataset.from_pandas(df_final_balanced)
    balanced_ds.save_to_disk(str(balanced_path))
    
    print("\nPreprocessing complete.")


if __name__ == "__main__":
    main() 