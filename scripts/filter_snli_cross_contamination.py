import pandas as pd
import numpy as np
from datasets import load_dataset, Dataset
import os
from collections import Counter

def filter_cross_contaminated_records():
    """
    Loads the SNLI dataset, identifies and removes records where the same text
    appears as both premise and hypothesis (in different records), and saves
    the filtered dataset.
    """
    
    # 1. Load the SNLI dataset
    print("Loading SNLI dataset...")
    snli_dataset = load_dataset('arrow', data_files={
        'train': 'data/snli/dataset/data-00000-of-00001.arrow'
    })
    
    print(f"Original dataset size: {len(snli_dataset['train'])}")
    
    # Convert to pandas for easier manipulation
    df = snli_dataset['train'].to_pandas()
    
    # Clean data: remove entries with label -1 (unlabeled)
    df = df[df['label'] != -1].copy()
    print(f"After removing unlabeled data: {len(df)}")
    
    # 2. Identify cross-contaminated texts
    unique_premises = set(df['premise'].unique())
    unique_hypotheses = set(df['hypothesis'].unique())
    
    # Find texts that appear as both premise and hypothesis
    cross_contaminated_texts = unique_premises.intersection(unique_hypotheses)
    print(f"Found {len(cross_contaminated_texts)} texts appearing as both premise and hypothesis")
    
    # 3. Create filter mask
    # Remove records where either premise OR hypothesis is in the contaminated set
    contaminated_mask = (
        df['premise'].isin(cross_contaminated_texts) | 
        df['hypothesis'].isin(cross_contaminated_texts)
    )
    
    print(f"Records to be removed: {contaminated_mask.sum()}")
    print(f"Records to be kept: {(~contaminated_mask).sum()}")
    
    # 4. Filter the dataset
    filtered_df = df[~contaminated_mask].copy()
    
    # 5. Verify the filtering worked
    remaining_premises = set(filtered_df['premise'].unique())
    remaining_hypotheses = set(filtered_df['hypothesis'].unique())
    remaining_overlap = remaining_premises.intersection(remaining_hypotheses)
    
    print(f"Verification - remaining cross-contamination: {len(remaining_overlap)}")
    assert len(remaining_overlap) == 0, "Filtering failed - cross-contamination still exists"
    
    # 6. Convert back to HuggingFace Dataset format
    filtered_dataset = Dataset.from_pandas(filtered_df)
    
    # 7. Save the filtered dataset
    output_dir = "data/snli/filtered"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as arrow file
    filtered_dataset.save_to_disk(f"{output_dir}/snli_filtered")
    print(f"Filtered dataset saved to {output_dir}/snli_filtered")
    
    # 8. Generate filtering report
    report_path = f"{output_dir}/filtering_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("SNLI Dataset Cross-Contamination Filtering Report\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Original dataset size: {len(snli_dataset['train'])}\n")
        f.write(f"After removing unlabeled: {len(df)}\n")
        f.write(f"Cross-contaminated texts found: {len(cross_contaminated_texts)}\n")
        f.write(f"Records removed: {contaminated_mask.sum()}\n")
        f.write(f"Final dataset size: {len(filtered_df)}\n")
        f.write(f"Reduction: {contaminated_mask.sum() / len(df) * 100:.2f}%\n\n")
        
        f.write("Label distribution in filtered dataset:\n")
        label_map = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}
        label_counts = filtered_df['label'].value_counts().sort_index()
        for label_idx, count in label_counts.items():
            label_name = label_map.get(label_idx, f'unknown_{label_idx}')
            f.write(f"- {label_name}: {count}\n")
        
        f.write(f"\nVerification - remaining cross-contamination: {len(remaining_overlap)}\n")
        
        # Sample of removed texts
        f.write(f"\nSample of cross-contaminated texts (first 10):\n")
        for i, text in enumerate(list(cross_contaminated_texts)[:10]):
            f.write(f"{i+1}. \"{text}\"\n")
    
    print(f"Filtering report saved to {report_path}")
    
    return filtered_dataset, len(cross_contaminated_texts), contaminated_mask.sum()

if __name__ == '__main__':
    filtered_dataset, num_contaminated_texts, num_removed_records = filter_cross_contaminated_records()

    final_size = len(filtered_dataset)
    # The count before this filter is the final size plus the number of removed records.
    # This corresponds to the size after removing unlabeled records from the original dataset.
    initial_size = final_size + num_removed_records

    print("\n--- Resumen del Filtrado ---")
    print(f"Registros antes del filtrado (sin 'unlabeled'): {initial_size}")
    print(f"Registros eliminados por contaminación:         {num_removed_records}")
    print(f"Registros después del filtrado (final):         {final_size}")
    print("--------------------------------------------------")
    print(f"Textos únicos contaminados encontrados: {num_contaminated_texts}")
