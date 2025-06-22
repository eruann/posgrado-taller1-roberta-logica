#!/usr/bin/env python
"""
scripts/fix_column_names.py
===========================

Fix the column names in the wide format embedding files to match
the expected format for the pipeline scripts.

Expected structure:
- premise_0, premise_1, ..., premise_767 (768 features)
- hypothesis_0, hypothesis_1, ..., hypothesis_767 (768 features)  
- delta_0, delta_1, ..., delta_767 (768 features)
- label

Usage:
    python scripts/fix_column_names.py --input_file data/snli/embeddings/embeddings_snli_layer_9.parquet
"""

import argparse
import pandas as pd
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Fix column names in wide format embedding files")
    parser.add_argument("--input_file", type=Path, help="Input parquet file to fix")
    parser.add_argument("--input_dir", type=Path, help="Directory containing multiple files to fix")
    parser.add_argument("--output_suffix", default="_fixed", help="Suffix to add to output files")
    parser.add_argument("--in_place", action='store_true', help="Modify files in place (backup originals)")
    parser.add_argument("--embedding_dim", type=int, default=768, help="Dimension of each embedding vector")
    return parser.parse_args()

def fix_column_names(df, embedding_dim=768):
    """Fix column names to match expected format."""
    
    # Expected total features: premise + hypothesis + delta
    expected_features = embedding_dim * 3
    
    # Get current feature columns (excluding label)
    feature_cols = [col for col in df.columns if col.startswith('feature_')]
    
    if len(feature_cols) != expected_features:
        print(f"WARNING: Expected {expected_features} features, found {len(feature_cols)}")
        print(f"This might not be the expected format!")
    
    # Create new column names
    new_columns = {}
    
    # Premise features (first embedding_dim features)
    for i in range(embedding_dim):
        if f'feature_{i}' in df.columns:
            new_columns[f'feature_{i}'] = f'premise_{i}'
    
    # Hypothesis features (second embedding_dim features)
    for i in range(embedding_dim, embedding_dim * 2):
        if f'feature_{i}' in df.columns:
            new_columns[f'feature_{i}'] = f'hypothesis_{i - embedding_dim}'
    
    # Delta features (third embedding_dim features)
    for i in range(embedding_dim * 2, embedding_dim * 3):
        if f'feature_{i}' in df.columns:
            new_columns[f'feature_{i}'] = f'delta_{i - embedding_dim * 2}'
    
    # Rename columns
    df_fixed = df.rename(columns=new_columns)
    
    print(f"Renamed {len(new_columns)} columns:")
    print(f"  - premise_0 to premise_{embedding_dim-1}")
    print(f"  - hypothesis_0 to hypothesis_{embedding_dim-1}")
    print(f"  - delta_0 to delta_{embedding_dim-1}")
    
    return df_fixed

def main():
    args = parse_args()
    
    if args.input_file:
        files_to_process = [args.input_file]
    elif args.input_dir:
        files_to_process = list(args.input_dir.glob("*.parquet"))
    else:
        print("ERROR: Must specify either --input_file or --input_dir")
        return
    
    for input_file in files_to_process:
        print(f"\n=== Processing: {input_file} ===")
        
        # Load file
        print("Loading file...")
        df = pd.read_parquet(input_file)
        print(f"Original shape: {df.shape}")
        print(f"Original columns: {df.columns.tolist()[:5]}... (showing first 5)")
        
        # Fix column names
        print("Fixing column names...")
        df_fixed = fix_column_names(df, args.embedding_dim)
        
        # Determine output path
        if args.in_place:
            # Backup original
            backup_path = input_file.with_suffix('.parquet.backup')
            if not backup_path.exists():
                print(f"Creating backup: {backup_path}")
                df.to_parquet(backup_path)
            output_path = input_file
        else:
            # Create new file with suffix
            output_path = input_file.with_stem(input_file.stem + args.output_suffix)
        
        # Save fixed file
        print(f"Saving to: {output_path}")
        df_fixed.to_parquet(output_path)
        
        # Verify the fix
        print("Verification:")
        premise_cols = [col for col in df_fixed.columns if col.startswith('premise_')]
        hypothesis_cols = [col for col in df_fixed.columns if col.startswith('hypothesis_')]
        delta_cols = [col for col in df_fixed.columns if col.startswith('delta_')]
        
        print(f"  - Premise columns: {len(premise_cols)}")
        print(f"  - Hypothesis columns: {len(hypothesis_cols)}")
        print(f"  - Delta columns: {len(delta_cols)}")
        print(f"  - Label column: {'label' in df_fixed.columns}")
        print(f"  - Total columns: {len(df_fixed.columns)}")
        
        if len(premise_cols) == args.embedding_dim and len(hypothesis_cols) == args.embedding_dim and len(delta_cols) == args.embedding_dim:
            print("✅ Column names fixed successfully!")
        else:
            print("⚠️  Column counts don't match expected format")

if __name__ == "__main__":
    main() 