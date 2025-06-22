#!/usr/bin/env python
"""
Debug script to identify the NaN data issue
"""
import cudf
import cupy as cp
from pathlib import Path

def check_file(filepath, name):
    """Check a parquet file for issues"""
    print(f"\n{'='*50}")
    print(f"CHECKING: {name}")
    print(f"File: {filepath}")
    print(f"{'='*50}")
    
    if not filepath.exists():
        print(f"❌ File does not exist: {filepath}")
        return False
    
    print(f"✅ File exists, size: {filepath.stat().st_size / (1024**2):.1f} MB")
    
    try:
        # Load file
        df = cudf.read_parquet(filepath)
        print(f"✅ File loaded successfully")
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {len(df.columns)} total")
        print(f"   Sample columns: {list(df.columns)[:5]}...")
        
        # Check data types
        print(f"   Data types: {df.dtypes.value_counts().to_dict()}")
        
        # Check for NaN values
        nan_mask = df.isnull()
        total_nans = nan_mask.sum().sum()
        print(f"   Total NaN values: {total_nans}")
        
        if total_nans > 0:
            # Count NaN per column
            nan_per_col = nan_mask.sum()
            nan_cols = nan_per_col[nan_per_col > 0]
            print(f"   Columns with NaN: {len(nan_cols)} out of {len(nan_per_col)}")
            print(f"   Top NaN columns: {nan_cols.head()}")
            
            # Count rows with any NaN
            rows_with_nan = nan_mask.any(axis=1).sum()
            print(f"   Rows with any NaN: {rows_with_nan} out of {len(df)}")
            
            if rows_with_nan == len(df):
                print(f"   ❌ ALL ROWS have NaN values!")
        
        # Check for Inf values
        try:
            inf_mask = df.isin([float('inf'), float('-inf')])
            total_infs = inf_mask.sum().sum()
            print(f"   Total Inf values: {total_infs}")
        except Exception as e:
            print(f"   Could not check Inf values: {e}")
        
        # Sample some actual values
        if 'label' in df.columns:
            print(f"   Label distribution: {df['label'].value_counts().head()}")
            
            # Check a few non-label columns
            feature_cols = [col for col in df.columns if col != 'label'][:3]
            for col in feature_cols:
                sample_vals = df[col].head()
                print(f"   Sample {col}: {sample_vals.tolist()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return False

def main():
    print("DIAGNOSTIC SCRIPT: Checking data files for NaN issues")
    
    # Check source embedding file
    source_file = Path("data/snli/embeddings/embeddings_snli_layer_9.parquet")
    check_file(source_file, "SOURCE EMBEDDINGS")
    
    # Check if there are any PCA output files from previous runs
    pca_files = [
        Path("data/snli/normalization_comparison/02_pca_none/50_components/pca_snli_layer9_none.parquet"),
        Path("data/snli/normalization_comparison/02_pca_none/50_components/zca_snli_layer9_none.parquet"),
        Path("data/test_pca_output.parquet"),
        Path("pca_data/test_pca_output.parquet")
    ]
    
    for pca_file in pca_files:
        if pca_file.exists():
            check_file(pca_file, f"PCA OUTPUT: {pca_file.name}")

if __name__ == "__main__":
    main() 