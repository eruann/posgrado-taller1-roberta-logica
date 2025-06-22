#!/usr/bin/env python
"""
scripts/slice_parquet_vectors.py - Slices the first 'n' components from vectors in a Parquet file using GPU.
=========================================================================================================

This script loads vectors from a Parquet file, removes the first 'n' components
from each vector using cudf for GPU acceleration, and saves the resulting shorter
vectors to a new Parquet file.

Uso:
-----
python scripts/slice_parquet_vectors.py \\
       --input_parquet data/pca_vectors.parquet \\
       --output_parquet data/sliced_pca_vectors.parquet \\
       --skip_first_n 5

Parámetros:
----------
--input_parquet: Path to the input Parquet file. Must contain a 'vector' column
                 where each entry is a list/array of numerical features, and ideally a 'label' column.
--output_parquet: Path to save the output Parquet file with sliced vectors.
--skip_first_n:   Integer, the number of initial components to remove from each vector.
"""

import argparse
import cudf
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Slice first 'n' components from vectors in a Parquet file using GPU.")
    parser.add_argument("--input_parquet", required=True, type=Path,
                        help="Path to the input Parquet file (must contain 'vector' and ideally 'label' columns).")
    parser.add_argument("--output_parquet", required=True, type=Path,
                        help="Path to save the output Parquet file with sliced vectors.")
    parser.add_argument("--skip_first_n", required=True, type=int,
                        help="Number of initial components to remove from each vector.")
    return parser.parse_args()



def main():
    args = parse_args()

    if not args.input_parquet.exists():
        raise FileNotFoundError(f"Error: Input Parquet file not found at {args.input_parquet}")

    if args.skip_first_n < 0:
        raise ValueError("Error: --skip_first_n must be a non-negative integer.")

    print(f"Loading Parquet file into GPU memory from: {args.input_parquet}")
    gdf = cudf.read_parquet(args.input_parquet)

    # Separate labels from feature columns
    labels = gdf['label']
    feature_cols = [col for col in gdf.columns if col != 'label']
    
    original_dims = len(feature_cols)
    print(f"Original vector dimensions: {original_dims}")

    if args.skip_first_n >= original_dims:
        raise ValueError(f"Error: skip_first_n ({args.skip_first_n}) is greater than or equal to "
                         f"the vector dimension ({original_dims}). Cannot slice all components.")

    # Select columns to keep
    cols_to_keep = feature_cols[args.skip_first_n:]
    new_dims = len(cols_to_keep)
    print(f"Slicing off the first {args.skip_first_n} components. New vector dimensions: {new_dims}")

    # Create the new DataFrame with the sliced features and the label
    output_gdf = gdf[cols_to_keep]
    output_gdf['label'] = labels

    # Create simple sliced component visualization
    try:
        # Create a simple plot showing which components remain
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        # Show all original components
        all_components = np.arange(original_dims)
        colors = ['red' if i < args.skip_first_n else 'green' for i in all_components]
        
        # Calculate variance for visualization (simple approach)
        sample_data = gdf[feature_cols].head(min(5000, len(gdf))).values
        variances = np.var(cp.asnumpy(sample_data), axis=0)
        
        bars = ax.bar(all_components, variances, color=colors, alpha=0.7)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', alpha=0.7, label=f'Removed ({args.skip_first_n} components)'),
            Patch(facecolor='green', alpha=0.7, label=f'Kept ({new_dims} components)')
        ]
        ax.legend(handles=legend_elements)
        
        ax.set_xlabel('Component Index')
        ax.set_ylabel('Component Variance')
        ax.set_title(f'Component Slicing: Skip First {args.skip_first_n} Components')
        ax.grid(True, alpha=0.3)
        
        # Add summary text
        total_var = np.sum(variances)
        kept_var = np.sum(variances[args.skip_first_n:])
        removed_var = np.sum(variances[:args.skip_first_n])
        
        ax.text(0.02, 0.98, 
                f'Total variance: 100%\nKept: {kept_var/total_var*100:.1f}%\nRemoved: {removed_var/total_var*100:.1f}%',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Save plot
        plot_path = args.output_parquet.parent / f"sliced_components_skip_{args.skip_first_n}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Slicing visualization saved: {plot_path}")
        
    except Exception as e:
        print(f"⚠️  Failed to create slicing visualization: {e}")

    # Save the result
    args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving sliced vectors to: {args.output_parquet}")
    output_gdf.to_parquet(args.output_parquet)

    print("\n--- Summary ---")
    print(f"Successfully processed {len(gdf)} vectors.")
    print(f"Original vector dimensions: {original_dims}")
    print(f"Number of components skipped: {args.skip_first_n}")
    print(f"New vector dimensions: {new_dims}")
    print(f"Output saved to {args.output_parquet}")
    print("-----------------")

if __name__ == "__main__":
    main() 