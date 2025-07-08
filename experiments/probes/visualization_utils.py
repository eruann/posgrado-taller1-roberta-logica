#!/usr/bin/env python
"""
experiments/probes/visualization_utils.py
=========================================
Utilities for visualizing a trained decision tree probe.
Requires graphviz to be installed on the system.
"""
import subprocess
from pathlib import Path
from sklearn.tree import export_graphviz

def plot_tree_to_file(
    model, 
    feature_names: list, 
    class_names: list, 
    output_dir: Path,
    filename_prefix: str,
    max_depth: int = 3
) -> Path:
    """
    Exports the decision tree to a .dot file and renders it to a .png file.

    Args:
        model: The trained cuML DecisionTreeClassifier.
        feature_names: List of names for the features.
        class_names: List of names for the target classes.
        output_dir: The directory to save the files in.
        filename_prefix: Prefix for the output filenames.
        max_depth: The maximum depth of the tree to visualize.

    Returns:
        The path to the generated PNG file.
    """
    dot_path = output_dir / f"{filename_prefix}_tree.dot"
    png_path = output_dir / f"{filename_prefix}_tree.png"
    
    print(f"Exporting tree visualization (max_depth={max_depth}) to {png_path}...")
    
    try:
        export_graphviz(
            model,
            out_file=str(dot_path),
            feature_names=feature_names,
            class_names=class_names,
            filled=True,
            rounded=True,
            special_characters=True,
            max_depth=max_depth
        )
    except Exception as e:
        print(f"⚠️  Could not generate .dot file with cuML's export_graphviz: {e}")
        return None

    # Render the .dot file to a .png file using system's graphviz
    try:
        subprocess.run(
            ["dot", "-Tpng", str(dot_path), "-o", str(png_path)],
            check=True,
            capture_output=True,
            text=True
        )
        print(f"✓ Tree plot saved to {png_path}")
        # Clean up the intermediate .dot file
        dot_path.unlink()
    except FileNotFoundError:
        print("✗ ERROR: `dot` command not found. Is Graphviz installed and in your PATH?")
        print(f"      Intermediate .dot file saved at: {dot_path}")
        return None
    except subprocess.CalledProcessError as e:
        print(f"✗ ERROR: Graphviz failed to render the tree.")
        print(f"      Stderr: {e.stderr}")
        print(f"      Intermediate .dot file saved at: {dot_path}")
        return None
        
    return png_path 