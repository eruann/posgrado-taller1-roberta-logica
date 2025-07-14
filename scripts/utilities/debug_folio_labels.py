
"""
A simple utility script to load the FOLIO dataset and inspect its labels.

This helps diagnose issues with label mapping by showing the actual, raw
values present in the 'label' column of the dataset.
"""
from pathlib import Path
from datasets import load_from_disk

def inspect_labels():
    """Loads the dataset and prints unique labels."""
    dataset_path = Path("data/folio/dataset")

    if not dataset_path.exists():
        print(f"❌ Error: Dataset not found at '{dataset_path}'")
        print("Please ensure the FOLIO dataset is located correctly.")
        return

    try:
        print(f"-> Loading dataset from: {dataset_path}")
        ds = load_from_disk(str(dataset_path))
        
        if "label" not in ds.column_names:
            print(f"❌ Error: Column 'label' not found in the dataset.")
            print(f"-> Available columns are: {ds.column_names}")
            return
            
        unique_labels = set(ds["label"])
        
        print("\n✅ Inspection complete.")
        print(f"-> Unique values found in the 'label' column: {unique_labels}")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    inspect_labels() 