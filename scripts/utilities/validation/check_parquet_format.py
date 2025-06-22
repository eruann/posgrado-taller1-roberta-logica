#!/usr/bin/env python
"""
Checks the format of a Parquet file to determine if it's 'wide' or 'narrow'.

- 'Narrow' format has a single 'vector' column containing lists.
- 'Wide' format has multiple 'feature_i' columns.

Usage:
    python scripts/check_parquet_format.py --path /path/to/your/file.parquet
"""
import argparse
from pathlib import Path
import pyarrow.parquet as pq

def check_format(file_path: Path):
    """Reads the schema of a Parquet file and checks its format."""
    if not file_path.exists():
        print(f"Error: File not found at {file_path}")
        return

    try:
        schema = pq.read_schema(file_path)
        column_names = {name for name in schema.names}

        if "vector" in column_names:
            print(f"✅ [{file_path.name}]: NARROW format (contains a 'vector' column).")
        elif any(name.startswith("feature_") for name in column_names):
            # Check for a representative feature column
            if "feature_0" in column_names:
                 print(f"✅ [{file_path.name}]: WIDE format (contains 'feature_i' columns).")
            else:
                print(f"🤔 [{file_path.name}]: Looks like a WIDE format, but 'feature_0' is missing.")

        else:
            print(f"❓ [{file_path.name}]: UNKNOWN format. Columns found: {sorted(list(column_names))}")

    except Exception as e:
        print(f"Error reading schema for {file_path.name}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Check if a Parquet file is in 'wide' or 'narrow' format.")
    parser.add_argument("--path", type=Path, required=True, help="Path to the Parquet file to check.")
    args = parser.parse_args()
    check_format(args.path)

if __name__ == "__main__":
    main() 