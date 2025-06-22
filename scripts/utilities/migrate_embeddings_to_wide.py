#!/usr/bin/env python
"""
Orchestrates the one-time migration of embedding files from 'narrow' to 'wide' format.

This script will:
1.  Identify target directories (e.g., 'embeddings', 'delta_embeddings').
2.  Create a backup sub-directory for the original 'narrow' files.
3.  Move the original files to the backup location.
4.  Call the conversion utility to transform each backup file into the 'wide'
    format, saving it back to the original location.

This ensures a safe, repeatable migration process.

Usage:
    # Run the migration for the default 'data/snli' directory
    python scripts/migrate_embeddings_to_wide.py

    # Specify a different base directory
    python scripts/migrate_embeddings_to_wide.py --base_dir data/folio
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

# -- Path Hack --
# Add the project root to sys.path to allow for absolute imports
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
# ---------------

# The conversion script must be available in the specified path
from scripts.experimental.convert_to_wide_format import convert_to_wide


def migrate_directory(base_dir: Path, dir_name: str):
    """
    Performs the backup and narrow-to-wide conversion for a single directory.
    
    Args:
        base_dir: The parent directory containing the target embeddings (e.g., 'data/snli').
        dir_name: The name of the directory to process (e.g., 'embeddings').
    """
    source_dir = base_dir / dir_name
    backup_dir = base_dir / f"{dir_name}_narrow_backup"

    if not source_dir.exists():
        print(f"⏩ Directory not found: {source_dir}. Skipping.")
        return

    # Find parquet files to migrate
    files_to_migrate = list(source_dir.glob("*.parquet"))
    if not files_to_migrate:
        print(f"No .parquet files found in {source_dir}. Nothing to do.")
        return

    print(f"\n--- Migrating directory: {source_dir} ---")

    # Create backup directory
    print(f"Creating backup directory at: {backup_dir}")
    backup_dir.mkdir(parents=True, exist_ok=True)

    # Move files to backup
    print(f"Moving {len(files_to_migrate)} files to backup directory...")
    for f in files_to_migrate:
        try:
            shutil.move(str(f), str(backup_dir / f.name))
        except Exception as e:
            print(f"Could not move {f.name}: {e}. Halting migration for this directory.")
            return

    # Run conversion on backed-up files
    backed_up_files = sorted(list(backup_dir.glob("*.parquet")))
    print(f"Starting conversion for {len(backed_up_files)} files...")
    for source_file in backed_up_files:
        dest_file = source_dir / source_file.name
        print(f"  Converting {source_file.name} -> {dest_file.name}")
        # Let exceptions propagate up to halt the script on any failure
        convert_to_wide(source_file, dest_file)
    
    print(f"--- Migration complete for: {source_dir} ---")


def main():
    parser = argparse.ArgumentParser(description="Migrate Parquet embeddings from narrow to wide format.")
    parser.add_argument(
        "--base_dir",
        type=Path,
        default=Path("data/snli"),
        help="Base directory containing embedding folders (e.g., 'data/snli')."
    )
    parser.add_argument(
        "--dirs_to_process",
        nargs='+',  # This allows for one or more arguments
        default=["embeddings", "delta_embeddings", "difference_embeddings"],
        help="A list of specific sub-directory names to process (e.g., 'embeddings' 'difference_embeddings')."
    )
    args = parser.parse_args()

    # Directories to process are now taken from the command line
    target_dirs = args.dirs_to_process
    
    print(f"Starting migration for base directory: {args.base_dir}")
    print(f"Processing target directories: {target_dirs}")
    try:
        for dir_name in target_dirs:
            migrate_directory(args.base_dir, dir_name)
        print("\n✅ All migration tasks finished successfully.")
    except Exception as e:
        print(f"\n❌ MIGRATION FAILED: An error occurred: {e}")
        print("   The process has been halted. Please check the error message and resolve the issue before re-running.")
        sys.exit(1)


if __name__ == "__main__":
    main() 