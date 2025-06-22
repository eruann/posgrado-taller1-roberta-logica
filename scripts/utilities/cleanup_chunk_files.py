#!/usr/bin/env python3
"""
scripts/cleanup_chunk_files.py
==============================
Utility script to find and remove leftover chunk files from failed normalization runs.
"""

import argparse
from pathlib import Path
import sys

def find_chunk_files(base_dir):
    """Find all chunk files in the directory structure"""
    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"Directory {base_dir} does not exist")
        return []
    
    # Look for chunk files with pattern: *_chunk_*.parquet
    chunk_files = list(base_path.rglob("*_chunk_*.parquet"))
    
    # Also look for temp files
    temp_files = list(base_path.rglob("*.temp.parquet"))
    
    return chunk_files + temp_files

def main():
    parser = argparse.ArgumentParser(description="Clean up leftover chunk files")
    parser.add_argument("--base_dir", required=True, help="Base directory to search for chunk files")
    parser.add_argument("--dry_run", action="store_true", help="Show what would be deleted without actually deleting")
    parser.add_argument("--confirm", action="store_true", help="Skip confirmation prompt")
    
    args = parser.parse_args()
    
    print(f"Searching for chunk files in: {args.base_dir}")
    chunk_files = find_chunk_files(args.base_dir)
    
    if not chunk_files:
        print("✓ No chunk files found")
        return
    
    print(f"Found {len(chunk_files)} chunk/temp files:")
    total_size = 0
    for file_path in chunk_files:
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            total_size += size_mb
            print(f"  {file_path} ({size_mb:.1f} MB)")
        else:
            print(f"  {file_path} (missing)")
    
    print(f"\nTotal size: {total_size:.1f} MB")
    
    if args.dry_run:
        print("\n[DRY RUN] Would delete the above files")
        return
    
    if not args.confirm:
        response = input(f"\nDelete {len(chunk_files)} files? [y/N]: ")
        if response.lower() not in ['y', 'yes']:
            print("Cancelled")
            return
    
    # Delete files
    deleted_count = 0
    deleted_size = 0
    for file_path in chunk_files:
        try:
            if file_path.exists():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                file_path.unlink()
                deleted_count += 1
                deleted_size += size_mb
                print(f"Deleted: {file_path}")
            else:
                print(f"Already gone: {file_path}")
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")
    
    print(f"\n✓ Deleted {deleted_count}/{len(chunk_files)} files ({deleted_size:.1f} MB freed)")

if __name__ == "__main__":
    main() 