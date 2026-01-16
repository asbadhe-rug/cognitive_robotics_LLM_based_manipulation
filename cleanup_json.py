#!/usr/bin/env python3
"""
Quick script to clean up corrupted JSON files.
Run this if you get JSONDecodeError.
"""

import os
import shutil
from datetime import datetime

def cleanup_corrupted_files():
    """Backup and remove corrupted JSON files."""
    
    files_to_check = [
        "coordinate_metrics.json",
        "aggregate_stats.json",
        "overall_aggregate.json"
    ]
    
    print("="*60)
    print("JSON FILE CLEANUP")
    print("="*60)
    
    for filename in files_to_check:
        if not os.path.exists(filename):
            print(f"✓ {filename}: Not found (OK)")
            continue
        
        try:
            import json
            with open(filename, 'r') as f:
                data = json.load(f)
            print(f"✓ {filename}: Valid ({len(data) if isinstance(data, list) else 1} entries)")
        except json.JSONDecodeError as e:
            print(f"✗ {filename}: CORRUPTED")
            print(f"  Error: {e}")
            
            # Create backup
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_name = f"{filename}.backup_{timestamp}"
            
            try:
                shutil.copy(filename, backup_name)
                print(f"  ✓ Backed up to: {backup_name}")
            except Exception as backup_error:
                print(f"  ✗ Backup failed: {backup_error}")
            
            # Remove corrupted file
            try:
                os.remove(filename)
                print(f"  ✓ Removed corrupted file")
            except Exception as remove_error:
                print(f"  ✗ Remove failed: {remove_error}")
    
    print("\n" + "="*60)
    print("CLEANUP COMPLETE")
    print("="*60)
    print("\nYou can now run your experiments again.")
    print("Corrupted files have been backed up with .backup_ prefix")


if __name__ == "__main__":
    cleanup_corrupted_files()