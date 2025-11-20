"""
Rename dataset files to numeric format required by GSDiff.
This script renames all .npy files in the specified directories to numeric format (0.npy, 1.npy, 2.npy, etc.)
"""
import os
import shutil
from pathlib import Path

def rename_files_in_directory(directory_path):
    """Rename all .npy files in a directory to numeric format."""
    if not os.path.exists(directory_path):
        print(f"Directory not found: {directory_path}")
        return
    
    # Get all .npy files
    files = [f for f in os.listdir(directory_path) if f.endswith('.npy')]
    
    if not files:
        print(f"No .npy files found in {directory_path}")
        return
    
    # Sort files to maintain consistent ordering
    files = sorted(files)
    
    print(f"\nProcessing {directory_path}")
    print(f"Found {len(files)} files to rename")
    
    # Create a mapping file to track original names
    mapping_file = os.path.join(directory_path, 'filename_mapping.txt')
    with open(mapping_file, 'w') as f:
        f.write("# Mapping of new numeric names to original names\n")
        f.write("# Format: new_name.npy -> original_name.npy\n\n")
        
        # Rename files to temporary names first to avoid conflicts
        temp_names = {}
        for idx, old_name in enumerate(files):
            old_path = os.path.join(directory_path, old_name)
            temp_name = f"temp_{idx}.npy"
            temp_path = os.path.join(directory_path, temp_name)
            shutil.move(old_path, temp_path)
            temp_names[idx] = (temp_name, old_name)
        
        # Rename from temporary to final numeric names
        for idx, (temp_name, old_name) in temp_names.items():
            temp_path = os.path.join(directory_path, temp_name)
            new_name = f"{idx}.npy"
            new_path = os.path.join(directory_path, new_name)
            shutil.move(temp_path, new_path)
            f.write(f"{new_name} -> {old_name}\n")
            print(f"  {old_name} -> {new_name}")
    
    print(f"✓ Renamed {len(files)} files")
    print(f"✓ Mapping saved to {mapping_file}")

if __name__ == '__main__':
    base_path = Path('datasets')
    
    # Directories to process
    directories = [
        base_path / 'rplang-v3-withsemantics' / 'train',
        base_path / 'rplang-v3-withsemantics' / 'val',
        base_path / 'rplang-v3-withsemantics' / 'test',
        base_path / 'rplang-v3-withsemantics-withboundary' / 'train',
        base_path / 'rplang-v3-withsemantics-withboundary' / 'val',
        base_path / 'rplang-v3-withsemantics-withboundary' / 'test',
    ]
    
    print("=" * 60)
    print("Renaming dataset files to numeric format")
    print("=" * 60)
    
    for directory in directories:
        rename_files_in_directory(str(directory))
    
    print("\n" + "=" * 60)
    print("✓ All files renamed successfully!")
    print("=" * 60)
    print("\nNote: Original filename mappings are saved in 'filename_mapping.txt' in each directory.")
