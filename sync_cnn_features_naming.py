"""
Sync CNN feature map filenames with the renamed graph dataset files.
This reads the mapping files from the graph directories and applies the same naming to CNN features.
"""
import os
import shutil
from pathlib import Path

def sync_cnn_features_with_mapping(train_mapping_file, val_mapping_file, test_mapping_file, cnn_dir):
    """Rename CNN feature files to match the graph dataset numeric names."""
    
    # Parse mapping files
    def parse_mapping(mapping_file):
        """Parse a mapping file and return dict of {original_name: new_numeric_name}"""
        mapping = {}
        if not os.path.exists(mapping_file):
            print(f"Warning: Mapping file not found: {mapping_file}")
            return mapping
        
        with open(mapping_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '->' in line:
                    parts = line.split(' -> ')
                    if len(parts) == 2:
                        new_name = parts[0].strip()  # e.g., "0.npy"
                        old_name = parts[1].strip()  # e.g., "01-1st-FFL_apartment_3_vflip.npy"
                        mapping[old_name] = new_name
        return mapping
    
    # Get all mappings
    train_mapping = parse_mapping(train_mapping_file)
    val_mapping = parse_mapping(val_mapping_file)
    test_mapping = parse_mapping(test_mapping_file)
    
    # Combine all mappings
    all_mappings = {**train_mapping, **val_mapping, **test_mapping}
    
    print(f"\nTotal mappings found: {len(all_mappings)}")
    print(f"  Train: {len(train_mapping)}")
    print(f"  Val: {len(val_mapping)}")
    print(f"  Test: {len(test_mapping)}")
    
    # Get all CNN feature files
    if not os.path.exists(cnn_dir):
        print(f"Error: CNN directory not found: {cnn_dir}")
        return
    
    cnn_files = [f for f in os.listdir(cnn_dir) if f.endswith('.npy')]
    print(f"\nCNN feature files found: {len(cnn_files)}")
    
    # Rename CNN files based on mapping
    renamed_count = 0
    not_found_count = 0
    
    # First, rename to temporary names to avoid conflicts
    temp_mapping = {}
    for cnn_file in cnn_files:
        if cnn_file in all_mappings:
            new_name = all_mappings[cnn_file]
            temp_name = f"temp_{renamed_count}_{new_name}"
            old_path = os.path.join(cnn_dir, cnn_file)
            temp_path = os.path.join(cnn_dir, temp_name)
            shutil.move(old_path, temp_path)
            temp_mapping[temp_name] = new_name
            renamed_count += 1
        else:
            not_found_count += 1
            print(f"  Warning: No mapping found for: {cnn_file}")
    
    # Then rename from temporary to final names
    for temp_name, final_name in temp_mapping.items():
        temp_path = os.path.join(cnn_dir, temp_name)
        final_path = os.path.join(cnn_dir, final_name)
        shutil.move(temp_path, final_path)
    
    print(f"\n✓ Renamed {renamed_count} CNN feature files")
    if not_found_count > 0:
        print(f"⚠ {not_found_count} files had no mapping (kept original names)")

if __name__ == '__main__':
    base_path = Path('datasets')
    
    # Mapping files from the graph directories
    train_mapping = base_path / 'rplang-v3-withsemantics-withboundary' / 'train' / 'filename_mapping.txt'
    val_mapping = base_path / 'rplang-v3-withsemantics-withboundary' / 'val' / 'filename_mapping.txt'
    test_mapping = base_path / 'rplang-v3-withsemantics-withboundary' / 'test' / 'filename_mapping.txt'
    
    # CNN feature directory
    cnn_dir = base_path / 'prerunning_cnn_featuremaps'
    
    print("=" * 60)
    print("Syncing CNN feature map filenames with graph dataset")
    print("=" * 60)
    
    sync_cnn_features_with_mapping(str(train_mapping), str(val_mapping), str(test_mapping), str(cnn_dir))
    
    print("\n" + "=" * 60)
    print("✓ Sync complete!")
    print("=" * 60)
