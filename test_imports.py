#!/usr/bin/env python
"""
Import smoke test to verify all translated files have no syntax/encoding issues.
"""
import sys
import traceback

# List of modules to test import
modules_to_test = [
    "datasets.rplang_bubble_diagram",
    "datasets.rplang_edge_semantics_simplified",
    "datasets.rplang_edge_semantics_simplified_55_106",
    "datasets.rplang_edge_semantics_simplified_56_31",
    "datasets.rplang_edge_semantics_simplified_56_32",
    "datasets.rplang_edge_semantics_simplified_78_10",
    "datasets.rplang_edge_semantics_simplified_78_10_prerunCNN",
]

def test_imports():
    """Test importing each module."""
    failed = []
    passed = []
    
    for module_name in modules_to_test:
        try:
            __import__(module_name)
            passed.append(module_name)
            print(f"✓ {module_name}")
        except Exception as e:
            failed.append((module_name, str(e)))
            print(f"✗ {module_name}: {e}")
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print(f"Results: {len(passed)} passed, {len(failed)} failed")
    print(f"{'='*60}")
    
    if failed:
        print("\nFailed imports:")
        for module_name, error in failed:
            print(f"  - {module_name}: {error}")
        return 1
    else:
        print("\n✓ All modules imported successfully!")
        return 0

if __name__ == "__main__":
    sys.exit(test_imports())
