"""
Path utility for GSDiff project to handle dataset paths consistently.
This ensures paths work regardless of where scripts are run from.
"""
import os

def get_datasets_dir():
    """
    Get the absolute path to the datasets directory.
    Works from anywhere in the project.
    """
    # Get the directory of this file (datasets/)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # current_dir is already the datasets folder
    return current_dir

def get_project_root():
    """
    Get the absolute path to the project root directory.
    """
    # Get the directory of this file (datasets/)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level to get project root
    return os.path.dirname(current_dir)

def get_data_path(*path_parts):
    """
    Get an absolute path to a data file/directory.
    
    Args:
        *path_parts: Path components relative to the datasets directory
        
    Example:
        get_data_path('rplang-v3-withsemantics', 'train')
        returns: /absolute/path/to/datasets/rplang-v3-withsemantics/train
    """
    return os.path.join(get_datasets_dir(), *path_parts)
