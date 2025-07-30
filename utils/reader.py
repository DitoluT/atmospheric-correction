"""
Band reading and loading utilities for Sentinel-2 L1C data.
Handles .npy format files containing pre-stacked band data.
"""

import numpy as np
import os
from typing import Dict, Tuple, Any
import config


def read_band_stack(scene_path: str) -> Dict[str, Any]:
    """
    Read Sentinel-2 L1C band stack from .npy file.
    
    Args:
        scene_path (str): Path to .npy file containing band stack
        
    Returns:
        dict: Dictionary containing:
            - metadata: Basic file information
            - bands: Dictionary of band_name -> array
            - maskable_bands: Resampled bands for s2cloudless (10 bands)
    """
    
    if not os.path.exists(scene_path):
        raise FileNotFoundError(f"Scene file not found: {scene_path}")
    
    # Load the .npy file
    try:
        band_stack = np.load(scene_path)
    except Exception as e:
        raise ValueError(f"Failed to load .npy file: {e}")
    
    # Validate dimensions
    if len(band_stack.shape) != 3:
        raise ValueError(f"Expected 3D array, got shape {band_stack.shape}")
    
    height, width, num_bands = band_stack.shape
    
    if num_bands != len(config.ALL_BANDS):
        raise ValueError(f"Expected {len(config.ALL_BANDS)} bands, got {num_bands}")
    
    # Create metadata
    metadata = {
        "file_path": scene_path,
        "file_name": os.path.basename(scene_path),
        "shape": band_stack.shape,
        "height": height,
        "width": width,
        "num_bands": num_bands,
        "dtype": str(band_stack.dtype)
    }
    
    # Create bands dictionary
    bands = {}
    for i, band_name in enumerate(config.ALL_BANDS):
        bands[band_name] = band_stack[:, :, i].astype(np.float32)
    
    # Extract bands needed for s2cloudless (10 bands in specific order)
    maskable_bands = np.zeros((height, width, len(config.CLOUD_BANDS)), dtype=np.float32)
    for i, band_name in enumerate(config.CLOUD_BANDS):
        band_idx = config.BAND_INDICES[band_name]
        maskable_bands[:, :, i] = band_stack[:, :, band_idx].astype(np.float32)
    
    return {
        "metadata": metadata,
        "bands": bands,
        "maskable_bands": maskable_bands
    }


def validate_band_data(bands: Dict[str, np.ndarray]) -> bool:
    """
    Validate that band data is consistent and within expected ranges.
    
    Args:
        bands (dict): Dictionary of band_name -> array
        
    Returns:
        bool: True if validation passes
    """
    
    if not bands:
        return False
    
    # Check that all required bands are present
    required_bands = set(config.CORRECTION_BANDS + config.CLOUD_BANDS)
    available_bands = set(bands.keys())
    
    if not required_bands.issubset(available_bands):
        missing = required_bands - available_bands
        raise ValueError(f"Missing required bands: {missing}")
    
    # Check that all bands have the same shape
    shapes = [band.shape for band in bands.values()]
    if not all(shape == shapes[0] for shape in shapes):
        raise ValueError("All bands must have the same spatial dimensions")
    
    # Check for reasonable reflectance values (0-10000 for Sentinel-2 L1C)
    for band_name, band_data in bands.items():
        if np.any(band_data < 0) or np.any(band_data > 15000):
            print(f"Warning: Band {band_name} has values outside expected range [0, 15000]")
    
    return True


def get_scene_info(scene_path: str) -> Dict[str, Any]:
    """
    Get basic information about a scene without loading the full data.
    
    Args:
        scene_path (str): Path to .npy file
        
    Returns:
        dict: Scene information
    """
    
    if not os.path.exists(scene_path):
        raise FileNotFoundError(f"Scene file not found: {scene_path}")
    
    # Load just to get shape info
    try:
        band_stack = np.load(scene_path, mmap_mode='r')  # Memory-mapped for efficiency
        shape = band_stack.shape
    except Exception as e:
        raise ValueError(f"Failed to read scene info: {e}")
    
    return {
        "file_path": scene_path,
        "file_name": os.path.basename(scene_path),
        "shape": shape,
        "file_size_mb": os.path.getsize(scene_path) / (1024 * 1024)
    }