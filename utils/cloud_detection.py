"""
Cloud detection using s2cloudless for Sentinel-2 imagery.
Generates cloud probability maps and binary cloud masks.
"""

import numpy as np
from typing import Tuple
import config

try:
    from s2cloudless import S2PixelCloudDetector
except ImportError:
    print("Warning: s2cloudless not available. Install with: pip install s2cloudless")
    S2PixelCloudDetector = None


def preprocess_for_s2cloudless(band_stack: np.ndarray) -> np.ndarray:
    """
    Preprocess band stack for s2cloudless input requirements.
    s2cloudless expects values in range [0, 1] and specific band ordering.
    
    Args:
        band_stack (np.ndarray): Band stack with shape (H, W, 10) for cloud bands
        
    Returns:
        np.ndarray: Preprocessed band stack ready for s2cloudless
    """
    
    # Ensure we have the right number of bands
    if band_stack.shape[2] != len(config.CLOUD_BANDS):
        raise ValueError(f"Expected {len(config.CLOUD_BANDS)} bands, got {band_stack.shape[2]}")
    
    # Convert to float32 and normalize to [0, 1] range
    # Sentinel-2 L1C values are typically 0-10000
    processed = band_stack.astype(np.float32) / config.REFLECTANCE_SCALE
    
    # Clip to [0, 1] range to handle any outliers
    processed = np.clip(processed, 0, 1)
    
    return processed


def generate_cloud_mask(band_stack: np.ndarray, 
                       threshold: float = None,
                       average_over: int = 4,
                       dilation_size: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate cloud probability maps and binary cloud masks using s2cloudless.
    
    Args:
        band_stack (np.ndarray): Band stack with shape (H, W, 10) in s2cloudless order
        threshold (float, optional): Cloud probability threshold. Defaults to config.CLOUD_THRESHOLD
        average_over (int): Averaging factor for cloud detection
        dilation_size (int): Dilation size for morphological operations
        
    Returns:
        tuple: (cloud_probability_map, binary_cloud_mask)
    """
    
    if S2PixelCloudDetector is None:
        raise ImportError("s2cloudless package is required for cloud detection")
    
    if threshold is None:
        threshold = config.CLOUD_THRESHOLD
    
    # Preprocess the data
    processed_bands = preprocess_for_s2cloudless(band_stack)
    
    # Initialize the cloud detector
    # s2cloudless expects exactly 10 bands, so we set all_bands=False
    cloud_detector = S2PixelCloudDetector(
        threshold=threshold,
        average_over=average_over,
        dilation_size=dilation_size,
        all_bands=False  # We provide exactly the 10 bands it needs
    )
    
    # s2cloudless expects input with batch dimension: (batch, H, W, bands)
    batch_input = processed_bands[np.newaxis, ...]
    
    try:
        # Get cloud probability maps
        cloud_probs = cloud_detector.get_cloud_probability_maps(batch_input)
        
        # Get binary cloud masks
        cloud_masks = cloud_detector.get_cloud_masks(batch_input)
        
        # Remove batch dimension
        cloud_prob_map = cloud_probs[0]  # Shape: (H, W)
        binary_cloud_mask = cloud_masks[0]  # Shape: (H, W)
        
    except Exception as e:
        raise RuntimeError(f"Cloud detection failed: {e}")
    
    return cloud_prob_map, binary_cloud_mask


def create_conservative_cloud_mask(cloud_prob: np.ndarray, 
                                 threshold: float = None,
                                 buffer_pixels: int = 2) -> np.ndarray:
    """
    Create a conservative (dilated) cloud mask for safer atmospheric correction.
    
    Args:
        cloud_prob (np.ndarray): Cloud probability map (0-1)
        threshold (float): Probability threshold for cloud detection
        buffer_pixels (int): Number of pixels to dilate the mask
        
    Returns:
        np.ndarray: Conservative binary cloud mask
    """
    
    if threshold is None:
        threshold = config.CLOUD_THRESHOLD
    
    # Create initial binary mask
    binary_mask = cloud_prob > threshold
    
    # Apply morphological dilation for buffer
    if buffer_pixels > 0:
        from scipy.ndimage import binary_dilation
        structure = np.ones((2*buffer_pixels+1, 2*buffer_pixels+1))
        binary_mask = binary_dilation(binary_mask, structure=structure)
    
    return binary_mask.astype(np.bool_)


def cloud_coverage_stats(cloud_prob: np.ndarray, cloud_mask: np.ndarray) -> dict:
    """
    Calculate cloud coverage statistics for the scene.
    
    Args:
        cloud_prob (np.ndarray): Cloud probability map
        cloud_mask (np.ndarray): Binary cloud mask
        
    Returns:
        dict: Statistics including coverage percentage, mean probability, etc.
    """
    
    total_pixels = cloud_prob.size
    cloud_pixels = np.sum(cloud_mask)
    cloud_coverage = (cloud_pixels / total_pixels) * 100
    
    mean_cloud_prob = np.mean(cloud_prob)
    max_cloud_prob = np.max(cloud_prob)
    
    # Statistics for cloud pixels only
    cloud_areas = cloud_prob[cloud_mask]
    mean_cloud_prob_in_clouds = np.mean(cloud_areas) if len(cloud_areas) > 0 else 0
    
    return {
        "total_pixels": total_pixels,
        "cloud_pixels": int(cloud_pixels),
        "cloud_coverage_percent": cloud_coverage,
        "mean_cloud_probability": float(mean_cloud_prob),
        "max_cloud_probability": float(max_cloud_prob),
        "mean_probability_in_clouds": float(mean_cloud_prob_in_clouds)
    }