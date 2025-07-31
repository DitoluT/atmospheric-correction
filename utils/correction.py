"""
Atmospheric correction utilities implementing Cloud-Optimized Dark Object Subtraction (CO-DOS).
Provides lightweight atmospheric correction for Sentinel-2 L1C imagery.
"""

import numpy as np
from typing import Dict, Optional
import config


def apply_atmospheric_correction(band_data: Dict[str, np.ndarray], 
                               cloud_mask: np.ndarray,
                               percentile: float = None) -> Dict[str, np.ndarray]:
    """
    Apply Cloud-Optimized Dark Object Subtraction (CO-DOS) atmospheric correction.
    
    Args:
        band_data (dict): Dictionary of band_name -> array
        cloud_mask (np.ndarray): Binary cloud mask (True = cloud)
        percentile (float): Percentile for dark object detection (default: config.DOS_PERCENTILE)
        
    Returns:
        dict: Dictionary of corrected bands
    """
    
    if percentile is None:
        percentile = config.DOS_PERCENTILE
    
    # Create cloud-free pixel mask
    cloud_free_mask = ~cloud_mask
    
    # Check if we have enough cloud-free pixels
    cloud_free_pixels = np.sum(cloud_free_mask)
    total_pixels = cloud_mask.size
    cloud_free_ratio = cloud_free_pixels / total_pixels
    
    if cloud_free_ratio < 0.1:  # Less than 10% cloud-free
        print(f"Warning: Only {cloud_free_ratio:.1%} cloud-free pixels available for correction")
    
    corrected_bands = {}
    
    # Process each correction band
    for band_name in config.CORRECTION_BANDS:
        if band_name not in band_data:
            print(f"Warning: Band {band_name} not found in input data")
            continue
            
        band_array = band_data[band_name].copy()
        
        # Extract cloud-free pixels for dark object calculation
        cloud_free_pixels = band_array[cloud_free_mask]
        
        if len(cloud_free_pixels) == 0:
            print(f"Warning: No cloud-free pixels for band {band_name}, skipping correction")
            corrected_bands[band_name] = band_array / config.REFLECTANCE_SCALE
            continue
        
        # Calculate dark object value (low percentile of cloud-free pixels)
        dos_value = np.percentile(cloud_free_pixels, percentile * 100)
        
        # Apply dark object subtraction
        corrected = band_array - dos_value
        
        # Clip negative values to 0
        corrected = np.clip(corrected, 0, None)
        
        # Apply scale factor to convert to reflectance [0, 1]
        corrected = corrected / config.REFLECTANCE_SCALE
        
        corrected_bands[band_name] = corrected.astype(np.float32)
    
    return corrected_bands


def estimate_atmospheric_path_radiance(band_data: Dict[str, np.ndarray],
                                     cloud_mask: np.ndarray,
                                     water_mask: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Estimate atmospheric path radiance for each band using dark objects.
    
    Args:
        band_data (dict): Dictionary of band_name -> array
        cloud_mask (np.ndarray): Binary cloud mask
        water_mask (np.ndarray, optional): Water body mask for better dark object selection
        
    Returns:
        dict: Estimated path radiance values per band
    """
    
    cloud_free_mask = ~cloud_mask
    
    # If water mask is available, prefer water bodies for dark object detection
    if water_mask is not None:
        dark_object_mask = cloud_free_mask & water_mask
        if np.sum(dark_object_mask) < 100:  # Fallback if not enough water pixels
            dark_object_mask = cloud_free_mask
    else:
        dark_object_mask = cloud_free_mask
    
    path_radiance = {}
    
    for band_name in config.CORRECTION_BANDS:
        if band_name not in band_data:
            continue
            
        band_array = band_data[band_name]
        dark_pixels = band_array[dark_object_mask]
        
        if len(dark_pixels) > 0:
            # Use 1st percentile as estimate of path radiance
            path_radiance[band_name] = float(np.percentile(dark_pixels, 1.0))
        else:
            path_radiance[band_name] = 0.0
    
    return path_radiance


def apply_simple_rayleigh_correction(band_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Apply simplified Rayleigh scattering correction based on wavelength.
    This is a basic correction that can be applied before or after DOS.
    
    Args:
        band_data (dict): Dictionary of band_name -> array
        
    Returns:
        dict: Rayleigh-corrected bands
    """
    
    # Approximate Rayleigh optical thickness at sea level for Sentinel-2 bands
    rayleigh_coeffs = {
        'B01': 0.52,   # 443 nm
        'B02': 0.33,   # 490 nm
        'B03': 0.23,   # 560 nm
        'B04': 0.18,   # 665 nm
        'B05': 0.11,   # 705 nm
        'B06': 0.09,   # 740 nm
        'B07': 0.08,   # 783 nm
        'B08': 0.07,   # 842 nm
        'B8A': 0.06,   # 865 nm
        'B09': 0.04,   # 945 nm
        'B10': 0.03,   # 1375 nm (cirrus)
        'B11': 0.01,   # 1610 nm
        'B12': 0.005   # 2190 nm
    }
    
    corrected_bands = {}
    
    for band_name, band_array in band_data.items():
        if band_name in rayleigh_coeffs:
            # Simple exponential correction: R_corrected = R_observed * exp(tau_rayleigh)
            correction_factor = np.exp(rayleigh_coeffs[band_name])
            corrected_bands[band_name] = band_array * correction_factor
        else:
            corrected_bands[band_name] = band_array.copy()
    
    return corrected_bands


def quality_assessment(original_bands: Dict[str, np.ndarray],
                      corrected_bands: Dict[str, np.ndarray],
                      cloud_mask: np.ndarray) -> Dict[str, dict]:
    """
    Assess the quality of atmospheric correction by comparing before/after statistics.
    
    Args:
        original_bands (dict): Original TOA reflectance bands
        corrected_bands (dict): Atmospherically corrected bands
        cloud_mask (np.ndarray): Cloud mask
        
    Returns:
        dict: Quality assessment metrics per band
    """
    
    cloud_free_mask = ~cloud_mask
    assessment = {}
    
    for band_name in corrected_bands:
        if band_name not in original_bands:
            continue
            
        orig = original_bands[band_name][cloud_free_mask]
        corr = corrected_bands[band_name][cloud_free_mask]
        
        if len(orig) == 0:
            continue
        
        # Calculate statistics
        orig_mean = np.mean(orig)
        corr_mean = np.mean(corr)
        orig_std = np.std(orig)
        corr_std = np.std(corr)
        
        # Calculate correction magnitude
        correction_magnitude = orig_mean - corr_mean
        relative_correction = correction_magnitude / orig_mean if orig_mean > 0 else 0
        
        assessment[band_name] = {
            "original_mean": float(orig_mean),
            "corrected_mean": float(corr_mean),
            "original_std": float(orig_std),
            "corrected_std": float(corr_std),
            "correction_magnitude": float(correction_magnitude),
            "relative_correction_percent": float(relative_correction * 100)
        }
    
    return assessment