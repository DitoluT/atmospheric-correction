"""
Spectral indices computation for vegetation analysis and fire monitoring.
Computes NDVI, NDWI, NBR and scene classification from corrected bands.
"""

import numpy as np
from typing import Dict, Optional
import config


def compute_indices(corrected_bands: Dict[str, np.ndarray],
                   cloud_mask: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """
    Compute spectral indices from atmospherically corrected bands.
    
    Args:
        corrected_bands (dict): Dictionary of corrected band arrays
        cloud_mask (np.ndarray, optional): Cloud mask to exclude cloudy pixels
        
    Returns:
        dict: Dictionary of computed spectral indices
    """
    
    indices = {}
    
    # Helper function to safely compute normalized difference
    def safe_normalized_difference(band1: np.ndarray, band2: np.ndarray) -> np.ndarray:
        """Compute normalized difference with division by zero protection."""
        numerator = band1 - band2
        denominator = band1 + band2
        
        # Avoid division by zero
        result = np.zeros_like(numerator)
        valid_mask = denominator != 0
        result[valid_mask] = numerator[valid_mask] / denominator[valid_mask]
        
        # Clip to [-1, 1] range
        result = np.clip(result, -1, 1)
        
        return result
    
    # NDVI (Normalized Difference Vegetation Index)
    if 'B08' in corrected_bands and 'B04' in corrected_bands:
        nir = corrected_bands['B08']   # Near Infrared
        red = corrected_bands['B04']   # Red
        indices['NDVI'] = safe_normalized_difference(nir, red)
    
    # NDWI (Normalized Difference Water Index)
    if 'B03' in corrected_bands and 'B08' in corrected_bands:
        green = corrected_bands['B03']  # Green
        nir = corrected_bands['B08']    # Near Infrared
        indices['NDWI'] = safe_normalized_difference(green, nir)
    
    # NBR (Normalized Burn Ratio) - important for fire monitoring
    if 'B08' in corrected_bands and 'B12' in corrected_bands:
        nir = corrected_bands['B08']   # Near Infrared
        swir2 = corrected_bands['B12'] # SWIR 2
        indices['NBR'] = safe_normalized_difference(nir, swir2)
    
    # NDII (Normalized Difference Infrared Index) - vegetation moisture
    if 'B08' in corrected_bands and 'B11' in corrected_bands:
        nir = corrected_bands['B08']   # Near Infrared
        swir1 = corrected_bands['B11'] # SWIR 1
        indices['NDII'] = safe_normalized_difference(nir, swir1)
    
    # BAI (Burned Area Index) - specific for burned area detection
    if 'B04' in corrected_bands and 'B12' in corrected_bands:
        red = corrected_bands['B04']
        swir2 = corrected_bands['B12']
        # BAI = 1 / ((0.1 - RED)² + (0.06 - SWIR)²)
        # Simplified version using normalized difference
        indices['BAI'] = safe_normalized_difference(swir2, red)
    
    # SAVI (Soil-Adjusted Vegetation Index) - reduces soil background effects
    if 'B08' in corrected_bands and 'B04' in corrected_bands:
        nir = corrected_bands['B08']
        red = corrected_bands['B04']
        L = 0.5  # Soil brightness correction factor
        # SAVI = ((NIR - RED) / (NIR + RED + L)) * (1 + L)
        numerator = (nir - red) * (1 + L)
        denominator = nir + red + L
        savi = np.zeros_like(numerator)
        valid_mask = denominator != 0
        savi[valid_mask] = numerator[valid_mask] / denominator[valid_mask]
        indices['SAVI'] = np.clip(savi, -1, 1)
    
    # Apply cloud mask if provided
    if cloud_mask is not None:
        # Create a copy of the indices to avoid modifying during iteration
        indices_copy = indices.copy()
        for index_name, index_array in indices_copy.items():
            masked_index = index_array.copy()
            masked_index[cloud_mask] = np.nan
            indices[f"{index_name}_masked"] = masked_index
    
    return indices


def create_scene_classification(corrected_bands: Dict[str, np.ndarray],
                              cloud_mask: np.ndarray,
                              water_threshold: float = 0.3,
                              vegetation_threshold: float = 0.3,
                              burned_threshold: float = -0.3) -> np.ndarray:
    """
    Create a scene classification layer (SCL) based on spectral indices.
    
    Args:
        corrected_bands (dict): Corrected band reflectance values
        cloud_mask (np.ndarray): Binary cloud mask
        water_threshold (float): NDWI threshold for water detection
        vegetation_threshold (float): NDVI threshold for vegetation detection
        burned_threshold (float): NBR threshold for burned area detection
        
    Returns:
        np.ndarray: Scene classification map with class codes
    """
    
    # Classification codes
    CLASS_CODES = {
        'NO_DATA': 0,
        'CLOUD': 1,
        'WATER': 2,
        'VEGETATION': 3,
        'BARE_SOIL': 4,
        'BURNED': 5,
        'SNOW_ICE': 6,
        'UNCLASSIFIED': 7
    }
    
    # Initialize classification array
    height, width = cloud_mask.shape
    scl = np.full((height, width), CLASS_CODES['UNCLASSIFIED'], dtype=np.uint8)
    
    # Compute necessary indices
    indices = compute_indices(corrected_bands)
    
    # Apply cloud mask first
    scl[cloud_mask] = CLASS_CODES['CLOUD']
    
    # Water detection using NDWI
    if 'NDWI' in indices:
        water_mask = indices['NDWI'] > water_threshold
        scl[water_mask & ~cloud_mask] = CLASS_CODES['WATER']
    
    # Vegetation detection using NDVI
    if 'NDVI' in indices:
        vegetation_mask = indices['NDVI'] > vegetation_threshold
        scl[vegetation_mask & ~cloud_mask & ~(scl == CLASS_CODES['WATER'])] = CLASS_CODES['VEGETATION']
    
    # Burned area detection using NBR
    if 'NBR' in indices:
        burned_mask = indices['NBR'] < burned_threshold
        scl[burned_mask & ~cloud_mask & ~(scl == CLASS_CODES['WATER'])] = CLASS_CODES['BURNED']
    
    # Snow/ice detection using NDSI (if available bands)
    if 'B03' in corrected_bands and 'B11' in corrected_bands:
        green = corrected_bands['B03']
        swir1 = corrected_bands['B11']
        
        # Compute NDSI
        numerator = green - swir1
        denominator = green + swir1
        ndsi = np.zeros_like(numerator)
        valid_mask = denominator != 0
        ndsi[valid_mask] = numerator[valid_mask] / denominator[valid_mask]
        
        # Snow/ice threshold
        snow_mask = (ndsi > 0.4) & (corrected_bands['B03'] > 0.15)
        scl[snow_mask & ~cloud_mask] = CLASS_CODES['SNOW_ICE']
    
    # Bare soil (remaining non-cloud, non-water pixels with low vegetation)
    if 'NDVI' in indices:
        bare_soil_mask = (indices['NDVI'] < 0.2) & ~cloud_mask & ~(scl == CLASS_CODES['WATER']) & \
                        ~(scl == CLASS_CODES['BURNED']) & ~(scl == CLASS_CODES['SNOW_ICE'])
        scl[bare_soil_mask] = CLASS_CODES['BARE_SOIL']
    
    return scl


def calculate_index_statistics(indices: Dict[str, np.ndarray],
                             cloud_mask: Optional[np.ndarray] = None) -> Dict[str, dict]:
    """
    Calculate statistics for computed spectral indices.
    
    Args:
        indices (dict): Dictionary of spectral indices
        cloud_mask (np.ndarray, optional): Cloud mask to exclude cloudy pixels
        
    Returns:
        dict: Statistics for each index
    """
    
    stats = {}
    
    for index_name, index_array in indices.items():
        if index_name.endswith('_masked'):
            continue  # Skip masked versions for statistics
            
        valid_data = index_array.copy()
        
        # Apply cloud mask if provided
        if cloud_mask is not None:
            valid_data = valid_data[~cloud_mask]
        
        # Remove NaN values
        valid_data = valid_data[~np.isnan(valid_data)]
        
        if len(valid_data) > 0:
            stats[index_name] = {
                'mean': float(np.mean(valid_data)),
                'std': float(np.std(valid_data)),
                'min': float(np.min(valid_data)),
                'max': float(np.max(valid_data)),
                'median': float(np.median(valid_data)),
                'q25': float(np.percentile(valid_data, 25)),
                'q75': float(np.percentile(valid_data, 75)),
                'valid_pixels': len(valid_data)
            }
        else:
            stats[index_name] = {
                'mean': np.nan, 'std': np.nan, 'min': np.nan, 'max': np.nan,
                'median': np.nan, 'q25': np.nan, 'q75': np.nan, 'valid_pixels': 0
            }
    
    return stats


def detect_water_bodies(corrected_bands: Dict[str, np.ndarray],
                       ndwi_threshold: float = 0.3,
                       additional_constraints: bool = True) -> np.ndarray:
    """
    Detect water bodies using NDWI and additional spectral constraints.
    
    Args:
        corrected_bands (dict): Corrected spectral bands
        ndwi_threshold (float): NDWI threshold for water detection
        additional_constraints (bool): Apply additional spectral constraints
        
    Returns:
        np.ndarray: Binary water mask
    """
    
    indices = compute_indices(corrected_bands)
    
    if 'NDWI' not in indices:
        print("Warning: Cannot compute NDWI for water detection")
        return np.zeros(list(corrected_bands.values())[0].shape, dtype=bool)
    
    # Basic NDWI threshold
    water_mask = indices['NDWI'] > ndwi_threshold
    
    # Additional constraints to reduce false positives
    if additional_constraints and 'B08' in corrected_bands:
        # Water typically has low NIR reflectance
        nir_constraint = corrected_bands['B08'] < 0.1
        water_mask = water_mask & nir_constraint
        
        # Water typically has low NDVI
        if 'NDVI' in indices:
            ndvi_constraint = indices['NDVI'] < 0.1
            water_mask = water_mask & ndvi_constraint
    
    return water_mask