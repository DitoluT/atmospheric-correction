"""
Configuration file for atmospheric correction pipeline.
Contains band definitions, processing parameters, and file paths.
"""

# s2cloudless requirements (10 bands at 20m resolution)
CLOUD_BANDS = ['B01', 'B02', 'B04', 'B05', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12']

# Core bands for atmospheric correction
CORRECTION_BANDS = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12']

# All Sentinel-2 bands in order as they appear in .npy files
ALL_BANDS = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12']

# Target resolution for processing (meters)
TARGET_RES = 20

# Atmospheric correction parameters
DOS_PERCENTILE = 0.01  # Dark Object Subtraction percentile
CLOUD_THRESHOLD = 0.4  # Cloud probability threshold

# Scale factors
REFLECTANCE_SCALE = 10000  # Sentinel-2 scale factor for reflectance values

# RGB Configuration for True Color Visualization
RGB_BANDS = ['B04', 'B03', 'B02']  # True color order (Red, Green, Blue)

# Visualization Parameters
VIS_SCALE = 3000  # Reflectance scaling for RGB visualization
GAMMA_CORRECTION = 0.5  # Gamma correction for better visual appearance

# File paths
DATA_DIR = "data"
CLOUD_MASKS_DIR = "data/cloud_masks"
CORRECTED_BANDS_DIR = "data/corrected_bands"
INDICES_DIR = "data/indices"
VISUALIZATIONS_DIR = "data/visualizations"

# Band indices mapping for quick access
BAND_INDICES = {band: idx for idx, band in enumerate(ALL_BANDS)}

# s2cloudless band indices (subset of ALL_BANDS)
CLOUD_BAND_INDICES = [BAND_INDICES[band] for band in CLOUD_BANDS]

# Correction band indices (subset of ALL_BANDS)
CORRECTION_BAND_INDICES = [BAND_INDICES[band] for band in CORRECTION_BANDS]