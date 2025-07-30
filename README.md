
# 🌤️ Atmospheric Correction and Preprocessing Pipeline for Sentinel-2 L1C Imagery

## Objective
Build a Python pipeline for atmospheric correction and preprocessing of Sentinel-2 Level-1C (TOA reflectance) imagery. The pipeline must:
1. Generate cloud masks using **s2cloudless**
2. Perform lightweight atmospheric correction
3. Prepare corrected data for wildfire prediction models
4. Handle band resampling and cloud-aware processing
5. Maintain lightweight design without Sen2Cor/MAJA dependencies

---

## 🔧 System Requirements
- **Python 3.8+**
- **Key Packages**: 
  ```python
  rasterio, numpy, scikit-image, s2cloudless, matplotlib, scipy, tqdm, joblib, pandas
  ```

## 📁 Enhanced Project Structure
```
atmospheric-correction/
├── data/                          # Input Sentinel-2 L1C patches (256x256)
│   ├── S2A_MSIL1C_20230731_...    # .npy file with size [256, 256, 13] contains BANDS [B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B10, B11, B12]
│   ├── cloud_masks/               # Cloud probability masks <- s2cloudless output
│   ├── corrected_bands/           # Atmospherically corrected bands <- Output from correction.py
│   └── indices/                   # Computed vegetation indices <- Output from spectral_indices.py
├── utils/ 
│   ├── reader.py                  # Band reading/resampling
│   ├── cloud_detection.py         # s2cloudless integration
│   ├── correction.py              # Atmospheric correction methods
│   ├── spectral_indices.py        # Vegetation index calculations
│   └── resampling.py              # Resolution handling (Not needed to be implemented)
├── config.py                      # Central configuration
└── pipeline.py                    # Main processing workflow
```

## ✅ Enhanced Specifications

### config.py
```python
# s2cloudless requirements (10 bands at 20m)
CLOUD_BANDS = ['B01', 'B02', 'B04', 'B05', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12']

# Core bands for atmospheric correction
CORRECTION_BANDS = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12']

# Target resolution for processing (meters)
TARGET_RES = 20  

# Atmospheric correction parameters
DOS_PERCENTILE = 0.01  # Dark Object Subtraction percentile
CLOUD_THRESHOLD = 0.4  # Cloud probability threshold
```

### utils/resampling.py
**Function:** `resample_bands(band_dict, target_res=20)` (**DO NOT IMPLEMENT DATA ALREDY HAS THIS**)

- Resamples all bands to target resolution using bilinear interpolation
- Handles native resolutions (10m, 20m, 60m)
- Returns consistently sized numpy arrays

### utils/reader.py
**Function:** `read_band_stack(scene_path)`

- Reads all bands in `CLOUD_BANDS` + `CORRECTION_BANDS`
- Automatically handles .npy formats

Returns:
```python
{
    "metadata": gdal_info,
    "bands": {band_name: array},
    "maskable_bands": resampled_cloud_bands  # For s2cloudless
}
```

### utils/cloud_detection.py
**Function:** `generate_cloud_mask(band_stack)`

Input: 10-band stack in s2cloudless order (B01, B02, B04, B05, B08, B8A, B09, B10, B11, B12)

Processing:
```python
from s2cloudless import S2PixelCloudDetector

cloud_detector = S2PixelCloudDetector(
    threshold=config.CLOUD_THRESHOLD,
    all_bands=True
)
cloud_probs = cloud_detector.get_cloud_probability_maps(band_stack[np.newaxis, ...])
cloud_mask = cloud_probs > config.CLOUD_THRESHOLD
```

Output: `(cloud_probability.npy, binary_cloud_mask.npy)`

### utils/correction.py
**Function:** `apply_atmospheric_correction(band_data, cloud_mask)`

Implements Cloud-Optimized Dark Object Subtraction (CO-DOS):
- Create cloud-free pixels mask
- For each band:
  - Calculate 1st percentile over cloud-free areas
  - Subtract DOS value: `corrected = TOA - percentile_value`
- Clip negative values to 0
- Apply scale factor (1/10000)
- Returns corrected bands dictionary

### utils/spectral_indices.py
**Function:** `compute_indices(corrected_bands)`

Computes after atmospheric correction:
- **NDVI**: (B08 - B04) / (B08 + B04)
- **NDWI**: (B03 - B08) / (B03 + B08)
- **NBR**: (B08 - B12) / (B08 + B12)
- **SCL**: Scene Classification Layer (from cloud mask)

### pipeline.py
```python
def process_scene(scene_path):
    # 1. Read bands
    data = reader.read_band_stack(scene_path)
    
    # 2. Generate cloud mask (PRE PROCESS DATA BEFORE THIS, FOR HIGHER ACCURACY)
    cloud_prob, cloud_mask = cloud_detection.generate_cloud_mask(
        data["maskable_bands"]
    )
    
    # 3. Apply atmospheric correction (cloud-aware)
    corrected = correction.apply_atmospheric_correction(
        data["bands"], 
        cloud_mask
    )
    
    # 4. Compute spectral indices
    indices = spectral_indices.compute_indices(corrected)
    
    # 5. Save outputs
    save_outputs(corrected, indices, cloud_prob, cloud_mask)
```

## 🔄 Processing Workflow

**Input:** Raw Sentinel-2 L1C patches (.NPY FORMAT)

**Cloud Detection:**
- Generate cloud probability mask with s2cloudless (PRE PROCESS DATA FOR THIS)
- Create binary cloud mask using threshold

**Atmospheric Correction:**
- Apply CO-DOS using cloud-free pixels
- Clip negative reflectance values

**Spectral Indices:**
- Compute vegetation indices using corrected bands

**Output:**
- Corrected surface reflectance (ENVI .hdr)
- Cloud probability mask (.npy)
- Spectral indices (.npy)
- Metadata log (JSON)
- Sentinel-2 data corrected in format .npy

## 💡 Key Improvements


### Advanced Atmospheric Correction:
- Cloud-aware dark object subtraction
- Resolution-consistent processing

### Metadata Preservation:
- Maintains geospatial information
- Logs processing parameters

### Cloud Probability Output:
- Saves continuous cloud probability maps
- Provides binary masks at configurable thresholds

## 📝 Usage Example
```python
from pipeline import process_scene

SCENE_PATH = "data/S2A_MSIL1C_20230731_N0200_R094_T32TQS_20230731T102159"
process_scene(SCENE_PATH)
```

## 📌 Notes
- **File Formats**:
  - Corrected reflectance: ENVI format with HDR
  - Masks/Indices: Cloud-optimized .npy 
