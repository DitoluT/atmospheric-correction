# 🚀 Quick Start Guide

## Installation

```bash
# Clone the repository
git clone https://github.com/DitoluT/atmospheric-correction.git
cd atmospheric-correction

# Install dependencies
pip install -r requirements.txt
```

## Basic Usage

### Command Line Interface

```bash
# Process a single scene
python pipeline.py data/your_scene.npy

# Process with custom output name
python pipeline.py data/your_scene.npy --output-name my_corrected_scene

# Batch process all .npy files in a directory
python pipeline.py data/scenes_directory/ --batch

# Process without saving outputs (testing)
python pipeline.py data/your_scene.npy --no-save

# Quiet mode (suppress verbose output)
python pipeline.py data/your_scene.npy --quiet
```

### Python API

```python
from pipeline import process_scene

# Process a single scene
results = process_scene(
    scene_path="data/S2A_MSIL1C_20230731_example.npy",
    output_name="my_scene",
    save_outputs_flag=True,
    verbose=True
)

if results["success"]:
    print(f"✅ Processing completed!")
    print(f"Cloud coverage: {results['processing_metadata']['cloud_statistics']['cloud_coverage_percent']:.1f}%")
else:
    print(f"❌ Error: {results['error']}")
```

## Input Data Format

Your Sentinel-2 L1C data should be in `.npy` format with shape `[height, width, 13]` containing bands in this order:
```
[B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B10, B11, B12]
```

## Output Structure

```
data/
├── cloud_masks/          # Cloud probability and binary masks
├── corrected_bands/      # Atmospherically corrected bands
├── indices/             # Spectral indices (NDVI, NDWI, NBR, etc.)
└── *_processing_metadata.json  # Processing logs and statistics
```

## Testing

```bash
# Run the complete test suite
python test_pipeline.py

# Run usage examples
python examples.py
```

## Configuration

Edit `config.py` to modify processing parameters:
- `CLOUD_THRESHOLD`: Cloud detection sensitivity (default: 0.4)
- `DOS_PERCENTILE`: Dark object subtraction percentile (default: 0.01)
- Band definitions and file paths

---

**Ready to process your Sentinel-2 L1C data! 🛰️**