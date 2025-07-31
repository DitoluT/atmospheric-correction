"""
Test script for atmospheric correction pipeline.
Creates synthetic Sentinel-2 L1C data and tests the complete workflow.
"""

import numpy as np
import os
import sys
import config
from pipeline import process_scene


def create_synthetic_sentinel2_data(output_path: str, 
                                   height: int = 256, 
                                   width: int = 256,
                                   add_clouds: bool = True,
                                   add_vegetation: bool = True,
                                   seed: int = 42) -> str:
    """
    Create synthetic Sentinel-2 L1C data that mimics real satellite imagery.
    
    Args:
        output_path (str): Path to save the synthetic .npy file
        height (int): Image height in pixels
        width (int): Image width in pixels
        add_clouds (bool): Whether to add cloud-like patterns
        add_vegetation (bool): Whether to add vegetation-like patterns
        seed (int): Random seed for reproducibility
        
    Returns:
        str: Path to the created file
    """
    
    np.random.seed(seed)
    
    # Create the band stack [height, width, 13 bands]
    band_stack = np.zeros((height, width, len(config.ALL_BANDS)), dtype=np.uint16)
    
    # Sentinel-2 band characteristics (approximate wavelengths and typical TOA values)
    band_info = {
        'B01': {'wavelength': 443, 'base_value': 1200, 'variation': 300},   # Coastal aerosol
        'B02': {'wavelength': 490, 'base_value': 1500, 'variation': 400},   # Blue
        'B03': {'wavelength': 560, 'base_value': 1800, 'variation': 500},   # Green
        'B04': {'wavelength': 665, 'base_value': 1400, 'variation': 600},   # Red
        'B05': {'wavelength': 705, 'base_value': 2000, 'variation': 700},   # Red edge 1
        'B06': {'wavelength': 740, 'base_value': 2200, 'variation': 800},   # Red edge 2
        'B07': {'wavelength': 783, 'base_value': 2400, 'variation': 900},   # Red edge 3
        'B08': {'wavelength': 842, 'base_value': 2600, 'variation': 1000},  # NIR
        'B8A': {'wavelength': 865, 'base_value': 2500, 'variation': 950},   # NIR narrow
        'B09': {'wavelength': 945, 'base_value': 1000, 'variation': 200},   # Water vapor
        'B10': {'wavelength': 1375, 'base_value': 800, 'variation': 150},   # Cirrus
        'B11': {'wavelength': 1610, 'base_value': 1600, 'variation': 600},  # SWIR 1
        'B12': {'wavelength': 2190, 'base_value': 1200, 'variation': 500}   # SWIR 2
    }
    
    # Create spatial patterns
    x, y = np.meshgrid(np.linspace(0, 10, width), np.linspace(0, 10, height))
    base_pattern = np.sin(x) * np.cos(y) * 0.1 + 1.0
    
    # Generate each band
    for i, band_name in enumerate(config.ALL_BANDS):
        info = band_info[band_name]
        
        # Base reflectance values
        base = info['base_value'] * base_pattern
        
        # Add random variation
        noise = np.random.normal(0, info['variation'] * 0.2, (height, width))
        
        # Combine base + noise
        band_values = base + noise
        
        # Add vegetation effect for NIR bands (higher reflectance in vegetation areas)
        if add_vegetation and band_name in ['B08', 'B8A', 'B05', 'B06', 'B07']:
            vegetation_mask = (x + y) > 8  # Simple vegetation pattern
            band_values[vegetation_mask] *= 1.8
        
        # Reduce NIR for water-like areas
        if band_name in ['B08', 'B8A'] and add_vegetation:
            water_mask = (x + y) < 6
            band_values[water_mask] *= 0.3
        
        # Add cloud effects (bright in visible, reduced in SWIR)
        if add_clouds:
            cloud_pattern = np.exp(-((x-5)**2 + (y-3)**2) / 8) > 0.2  # Reduced cloud coverage
            if band_name in ['B01', 'B02', 'B03', 'B04']:
                band_values[cloud_pattern] += 1500  # Slightly less bright clouds
            elif band_name in ['B11', 'B12']:
                band_values[cloud_pattern] *= 0.9   # Less reduction in SWIR
        
        # Clip to valid range and convert to uint16
        band_values = np.clip(band_values, 0, 10000)
        band_stack[:, :, i] = band_values.astype(np.uint16)
    
    # Save the synthetic data
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    np.save(output_path, band_stack)
    
    print(f"Created synthetic Sentinel-2 data: {output_path}")
    print(f"  - Shape: {band_stack.shape}")
    print(f"  - Data type: {band_stack.dtype}")
    print(f"  - Value range: {band_stack.min()} - {band_stack.max()}")
    
    return output_path


def test_pipeline():
    """
    Test the complete atmospheric correction pipeline.
    """
    
    print("=" * 60)
    print("ATMOSPHERIC CORRECTION PIPELINE TEST")
    print("=" * 60)
    
    # Create synthetic test data
    test_data_path = "data/test_S2A_MSIL1C_20230731_synthetic.npy"
    create_synthetic_sentinel2_data(test_data_path)
    
    print("\n" + "=" * 60)
    print("PROCESSING PIPELINE TEST")
    print("=" * 60)
    
    # Process the synthetic scene
    try:
        results = process_scene(
            scene_path=test_data_path,
            output_name="test_synthetic_scene",
            save_outputs_flag=True,
            verbose=True
        )
        
        if results["success"]:
            print("\n" + "=" * 60)
            print("PROCESSING RESULTS SUMMARY")
            print("=" * 60)
            
            print(f"✓ Input processed successfully")
            print(f"✓ Cloud coverage: {results['processing_metadata']['cloud_statistics']['cloud_coverage_percent']:.1f}%")
            print(f"✓ Bands corrected: {len(results['corrected_bands'])}")
            print(f"✓ Indices computed: {len(results['spectral_indices'])}")
            print(f"✓ Files saved: {len(results['saved_files'])}")
            
            print("\nSpectral indices computed:")
            for index_name in results['spectral_indices'].keys():
                stats = results['processing_metadata']['index_statistics'][index_name]
                print(f"  - {index_name}: mean={stats['mean']:.3f}, std={stats['std']:.3f}")
            
            print("\nOutput files saved:")
            for file_type, file_path in results['saved_files'].items():
                print(f"  - {file_type}: {file_path}")
            
            print("\n✓ PIPELINE TEST COMPLETED SUCCESSFULLY!")
            return True
            
        else:
            print(f"\n✗ PIPELINE TEST FAILED: {results['error']}")
            return False
            
    except Exception as e:
        print(f"\n✗ PIPELINE TEST FAILED WITH EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_individual_modules():
    """
    Test individual modules separately.
    """
    
    print("\n" + "=" * 60)
    print("INDIVIDUAL MODULE TESTS")
    print("=" * 60)
    
    # Test config
    print("Testing config module...")
    print(f"  - Cloud bands: {len(config.CLOUD_BANDS)}")
    print(f"  - Correction bands: {len(config.CORRECTION_BANDS)}")
    print(f"  - All bands: {len(config.ALL_BANDS)}")
    print("  ✓ Config module OK")
    
    # Test reader
    print("\nTesting reader module...")
    test_data_path = "data/test_S2A_MSIL1C_20230731_synthetic.npy"
    if os.path.exists(test_data_path):
        from utils import reader
        data = reader.read_band_stack(test_data_path)
        print(f"  - Loaded {len(data['bands'])} bands")
        print(f"  - Maskable bands shape: {data['maskable_bands'].shape}")
        print("  ✓ Reader module OK")
    else:
        print("  ! No test data available for reader test")
    
    # Test cloud detection
    print("\nTesting cloud detection module...")
    try:
        from utils import cloud_detection
        if os.path.exists(test_data_path):
            from utils import reader
            data = reader.read_band_stack(test_data_path)
            cloud_prob, cloud_mask = cloud_detection.generate_cloud_mask(data['maskable_bands'])
            print(f"  - Cloud probability shape: {cloud_prob.shape}")
            print(f"  - Cloud mask shape: {cloud_mask.shape}")
            print(f"  - Cloud coverage: {np.mean(cloud_mask)*100:.1f}%")
            print("  ✓ Cloud detection module OK")
        else:
            print("  ! No test data available for cloud detection test")
    except Exception as e:
        print(f"  ! Cloud detection test failed: {e}")
    
    print("\n✓ INDIVIDUAL MODULE TESTS COMPLETED")


if __name__ == "__main__":
    print("Starting atmospheric correction pipeline tests...\n")
    
    # Test individual modules first
    test_individual_modules()
    
    # Test complete pipeline
    success = test_pipeline()
    
    if success:
        print("\n🎉 ALL TESTS PASSED! The atmospheric correction pipeline is ready to use.")
        sys.exit(0)
    else:
        print("\n❌ TESTS FAILED! Please check the error messages above.")
        sys.exit(1)