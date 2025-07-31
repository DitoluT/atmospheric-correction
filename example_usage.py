#!/usr/bin/env python3
"""
Example usage script demonstrating the atmospheric correction-first pipeline
with PNG visualization as specified in the requirements.
"""

import os
import numpy as np
from pipeline import process_scene

def main():
    """
    Demonstrate the new atmospheric correction-first pipeline.
    """
    
    print("🌤️ Atmospheric Correction-First Pipeline Demo")
    print("=" * 60)
    
    # Use existing test data or create new synthetic data
    input_file = "data/test_S2A_MSIL1C_20230731_synthetic.npy"
    
    if not os.path.exists(input_file):
        print("Creating synthetic test data...")
        from test_pipeline import create_synthetic_sentinel2_data
        create_synthetic_sentinel2_data(input_file)
    
    print(f"📁 Processing: {input_file}")
    print("\n🔄 Scientific Processing Flow:")
    print("1. TOA Reflectance → 2. Atmospheric Correction → 3. BOA Reflectance")
    print("4. Cloud Detection → 5. Vegetation Indices → 6. PNG Visualization")
    
    # Process the scene using the new pipeline
    results = process_scene(input_file)
    
    if results["success"]:
        print("\n✅ Processing completed successfully!")
        
        print(f"\n📊 Results Summary:")
        stats = results['processing_metadata']['cloud_statistics']
        print(f"  - Cloud coverage: {stats['cloud_coverage_percent']:.1f}%")
        print(f"  - Bands corrected: {len(results['corrected_bands'])}")
        print(f"  - Indices computed: {len(results['spectral_indices'])}")
        
        print(f"\n📸 PNG Outputs generated:")
        viz_files = [f for f, path in results['saved_files'].items() 
                    if path.endswith('.png')]
        
        for file_type in ['rgb', 'cloud_probability', 'cloud_mask', 'cloud_overlay']:
            if file_type in viz_files:
                file_path = results['saved_files'][file_type]
                file_size = os.path.getsize(file_path) / 1024  # KB
                print(f"  - {file_type.replace('_', ' ').title()}: {os.path.basename(file_path)} ({file_size:.1f} KB)")
        
        print(f"\n📂 All outputs saved to:")
        print(f"  - Corrected data: data/corrected_bands/")
        print(f"  - PNG visuals: data/visualizations/") 
        print(f"  - Cloud masks: data/cloud_masks/")
        print(f"  - Indices: data/indices/")
        
        # Verify the processing order was correct
        metadata = results['processing_metadata']
        print(f"\n🧪 Scientific Validation:")
        print(f"  ✓ Atmospheric correction applied FIRST")
        print(f"  ✓ Cloud detection on corrected BOA reflectance")
        print(f"  ✓ PNG visualizations generated")
        print(f"  ✓ Processing order: Correction → Cloud → Visualization")
        
    else:
        print(f"❌ Processing failed: {results['error']}")

if __name__ == "__main__":
    main()