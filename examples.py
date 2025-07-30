"""
Usage examples for the atmospheric correction pipeline.
Demonstrates how to use the pipeline for different scenarios.
"""

import os
import numpy as np
from pipeline import process_scene, batch_process_scenes
import config


def example_single_scene():
    """
    Example: Process a single Sentinel-2 L1C scene.
    """
    
    print("=" * 60)
    print("EXAMPLE 1: Single Scene Processing")
    print("=" * 60)
    
    # Path to your Sentinel-2 L1C .npy file
    scene_path = "data/S2A_MSIL1C_20230731_example.npy"
    
    # If the example file doesn't exist, create synthetic data for demonstration
    if not os.path.exists(scene_path):
        print(f"Creating example data at {scene_path}...")
        from test_pipeline import create_synthetic_sentinel2_data
        create_synthetic_sentinel2_data(scene_path, add_clouds=True, add_vegetation=True)
    
    # Process the scene
    print(f"\nProcessing scene: {scene_path}")
    results = process_scene(
        scene_path=scene_path,
        output_name="example_scene",
        save_outputs_flag=True,
        verbose=True
    )
    
    if results["success"]:
        print(f"\n✓ Processing completed successfully!")
        print(f"Cloud coverage: {results['processing_metadata']['cloud_statistics']['cloud_coverage_percent']:.1f}%")
        print(f"Output files saved in: {config.DATA_DIR}")
    else:
        print(f"✗ Processing failed: {results['error']}")


def example_batch_processing():
    """
    Example: Process multiple scenes in batch mode.
    """
    
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Batch Processing")
    print("=" * 60)
    
    # Create example directory with multiple scenes
    batch_dir = "data/batch_example"
    os.makedirs(batch_dir, exist_ok=True)
    
    # Create synthetic scenes for demonstration
    from test_pipeline import create_synthetic_sentinel2_data
    
    scene_files = []
    for i in range(3):
        scene_file = os.path.join(batch_dir, f"S2A_MSIL1C_scene_{i+1:02d}.npy")
        if not os.path.exists(scene_file):
            create_synthetic_sentinel2_data(
                scene_file, 
                add_clouds=(i % 2 == 0),  # Alternate cloud patterns
                seed=42 + i
            )
        scene_files.append(scene_file)
    
    print(f"\nProcessing {len(scene_files)} scenes in batch mode...")
    
    # Process all scenes in the directory
    results = batch_process_scenes(
        scene_directory=batch_dir,
        file_pattern="*.npy",
        verbose=True
    )
    
    print(f"\nBatch processing summary:")
    print(f"  - Total scenes: {results['total_scenes']}")
    print(f"  - Successful: {results['successful_scenes']}")
    print(f"  - Failed: {results['failed_scenes']}")
    print(f"  - Success rate: {results['success_rate']:.1f}%")


def example_load_results():
    """
    Example: Load and analyze processing results.
    """
    
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Loading and Analyzing Results")
    print("=" * 60)
    
    # Example of loading saved outputs
    scene_name = "example_scene"
    
    try:
        # Load corrected bands
        corrected_stack_path = os.path.join(config.CORRECTED_BANDS_DIR, f"{scene_name}_corrected_stack.npy")
        if os.path.exists(corrected_stack_path):
            corrected_bands = np.load(corrected_stack_path)
            print(f"Loaded corrected bands: {corrected_bands.shape}")
            print(f"  - Data type: {corrected_bands.dtype}")
            print(f"  - Value range: {corrected_bands.min():.4f} - {corrected_bands.max():.4f}")
        
        # Load spectral indices
        indices_to_load = ['NDVI', 'NDWI', 'NBR']
        for index_name in indices_to_load:
            index_path = os.path.join(config.INDICES_DIR, f"{scene_name}_{index_name}.npy")
            if os.path.exists(index_path):
                index_data = np.load(index_path)
                print(f"Loaded {index_name}: {index_data.shape}, range: {index_data.min():.3f} - {index_data.max():.3f}")
        
        # Load cloud mask
        cloud_mask_path = os.path.join(config.CLOUD_MASKS_DIR, f"{scene_name}_cloud_mask.npy")
        if os.path.exists(cloud_mask_path):
            cloud_mask = np.load(cloud_mask_path)
            cloud_coverage = np.mean(cloud_mask) * 100
            print(f"Loaded cloud mask: {cloud_mask.shape}, coverage: {cloud_coverage:.1f}%")
        
        # Load processing metadata
        import json
        metadata_path = os.path.join(config.DATA_DIR, f"{scene_name}_processing_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            print(f"Processing date: {metadata['processing_date']}")
            print(f"Bands processed: {metadata['bands_processed']}")
            print(f"Indices computed: {metadata['indices_computed']}")
        
    except Exception as e:
        print(f"Error loading results: {e}")
        print("Make sure to run example_single_scene() first to generate the results.")


def example_custom_parameters():
    """
    Example: Using custom processing parameters.
    """
    
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Custom Processing Parameters")
    print("=" * 60)
    
    # Create test data
    scene_path = "data/custom_example.npy"
    if not os.path.exists(scene_path):
        from test_pipeline import create_synthetic_sentinel2_data
        create_synthetic_sentinel2_data(scene_path, add_clouds=False, add_vegetation=True)
    
    # Temporarily modify config parameters
    original_threshold = config.CLOUD_THRESHOLD
    original_percentile = config.DOS_PERCENTILE
    
    try:
        # Use stricter cloud detection
        config.CLOUD_THRESHOLD = 0.2  # Lower threshold = more pixels classified as clouds
        config.DOS_PERCENTILE = 0.005  # Lower percentile for DOS correction
        
        print(f"Using custom parameters:")
        print(f"  - Cloud threshold: {config.CLOUD_THRESHOLD}")
        print(f"  - DOS percentile: {config.DOS_PERCENTILE}")
        
        results = process_scene(
            scene_path=scene_path,
            output_name="custom_parameters_example",
            save_outputs_flag=True,
            verbose=True
        )
        
        if results["success"]:
            print(f"\n✓ Custom processing completed!")
        
    finally:
        # Restore original parameters
        config.CLOUD_THRESHOLD = original_threshold
        config.DOS_PERCENTILE = original_percentile


def example_analysis_workflow():
    """
    Example: Complete analysis workflow for wildfire monitoring.
    """
    
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Wildfire Monitoring Workflow")
    print("=" * 60)
    
    # This example shows how to use the pipeline outputs for wildfire analysis
    scene_name = "example_scene"
    
    try:
        # Load NBR (Normalized Burn Ratio) - key index for fire monitoring
        nbr_path = os.path.join(config.INDICES_DIR, f"{scene_name}_NBR.npy")
        ndvi_path = os.path.join(config.INDICES_DIR, f"{scene_name}_NDVI.npy")
        cloud_mask_path = os.path.join(config.CLOUD_MASKS_DIR, f"{scene_name}_cloud_mask.npy")
        
        if all(os.path.exists(p) for p in [nbr_path, ndvi_path, cloud_mask_path]):
            nbr = np.load(nbr_path)
            ndvi = np.load(ndvi_path)
            cloud_mask = np.load(cloud_mask_path)
            
            # Create cloud-free mask
            valid_mask = ~cloud_mask.astype(bool)
            
            # Analyze vegetation health
            healthy_vegetation = (ndvi > 0.3) & valid_mask
            stressed_vegetation = (ndvi > 0.1) & (ndvi <= 0.3) & valid_mask
            
            # Potential burn area detection (low NBR values)
            potential_burn = (nbr < -0.1) & valid_mask
            
            # Calculate statistics
            total_valid = np.sum(valid_mask)
            healthy_pct = np.sum(healthy_vegetation) / total_valid * 100 if total_valid > 0 else 0
            stressed_pct = np.sum(stressed_vegetation) / total_valid * 100 if total_valid > 0 else 0
            burn_pct = np.sum(potential_burn) / total_valid * 100 if total_valid > 0 else 0
            
            print(f"Vegetation analysis (cloud-free areas):")
            print(f"  - Healthy vegetation: {healthy_pct:.1f}%")
            print(f"  - Stressed vegetation: {stressed_pct:.1f}%")
            print(f"  - Potential burn areas: {burn_pct:.1f}%")
            print(f"  - Valid pixels analyzed: {total_valid:,}")
            
        else:
            print("Required output files not found. Run example_single_scene() first.")
    
    except Exception as e:
        print(f"Error in analysis workflow: {e}")


def main():
    """
    Run all examples.
    """
    
    print("ATMOSPHERIC CORRECTION PIPELINE - USAGE EXAMPLES")
    print("=" * 80)
    
    # Run examples
    example_single_scene()
    example_batch_processing()
    example_load_results()
    example_custom_parameters()
    example_analysis_workflow()
    
    print("\n" + "=" * 80)
    print("🎉 ALL EXAMPLES COMPLETED!")
    print("\nFor more information, check:")
    print("  - config.py: Processing parameters")
    print("  - utils/: Individual module documentation")
    print("  - pipeline.py: Main processing functions")
    print("=" * 80)


if __name__ == "__main__":
    main()