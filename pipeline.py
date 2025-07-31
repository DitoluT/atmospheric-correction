"""
Main processing pipeline for atmospheric correction and preprocessing of Sentinel-2 L1C imagery.
Orchestrates the complete workflow from raw data to corrected products.
"""

import os
import json
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime
import config
from utils import reader, cloud_detection, correction, spectral_indices, visualize


def save_outputs(scene_name: str,
                corrected_bands: Dict[str, np.ndarray],
                indices: Dict[str, np.ndarray],
                cloud_prob: np.ndarray,
                cloud_mask: np.ndarray,
                scl: Optional[np.ndarray] = None,
                metadata: Optional[Dict[str, Any]] = None,
                save_png: bool = True) -> Dict[str, str]:
    """
    Save all processing outputs to appropriate directories.
    
    Args:
        scene_name (str): Base name for output files
        corrected_bands (dict): Atmospherically corrected bands
        indices (dict): Computed spectral indices
        cloud_prob (np.ndarray): Cloud probability map
        cloud_mask (np.ndarray): Binary cloud mask
        scl (np.ndarray, optional): Scene classification layer
        metadata (dict, optional): Processing metadata
        save_png (bool): Whether to generate PNG visualizations
        
    Returns:
        dict: Dictionary of saved file paths
    """
    
    # Ensure output directories exist
    os.makedirs(config.CLOUD_MASKS_DIR, exist_ok=True)
    os.makedirs(config.CORRECTED_BANDS_DIR, exist_ok=True)
    os.makedirs(config.INDICES_DIR, exist_ok=True)
    if save_png:
        os.makedirs(config.VISUALIZATIONS_DIR, exist_ok=True)
    
    saved_files = {}
    
    # Save cloud probability and mask
    cloud_prob_path = os.path.join(config.CLOUD_MASKS_DIR, f"{scene_name}_cloud_probability.npy")
    cloud_mask_path = os.path.join(config.CLOUD_MASKS_DIR, f"{scene_name}_cloud_mask.npy")
    
    np.save(cloud_prob_path, cloud_prob.astype(np.float32))
    np.save(cloud_mask_path, cloud_mask.astype(np.uint8))
    
    saved_files['cloud_probability'] = cloud_prob_path
    saved_files['cloud_mask'] = cloud_mask_path
    
    # Save corrected bands as individual files and stacked array
    band_stack = []
    band_order = []
    
    for band_name in config.CORRECTION_BANDS:
        if band_name in corrected_bands:
            band_path = os.path.join(config.CORRECTED_BANDS_DIR, f"{scene_name}_{band_name}_corrected.npy")
            np.save(band_path, corrected_bands[band_name].astype(np.float32))
            saved_files[f"corrected_{band_name}"] = band_path
            
            band_stack.append(corrected_bands[band_name])
            band_order.append(band_name)
    
    # Save stacked corrected bands
    if band_stack:
        stacked_bands = np.stack(band_stack, axis=-1)
        stacked_path = os.path.join(config.CORRECTED_BANDS_DIR, f"{scene_name}_corrected_stack.npy")
        np.save(stacked_path, stacked_bands.astype(np.float32))
        saved_files['corrected_stack'] = stacked_path
        
        # Save band order information
        band_order_path = os.path.join(config.CORRECTED_BANDS_DIR, f"{scene_name}_band_order.json")
        with open(band_order_path, 'w') as f:
            json.dump(band_order, f)
        saved_files['band_order'] = band_order_path
    
    # Save spectral indices
    for index_name, index_array in indices.items():
        if not index_name.endswith('_masked'):  # Save only the main indices
            index_path = os.path.join(config.INDICES_DIR, f"{scene_name}_{index_name}.npy")
            np.save(index_path, index_array.astype(np.float32))
            saved_files[f"index_{index_name}"] = index_path
    
    # Save scene classification if provided
    if scl is not None:
        scl_path = os.path.join(config.INDICES_DIR, f"{scene_name}_SCL.npy")
        np.save(scl_path, scl.astype(np.uint8))
        saved_files['scene_classification'] = scl_path
    
    # Save metadata
    if metadata is not None:
        metadata_path = os.path.join(config.DATA_DIR, f"{scene_name}_processing_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        saved_files['metadata'] = metadata_path
    
    # Generate PNG visualizations
    if save_png:
        try:
            png_files = visualize.save_png_visualizations(
                corrected_bands=corrected_bands,
                cloud_prob=cloud_prob,
                cloud_mask=cloud_mask,
                output_dir=config.VISUALIZATIONS_DIR,
                scene_name=scene_name
            )
            saved_files.update(png_files)
            
            # Create visualization summary
            if metadata and 'cloud_statistics' in metadata:
                cloud_stats = metadata['cloud_statistics']
            else:
                cloud_stats = {'cloud_coverage_percent': 0, 'mean_cloud_probability': 0}
            
            summary_path = visualize.create_visualization_summary(
                png_files, cloud_stats, config.VISUALIZATIONS_DIR, scene_name
            )
            saved_files['visualization_summary'] = summary_path
            
        except Exception as e:
            print(f"Warning: PNG visualization generation failed: {e}")
    
    return saved_files


def process_scene(scene_path: str, 
                 output_name: Optional[str] = None,
                 save_outputs_flag: bool = True,
                 verbose: bool = True) -> Dict[str, Any]:
    """
    Process a single Sentinel-2 L1C scene through the complete atmospheric correction pipeline.
    
    Args:
        scene_path (str): Path to .npy file containing Sentinel-2 L1C data
        output_name (str, optional): Base name for output files. If None, derived from input file
        save_outputs_flag (bool): Whether to save outputs to disk
        verbose (bool): Whether to print processing information
        
    Returns:
        dict: Complete processing results including all intermediate and final products
    """
    
    if verbose:
        print(f"Starting atmospheric correction pipeline for: {scene_path}")
    
    # Determine output name
    if output_name is None:
        output_name = os.path.splitext(os.path.basename(scene_path))[0]
    
    try:
        # Step 1: Read bands
        if verbose:
            print("Step 1: Reading band stack...")
        
        data = reader.read_band_stack(scene_path)
        
        if verbose:
            print(f"  - Loaded {data['metadata']['num_bands']} bands")
            print(f"  - Scene dimensions: {data['metadata']['height']} x {data['metadata']['width']}")
        
        # Step 2: Apply atmospheric correction FIRST
        if verbose:
            print("Step 2: Applying atmospheric correction...")
        
        # Apply DOS correction without cloud mask dependency
        corrected = correction.apply_atmospheric_correction(data["bands"], cloud_mask=None)
        
        if verbose:
            print(f"  - Corrected {len(corrected)} bands")
        
        # Step 3: Generate cloud mask on corrected BOA reflectance
        if verbose:
            print("Step 3: Generating cloud mask on corrected reflectance...")
        
        # Create corrected band stack for s2cloudless (using corrected reflectance)
        corrected_maskable_bands = np.zeros(
            (data['metadata']['height'], data['metadata']['width'], len(config.CLOUD_BANDS)), 
            dtype=np.float32
        )
        for i, band_name in enumerate(config.CLOUD_BANDS):
            if band_name in corrected:
                # s2cloudless expects values in [0, 1] range, our corrected bands are already scaled
                corrected_maskable_bands[:, :, i] = corrected[band_name]
            else:
                # Fallback to original scaled data if corrected band not available
                band_idx = config.BAND_INDICES[band_name]
                original_band = data['bands'][band_name] / config.REFLECTANCE_SCALE
                corrected_maskable_bands[:, :, i] = original_band
        
        # Generate cloud mask using corrected reflectance
        cloud_prob, cloud_mask = cloud_detection.generate_cloud_mask(corrected_maskable_bands)
        
        # Get cloud statistics
        cloud_stats = cloud_detection.cloud_coverage_stats(cloud_prob, cloud_mask)
        
        if verbose:
            print(f"  - Cloud coverage: {cloud_stats['cloud_coverage_percent']:.1f}%")
            print(f"  - Mean cloud probability: {cloud_stats['mean_cloud_probability']:.3f}")
        
        # Step 4: Compute spectral indices
        if verbose:
            print("Step 4: Computing spectral indices...")
        
        indices = spectral_indices.compute_indices(corrected, cloud_mask)
        
        # Remove masked versions for cleaner output
        main_indices = {k: v for k, v in indices.items() if not k.endswith('_masked')}
        
        if verbose:
            print(f"  - Computed {len(main_indices)} spectral indices: {list(main_indices.keys())}")
        
        # Step 5: Create scene classification
        if verbose:
            print("Step 5: Creating scene classification...")
        
        scl = spectral_indices.create_scene_classification(corrected, cloud_mask)
        
        # Compile processing metadata
        processing_metadata = {
            "processing_date": datetime.utcnow().isoformat(),
            "input_file": scene_path,
            "output_name": output_name,
            "scene_metadata": data["metadata"],
            "cloud_statistics": cloud_stats,
            "processing_parameters": {
                "cloud_threshold": config.CLOUD_THRESHOLD,
                "dos_percentile": config.DOS_PERCENTILE,
                "target_resolution": config.TARGET_RES
            },
            "bands_processed": list(corrected.keys()),
            "indices_computed": list(main_indices.keys())
        }
        
        # Quality assessment
        if verbose:
            print("Step 6: Quality assessment...")
            
        qa_results = correction.quality_assessment(data["bands"], corrected, cloud_mask)
        processing_metadata["quality_assessment"] = qa_results
        
        # Index statistics
        index_stats = spectral_indices.calculate_index_statistics(main_indices, cloud_mask)
        processing_metadata["index_statistics"] = index_stats
        
        # Step 7: Save outputs
        saved_files = {}
        if save_outputs_flag:
            if verbose:
                print("Step 7: Saving outputs...")
            
            saved_files = save_outputs(
                output_name, corrected, main_indices, cloud_prob, cloud_mask, scl, processing_metadata, 
                save_png=True
            )
            
            if verbose:
                print(f"  - Saved {len(saved_files)} output files")
        
        if verbose:
            print("Processing completed successfully!")
        
        # Return complete results
        return {
            "input_metadata": data["metadata"],
            "corrected_bands": corrected,
            "spectral_indices": main_indices,
            "cloud_probability": cloud_prob,
            "cloud_mask": cloud_mask,
            "scene_classification": scl,
            "processing_metadata": processing_metadata,
            "saved_files": saved_files,
            "success": True
        }
        
    except Exception as e:
        error_msg = f"Processing failed: {str(e)}"
        if verbose:
            print(f"ERROR: {error_msg}")
        
        return {
            "success": False,
            "error": error_msg,
            "input_file": scene_path
        }


def batch_process_scenes(scene_directory: str,
                        output_directory: Optional[str] = None,
                        file_pattern: str = "*.npy",
                        max_workers: int = 1,
                        verbose: bool = True) -> Dict[str, Any]:
    """
    Process multiple scenes in batch mode.
    
    Args:
        scene_directory (str): Directory containing .npy scene files
        output_directory (str, optional): Output directory. If None, uses config defaults
        file_pattern (str): Pattern to match scene files
        max_workers (int): Number of parallel workers (currently limited to 1)
        verbose (bool): Whether to print progress information
        
    Returns:
        dict: Batch processing results and summary
    """
    
    import glob
    from tqdm import tqdm
    
    # Find all scene files
    scene_pattern = os.path.join(scene_directory, file_pattern)
    scene_files = glob.glob(scene_pattern)
    
    if not scene_files:
        return {
            "success": False,
            "error": f"No files found matching pattern: {scene_pattern}",
            "processed_scenes": []
        }
    
    if verbose:
        print(f"Found {len(scene_files)} scenes to process")
    
    # Process each scene
    results = []
    failed_scenes = []
    
    for scene_file in tqdm(scene_files, desc="Processing scenes", disable=not verbose):
        try:
            result = process_scene(scene_file, verbose=False)
            results.append(result)
            
            if not result["success"]:
                failed_scenes.append(scene_file)
                
        except Exception as e:
            failed_scenes.append(scene_file)
            if verbose:
                print(f"Failed to process {scene_file}: {e}")
    
    # Compile summary
    successful_scenes = [r for r in results if r.get("success", False)]
    
    summary = {
        "total_scenes": len(scene_files),
        "successful_scenes": len(successful_scenes),
        "failed_scenes": len(failed_scenes),
        "success_rate": len(successful_scenes) / len(scene_files) * 100,
        "failed_scene_list": failed_scenes,
        "processing_results": results
    }
    
    if verbose:
        print(f"Batch processing completed:")
        print(f"  - Total scenes: {summary['total_scenes']}")
        print(f"  - Successful: {summary['successful_scenes']}")
        print(f"  - Failed: {summary['failed_scenes']}")
        print(f"  - Success rate: {summary['success_rate']:.1f}%")
    
    return summary


def main():
    """
    Main function for command-line usage.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Atmospheric Correction Pipeline for Sentinel-2 L1C")
    parser.add_argument("input", help="Input .npy file or directory containing scenes")
    parser.add_argument("--output-name", help="Output name prefix (for single file processing)")
    parser.add_argument("--batch", action="store_true", help="Process directory in batch mode")
    parser.add_argument("--no-save", action="store_true", help="Don't save outputs to disk")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    
    args = parser.parse_args()
    
    verbose = not args.quiet
    save_outputs_flag = not args.no_save
    
    if args.batch or os.path.isdir(args.input):
        # Batch processing
        results = batch_process_scenes(args.input, verbose=verbose)
        
        # Save batch summary
        if save_outputs_flag:
            summary_path = os.path.join(config.DATA_DIR, "batch_processing_summary.json")
            with open(summary_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"Batch summary saved to: {summary_path}")
    
    else:
        # Single file processing
        results = process_scene(args.input, args.output_name, save_outputs_flag, verbose)
        
        if results["success"]:
            print("Processing completed successfully!")
        else:
            print(f"Processing failed: {results['error']}")
            exit(1)


if __name__ == "__main__":
    main()