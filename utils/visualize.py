"""
PNG visualization utilities for atmospheric correction pipeline.
Generates RGB images, cloud probability maps, and overlay visualizations.
"""

import numpy as np
import os
from typing import Dict, Optional
import config

try:
    import imageio
except ImportError:
    print("Warning: imageio not available. Install with: pip install imageio")
    imageio = None


def create_rgb_image(corrected_bands: Dict[str, np.ndarray], 
                     scale: float = None,
                     gamma: float = None) -> np.ndarray:
    """
    Create RGB true color image from corrected surface reflectance bands.
    
    Args:
        corrected_bands (dict): Dictionary of corrected band arrays
        scale (float): Reflectance scaling factor (default: config.VIS_SCALE)
        gamma (float): Gamma correction factor (default: config.GAMMA_CORRECTION)
        
    Returns:
        np.ndarray: RGB image array (uint8, shape: H x W x 3)
    """
    
    if scale is None:
        scale = config.VIS_SCALE
    if gamma is None:
        gamma = config.GAMMA_CORRECTION
    
    # Check that all RGB bands are available
    missing_bands = [band for band in config.RGB_BANDS if band not in corrected_bands]
    if missing_bands:
        raise ValueError(f"Missing RGB bands: {missing_bands}")
    
    # Stack RGB bands in correct order
    rgb_stack = np.stack([corrected_bands[band] for band in config.RGB_BANDS], axis=-1)
    
    # Scale reflectance values for visualization
    # Reflectance is typically [0, 1], scale to reasonable display range
    rgb_scaled = np.clip(rgb_stack / scale, 0, 1)
    
    # Apply gamma correction for better visual appearance
    rgb_gamma = np.power(rgb_scaled, gamma)
    
    # Convert to 8-bit RGB
    rgb_uint8 = (rgb_gamma * 255).astype(np.uint8)
    
    return rgb_uint8


def create_cloud_probability_image(cloud_prob: np.ndarray) -> np.ndarray:
    """
    Create grayscale cloud probability heatmap image.
    
    Args:
        cloud_prob (np.ndarray): Cloud probability map (0-1 range)
        
    Returns:
        np.ndarray: 8-bit grayscale probability image
    """
    
    # Ensure probabilities are in [0, 1] range
    prob_clipped = np.clip(cloud_prob, 0, 1)
    
    # Convert to 8-bit grayscale (0 = black/no clouds, 255 = white/high probability)
    prob_uint8 = (prob_clipped * 255).astype(np.uint8)
    
    return prob_uint8


def create_cloud_mask_image(cloud_mask: np.ndarray) -> np.ndarray:
    """
    Create binary cloud mask image.
    
    Args:
        cloud_mask (np.ndarray): Binary cloud mask (boolean or 0/1)
        
    Returns:
        np.ndarray: 8-bit binary mask image (0 = black/clear, 255 = white/cloud)
    """
    
    # Convert to binary and then to 8-bit
    mask_binary = cloud_mask.astype(bool)
    mask_uint8 = np.where(mask_binary, 255, 0).astype(np.uint8)
    
    return mask_uint8


def create_cloud_overlay_image(rgb_image: np.ndarray, 
                              cloud_mask: np.ndarray,
                              overlay_color: tuple = (255, 0, 0),
                              overlay_alpha: float = 0.7) -> np.ndarray:
    """
    Create RGB image with cloud overlay in specified color.
    
    Args:
        rgb_image (np.ndarray): RGB base image (uint8, H x W x 3)
        cloud_mask (np.ndarray): Binary cloud mask
        overlay_color (tuple): RGB color for cloud overlay (default: red)
        overlay_alpha (float): Overlay transparency (0-1, default: 0.7)
        
    Returns:
        np.ndarray: RGB image with cloud overlay
    """
    
    overlay_image = rgb_image.copy()
    mask_binary = cloud_mask.astype(bool)
    
    # Apply color overlay where clouds are detected
    for i, color_value in enumerate(overlay_color):
        overlay_image[mask_binary, i] = (
            overlay_alpha * color_value + 
            (1 - overlay_alpha) * rgb_image[mask_binary, i]
        ).astype(np.uint8)
    
    return overlay_image


def save_png_visualizations(corrected_bands: Dict[str, np.ndarray],
                           cloud_prob: np.ndarray,
                           cloud_mask: np.ndarray,
                           output_dir: str,
                           scene_name: Optional[str] = None) -> Dict[str, str]:
    """
    Generate and save all PNG visualizations for a scene.
    
    Args:
        corrected_bands (dict): Dictionary of corrected surface reflectance bands
        cloud_prob (np.ndarray): Cloud probability map (0-1)
        cloud_mask (np.ndarray): Binary cloud mask
        output_dir (str): Output directory for PNG files
        scene_name (str, optional): Scene name for file naming
        
    Returns:
        dict: Dictionary of saved file paths
    """
    
    if imageio is None:
        raise ImportError("imageio package is required for PNG generation")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine file prefix
    prefix = f"{scene_name}_" if scene_name else ""
    
    saved_files = {}
    
    try:
        # 1. Generate RGB true color image
        rgb_image = create_rgb_image(corrected_bands)
        rgb_path = os.path.join(output_dir, f"{prefix}rgb.png")
        imageio.imwrite(rgb_path, rgb_image)
        saved_files['rgb'] = rgb_path
        
        # 2. Generate cloud probability heatmap
        prob_image = create_cloud_probability_image(cloud_prob)
        prob_path = os.path.join(output_dir, f"{prefix}cloud_prob.png")
        imageio.imwrite(prob_path, prob_image)
        saved_files['cloud_probability'] = prob_path
        
        # 3. Generate binary cloud mask
        mask_image = create_cloud_mask_image(cloud_mask)
        mask_path = os.path.join(output_dir, f"{prefix}cloud_mask.png")
        imageio.imwrite(mask_path, mask_image)
        saved_files['cloud_mask'] = mask_path
        
        # 4. Generate cloud overlay on RGB
        overlay_image = create_cloud_overlay_image(rgb_image, cloud_mask)
        overlay_path = os.path.join(output_dir, f"{prefix}cloud_overlay.png")
        imageio.imwrite(overlay_path, overlay_image)
        saved_files['cloud_overlay'] = overlay_path
        
    except Exception as e:
        raise RuntimeError(f"Failed to generate PNG visualizations: {e}")
    
    return saved_files


def create_visualization_summary(saved_files: Dict[str, str],
                               cloud_stats: dict,
                               output_dir: str,
                               scene_name: Optional[str] = None) -> str:
    """
    Create a simple text summary of the visualization outputs.
    
    Args:
        saved_files (dict): Dictionary of saved PNG file paths
        cloud_stats (dict): Cloud coverage statistics
        output_dir (str): Output directory
        scene_name (str, optional): Scene name
        
    Returns:
        str: Path to summary text file
    """
    
    prefix = f"{scene_name}_" if scene_name else ""
    summary_path = os.path.join(output_dir, f"{prefix}visualization_summary.txt")
    
    with open(summary_path, 'w') as f:
        f.write("PNG Visualization Summary\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Scene: {scene_name or 'Unknown'}\n")
        f.write(f"Cloud Coverage: {cloud_stats.get('cloud_coverage_percent', 0):.1f}%\n")
        f.write(f"Mean Cloud Probability: {cloud_stats.get('mean_cloud_probability', 0):.3f}\n\n")
        
        f.write("Generated Files:\n")
        for file_type, file_path in saved_files.items():
            filename = os.path.basename(file_path)
            f.write(f"  - {file_type}: {filename}\n")
        
        f.write("\nFile Descriptions:\n")
        f.write("  - rgb.png: True color RGB image (gamma corrected)\n")
        f.write("  - cloud_prob.png: Cloud probability heatmap (0-100%)\n")
        f.write("  - cloud_mask.png: Binary cloud mask (white=clouds)\n")
        f.write("  - cloud_overlay.png: RGB with red cloud overlay\n")
    
    return summary_path


def validate_visualization_inputs(corrected_bands: Dict[str, np.ndarray],
                                cloud_prob: np.ndarray,
                                cloud_mask: np.ndarray) -> bool:
    """
    Validate inputs for PNG visualization generation.
    
    Args:
        corrected_bands (dict): Dictionary of corrected bands
        cloud_prob (np.ndarray): Cloud probability map
        cloud_mask (np.ndarray): Binary cloud mask
        
    Returns:
        bool: True if inputs are valid
    """
    
    # Check RGB bands availability
    missing_rgb = [band for band in config.RGB_BANDS if band not in corrected_bands]
    if missing_rgb:
        raise ValueError(f"Missing RGB bands for visualization: {missing_rgb}")
    
    # Get reference shape from first RGB band
    ref_shape = corrected_bands[config.RGB_BANDS[0]].shape
    
    # Check that all RGB bands have same shape
    for band in config.RGB_BANDS:
        if corrected_bands[band].shape != ref_shape:
            raise ValueError(f"Band {band} shape mismatch: {corrected_bands[band].shape} vs {ref_shape}")
    
    # Check cloud data shapes match
    if cloud_prob.shape != ref_shape:
        raise ValueError(f"Cloud probability shape mismatch: {cloud_prob.shape} vs {ref_shape}")
    
    if cloud_mask.shape != ref_shape:
        raise ValueError(f"Cloud mask shape mismatch: {cloud_mask.shape} vs {ref_shape}")
    
    # Check cloud probability range
    if np.any(cloud_prob < 0) or np.any(cloud_prob > 1):
        print("Warning: Cloud probability values outside [0, 1] range")
    
    return True