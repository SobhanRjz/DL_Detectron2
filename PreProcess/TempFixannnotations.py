import json
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
from typing import Dict, List, Any, Tuple
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_segmentation_area(segmentation: List[List[float]]) -> float:
    """
    Calculate the area of a polygon segmentation using the Shoelace formula.
    
    Args:
        segmentation: List of polygon coordinates [x1, y1, x2, y2, ...]
    
    Returns:
        float: Area of the polygon
    """
    # Reshape the flat list into pairs of coordinates
    if not segmentation or len(segmentation) == 0:
        return 0
    
    # Handle RLE format or nested lists
    if isinstance(segmentation[0], list):
        # Take the first polygon if there are multiple
        points = segmentation[0]
    else:
        points = segmentation
    
    # Convert to numpy array and reshape to pairs
    points = np.array(points).reshape(-1, 2)
    
    # Shoelace formula
    x = points[:, 0]
    y = points[:, 1]
    area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    return area

def calculate_segmentation_dimensions(segmentation: List[List[float]]) -> Tuple[float, float]:
    """
    Calculate the width and height of a segmentation polygon.
    
    Args:
        segmentation: List of polygon coordinates [x1, y1, x2, y2, ...]
    
    Returns:
        Tuple[float, float]: Width and height of the segmentation
    """
    if not segmentation or len(segmentation) == 0:
        return 0, 0
    
    # Handle RLE format or nested lists
    if isinstance(segmentation[0], list):
        # Take the first polygon if there are multiple
        points = segmentation[0]
    else:
        points = segmentation
    
    # Convert to numpy array and reshape to pairs
    points = np.array(points).reshape(-1, 2)
    
    # Calculate min and max for x and y coordinates
    min_x, max_x = np.min(points[:, 0]), np.max(points[:, 0])
    min_y, max_y = np.min(points[:, 1]), np.max(points[:, 1])
    
    # Calculate width and height
    width = max_x - min_x
    height = max_y - min_y
    
    return width, height

def add_segmentation_from_bbox(annotations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Add segmentation data to annotations that have bbox but no segmentation.
    
    Args:
        annotations: List of annotation dictionaries
        
    Returns:
        List[Dict[str, Any]]: Updated annotations list
    """
    updated_count = 0
    for ann in annotations:
        if not ann.get('segmentation') and 'bbox' in ann:
            bbox = ann['bbox']
            # COCO bbox format: [x, y, width, height]
            x, y, width, height = bbox
            
            # Create polygon segmentation from bbox
            # Format: [x1, y1, x2, y2, x3, y3, x4, y4] (clockwise from top-left)
            segmentation = [[
                x, y,                   # top-left
                x + width, y,           # top-right
                x + width, y + height,  # bottom-right
                x, y + height           # bottom-left
            ]]
            
            # Add segmentation to annotation
            ann['segmentation'] = segmentation
            
            # Update area if not present
            if not ann.get('area'):
                ann['area'] = width * height
                
            updated_count += 1
    
    logger.info(f"Added segmentation from bounding boxes for {updated_count} annotations")
    return annotations

def delete_images_without_annotations(
    coco_data: Dict[str, Any], 
    images_dir: str = None
) -> Dict[str, Any]:
    """
    Delete image files that don't have any annotations and update the COCO data.
    
    Args:
        coco_data: COCO format data dictionary
        images_dir: Directory containing the image files (if None, files won't be deleted)
        
    Returns:
        Dict[str, Any]: Updated COCO data
    """
    # Create a mapping of image_id to annotations
    image_to_annotations = {}
    for ann in coco_data.get('annotations', []):
        image_id = ann.get('image_id')
        if image_id not in image_to_annotations:
            image_to_annotations[image_id] = []
        image_to_annotations[image_id].append(ann)
    
    # Identify images without annotations
    images = coco_data.get('images', [])
    images_with_annotations = []
    images_without_annotations = []
    
    for img in images:
        img_id = img.get('id')
        if img_id in image_to_annotations:
            images_with_annotations.append(img)
        else:
            images_without_annotations.append(img)
    
    # Delete image files if directory is provided
    deleted_files = 0
    if images_dir and os.path.isdir(images_dir):
        for img in images_without_annotations:
            file_name = img.get('file_name')
            if file_name:
                file_path = os.path.join(images_dir, file_name)
                if os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                        deleted_files += 1
                    except Exception as e:
                        logger.error(f"Error deleting file {file_path}: {e}")
    
    # Update COCO data
    coco_data['images'] = images_with_annotations
    
    logger.info(f"Removed {len(images_without_annotations)} images without annotations from JSON")
    if images_dir:
        logger.info(f"Deleted {deleted_files} image files without annotations")
    
    return coco_data

def filter_small_segmentations(
    input_path: str, 
    output_path: str = None, 
    min_area_threshold: float = 100.0,
    visualize: bool = True,
    images_dir: str = None,
    fix_missing_segmentations: bool = True
) -> Dict[str, Any]:
    """
    Filter out annotations with segmentations that are too small.
    
    Args:
        input_path: Path to the input COCO JSON file
        output_path: Path to save the filtered COCO JSON file (if None, will use input_path + '_filtered.json')
        min_area_threshold: Minimum area threshold for segmentations
        visualize: Whether to visualize the distribution of segmentation areas
        images_dir: Directory containing the image files (for deletion if no annotations)
        fix_missing_segmentations: Add segmentation from bbox if missing
    
    Returns:
        Dict: Filtered COCO annotations
    """
    logger.info(f"Reading annotations from {input_path}")
    
    try:
        with open(input_path, 'r') as f:
            coco_data = json.load(f)
    except Exception as e:
        logger.error(f"Error reading JSON file: {e}")
        return None
    
    # Extract annotations
    annotations = coco_data.get('annotations', [])
    original_count = len(annotations)
    logger.info(f"Found {original_count} annotations in the original file")
    
    # Fix missing segmentations using bounding boxes
    if fix_missing_segmentations:
        annotations = add_segmentation_from_bbox(annotations)
        coco_data['annotations'] = annotations
    
    # Calculate areas and dimensions and store them for visualization
    areas = []
    widths = []
    heights = []
    filtered_annotations = []
    removed_count = 0
    
    logger.info("Analyzing segmentation areas and dimensions...")
    for ann in tqdm(annotations):
        segmentation = ann.get('segmentation', [])
        area = calculate_segmentation_area(segmentation)
        width, height = calculate_segmentation_dimensions(segmentation)
        
        # Store metrics for visualization
        areas.append(area)
        widths.append(width)
        heights.append(height)
        
        # Filter based on area threshold
        if area >= min_area_threshold:
            filtered_annotations.append(ann)
        else:
            removed_count += 1
    
    # Update the COCO data with filtered annotations
    coco_data['annotations'] = filtered_annotations
    
    # Delete images without annotations
    coco_data = delete_images_without_annotations(coco_data, images_dir)
    
    # Log statistics
    logger.info(f"Removed {removed_count} annotations with area < {min_area_threshold}")
    logger.info(f"Kept {len(filtered_annotations)} annotations")
    
    # Save filtered data if output path is provided
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_filtered{ext}"
    
    with open(output_path, 'w') as f:
        json.dump(coco_data, f)
    
    logger.info(f"Saved filtered annotations to {output_path}")
    
    # Visualize the distribution of segmentation areas
    if visualize and areas:
        visualize_area_distribution(areas, min_area_threshold)
        visualize_dimension_distribution(widths, heights)
    
    return coco_data

def visualize_area_distribution(areas: List[float], threshold: float) -> None:
    """
    Visualize the distribution of segmentation areas.
    
    Args:
        areas: List of segmentation areas
        threshold: The threshold used for filtering
    """
    plt.figure(figsize=(12, 6))
    
    # Create a DataFrame for easier analysis
    df = pd.DataFrame({'area': areas})
    
    # Plot histogram
    plt.subplot(1, 2, 1)
    plt.hist(df['area'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(x=threshold, color='red', linestyle='--', 
                label=f'Threshold: {threshold}')
    plt.title('Distribution of Segmentation Areas')
    plt.xlabel('Area (pixels²)')
    plt.ylabel('Frequency')
    plt.legend()
    
    # Plot log-scale histogram to better see small values
    plt.subplot(1, 2, 2)
    plt.hist(df['area'], bins=50, alpha=0.7, color='skyblue', edgecolor='black', log=True)
    plt.axvline(x=threshold, color='red', linestyle='--', 
                label=f'Threshold: {threshold}')
    plt.title('Distribution of Segmentation Areas (Log Scale)')
    plt.xlabel('Area (pixels²)')
    plt.ylabel('Frequency (log scale)')
    plt.xscale('log')
    plt.legend()
    
    # Add summary statistics as text
    stats_text = (
        f"Total annotations: {len(areas)}\n"
        f"Min area: {min(areas):.2f}\n"
        f"Max area: {max(areas):.2f}\n"
        f"Mean area: {np.mean(areas):.2f}\n"
        f"Median area: {np.median(areas):.2f}\n"
        f"Annotations below threshold: {sum(a < threshold for a in areas)} "
        f"({sum(a < threshold for a in areas)/len(areas)*100:.2f}%)"
    )
    plt.figtext(0.5, 0.01, stats_text, ha='center', bbox={'facecolor': 'lightgray', 
                                                         'alpha': 0.5, 'pad': 5})
    
    plt.tight_layout()
    plt.savefig('segmentation_area_distribution.png', dpi=300)
    plt.show()
    
    logger.info(f"Saved distribution visualization to segmentation_area_distribution.png")

def visualize_dimension_distribution(widths: List[float], heights: List[float]) -> None:
    """
    Visualize the distribution of segmentation widths and heights.
    
    Args:
        widths: List of segmentation widths
        heights: List of segmentation heights
    """
    plt.figure(figsize=(15, 10))
    
    # Create a DataFrame for easier analysis
    df = pd.DataFrame({'width': widths, 'height': heights})
    
    # Plot width histogram
    plt.subplot(2, 2, 1)
    plt.hist(df['width'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('Distribution of Segmentation Widths')
    plt.xlabel('Width (pixels)')
    plt.ylabel('Frequency')
    
    # Plot height histogram
    plt.subplot(2, 2, 2)
    plt.hist(df['height'], bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
    plt.title('Distribution of Segmentation Heights')
    plt.xlabel('Height (pixels)')
    plt.ylabel('Frequency')
    
    # Plot width vs height scatter plot
    plt.subplot(2, 2, 3)
    plt.scatter(df['width'], df['height'], alpha=0.5, s=10)
    plt.title('Width vs Height of Segmentations')
    plt.xlabel('Width (pixels)')
    plt.ylabel('Height (pixels)')
    
    # Plot aspect ratio histogram
    aspect_ratios = np.array(widths) / np.array(heights)
    aspect_ratios = aspect_ratios[~np.isnan(aspect_ratios) & ~np.isinf(aspect_ratios)]  # Remove NaN and Inf values
    
    plt.subplot(2, 2, 4)
    plt.hist(aspect_ratios, bins=50, alpha=0.7, color='salmon', edgecolor='black')
    plt.title('Distribution of Aspect Ratios (Width/Height)')
    plt.xlabel('Aspect Ratio')
    plt.ylabel('Frequency')
    
    # Add summary statistics as text
    stats_text = (
        f"Width stats:\n"
        f"  Min: {min(widths):.2f}, Max: {max(widths):.2f}\n"
        f"  Mean: {np.mean(widths):.2f}, Median: {np.median(widths):.2f}\n\n"
        f"Height stats:\n"
        f"  Min: {min(heights):.2f}, Max: {max(heights):.2f}\n"
        f"  Mean: {np.mean(heights):.2f}, Median: {np.median(heights):.2f}"
    )
    plt.figtext(0.5, 0.01, stats_text, ha='center', bbox={'facecolor': 'lightgray', 
                                                         'alpha': 0.5, 'pad': 5})
    
    plt.tight_layout()
    plt.savefig('segmentation_dimension_distribution.png', dpi=300)
    plt.show()
    
    logger.info(f"Saved dimension distribution visualization to segmentation_dimension_distribution.png")

if __name__ == "__main__":
    input_file = r"C:\Users\sobha\Desktop\detectron2\Data\RoboFlowData\8.sewer-final.v3i.coco\combined\IV_ManualAnnotation.json"
    output_file = r"C:\Users\sobha\Desktop\detectron2\Data\RoboFlowData\8.sewer-final.v3i.coco\combined\IV_ManualAnnotation_filtered.json"
    images_dir = r"C:\Users\sobha\Desktop\detectron2\Data\RoboFlowData\8.sewer-final.v3i.coco\combined\images"
    
    # Filter small segmentations with a threshold of 100 square pixels
    filter_small_segmentations(
        input_path=input_file,
        output_path=output_file,
        min_area_threshold=100.0,
        visualize=True,
        images_dir=images_dir,
        fix_missing_segmentations=True
    )
