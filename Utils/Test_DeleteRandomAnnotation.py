import json
import os
import random
import shutil
from typing import Dict, List, Set
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def delete_random_annotations(json_path: str, images_dir: str, delete_percentage: float = 0.5) -> None:
    """
    Delete a random percentage of images from a COCO JSON annotation file and their corresponding image files.
    
    Args:
        json_path: Path to the COCO JSON annotation file
        images_dir: Directory containing the image files
        delete_percentage: Percentage of images to delete (0.0 to 1.0)
    """
    try:
        # Load the COCO JSON file
        logger.info(f"Loading annotations from {json_path}")
        with open(json_path, 'r') as f:
            coco_data = json.load(f)
        
        # Get the original counts
        original_image_count = len(coco_data['images'])
        original_annotation_count = len(coco_data['annotations'])
        
        logger.info(f"Original dataset: {original_image_count} images, {original_annotation_count} annotations")
        
        # Calculate how many images to delete
        num_to_delete = int(original_image_count * delete_percentage)
        logger.info(f"Will delete {num_to_delete} images ({delete_percentage * 100:.1f}%)")
        
        # Randomly select images to delete
        all_image_ids = [img['id'] for img in coco_data['images']]
        image_ids_to_delete = set(random.sample(all_image_ids, num_to_delete))
        
        # Create a mapping of image IDs to filenames for deletion
        image_id_to_filename = {img['id']: img['file_name'] for img in coco_data['images']}
        
        # Filter out the selected images
        new_images = [img for img in coco_data['images'] if img['id'] not in image_ids_to_delete]
        
        # Filter out annotations for deleted images
        new_annotations = [ann for ann in coco_data['annotations'] 
                          if ann['image_id'] not in image_ids_to_delete]
        
        # Update the COCO data
        coco_data['images'] = new_images
        coco_data['annotations'] = new_annotations
        
        # Save the updated JSON file
        output_json_path = json_path.replace('.json', '_reduced.json')
        with open(output_json_path, 'w') as f:
            json.dump(coco_data, f, indent=2)
        
        logger.info(f"Updated annotations saved to {output_json_path}")
        logger.info(f"New dataset: {len(new_images)} images, {len(new_annotations)} annotations")
        
        # Delete the corresponding image files
        deleted_files_count = 0
        for image_id in image_ids_to_delete:
            filename = image_id_to_filename[image_id]
            image_path = os.path.join(images_dir, filename)
            
            if os.path.exists(image_path):
                try:
                    os.remove(image_path)
                    deleted_files_count += 1
                except Exception as e:
                    logger.error(f"Failed to delete image {image_path}: {str(e)}")
            else:
                logger.warning(f"Image file not found: {image_path}")
        
        logger.info(f"Deleted {deleted_files_count} image files")
        
    except Exception as e:
        logger.error(f"Error processing annotations: {str(e)}")
        raise

if __name__ == "__main__":
    # Path to the COCO JSON annotation file
    json_path = r"C:\Users\sobha\Desktop\detectron2\Data\RoboFlowData\6.Concrete Crack Ver.1.v3i.coco-segmentation\combined\II_FixAnnotation_annotations.coco.json"
    
    # Path to the directory containing the image files
    images_dir = os.path.join(os.path.dirname(json_path), "images")
    
    # Delete 50% of the images and their annotations
    delete_random_annotations(json_path, images_dir, delete_percentage=0.5)
