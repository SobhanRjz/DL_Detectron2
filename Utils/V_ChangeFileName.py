import os
import json
from pathlib import Path
from typing import Dict, List, Optional
import shutil
from I_mergeTrainTestFile import DatasetPaths

class FileRenamer:
    """
    A utility class to rename image files based on their category in COCO annotation format.
    
    This class reads COCO JSON annotations and renames the associated image files
    by prefixing them with their category name, and updates the annotation file accordingly.
    """
    
    def __init__(self):
        """
        Initialize the FileRenamer with paths to annotations and images.
        
        Args:
            annotation_path: Path to the COCO format JSON annotation file
            image_folder: Path to the folder containing images to be renamed
            output_folder: Optional path to output renamed files. If None, files will be renamed in place.
        """
        self.BasePath = r"C:\Users\sobha\Desktop\detectron2\Data\RoboFlowData\8.sewer-final.v3i.coco\combined"
        self.annotation_path = Path(self.BasePath) / "final.json"
        self.image_folder = Path(self.BasePath) / "images"
        self.output_folder = Path(self.BasePath) / "images"
        self.category_map: Dict[int, str] = {}
        self.image_map: Dict[int, Dict] = {}
        self.annotations: List[Dict] = []
        self.annotation_data: Dict = {}
        
        # Validate inputs
        if not self.annotation_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {self.annotation_path}")
        if not self.image_folder.exists():
            raise FileNotFoundError(f"Image folder not found: {self.image_folder}")
        if self.output_folder and not self.output_folder.exists():
            os.makedirs(self.output_folder, exist_ok=True)
    
    def load_annotations(self) -> None:
        """Load and parse the COCO format annotation file."""
        with open(self.annotation_path, 'r') as f:
            self.annotation_data = json.load(f)
        
        # Create category ID to name mapping
        self.category_map = {cat['id']: cat['name'] for cat in self.annotation_data.get('categories', [])}
        
        # Create image ID to image info mapping
        self.image_map = {img['id']: img for img in self.annotation_data.get('images', [])}
        
        # Store annotations
        self.annotations = self.annotation_data.get('annotations', [])
        
        if not self.category_map:
            raise ValueError("No categories found in the annotation file")
        if not self.image_map:
            raise ValueError("No images found in the annotation file")
        if not self.annotations:
            raise ValueError("No annotations found in the annotation file")
    
    def get_image_categories(self) -> Dict[int, List[str]]:
        """
        Get categories for each image.
        
        Returns:
            A dictionary mapping image IDs to lists of category names
        """
        image_categories: Dict[int, List[str]] = {}
        
        for annotation in self.annotations:
            image_id = annotation.get('image_id')
            category_id = annotation.get('category_id')
            
            if image_id is None or category_id is None:
                continue
                
            category_name = self.category_map.get(category_id)
            if not category_name:
                continue
                
            if image_id not in image_categories:
                image_categories[image_id] = []
                
            if category_name not in image_categories[image_id]:
                image_categories[image_id].append(category_name)
                
        return image_categories
    
    def rename_files(self) -> None:
        """Rename image files based on their categories and update annotations."""
        image_categories = self.get_image_categories()
        renamed_count = 0
        filename_mapping = {}  # To track old to new filename mappings
        
        for image_id, categories in image_categories.items():
            if image_id not in self.image_map:
                continue
                
            image_info = self.image_map[image_id]
            original_filename = image_info.get('file_name')
            
            if not original_filename:
                continue
                
            # Create a category prefix by joining all categories with underscore
            category_prefix = "_".join(categories)
            
            # Get the base filename without path
            base_filename = os.path.basename(original_filename)
            # Get the folder name from the original filename
            folder_name = os.path.dirname(self.BasePath).split("\\")[-1]
            # Create new filename with category prefix
            new_filename = f"({folder_name})_{category_prefix}_{image_id}{Path(base_filename).suffix}"
            
            original_path = self.image_folder / base_filename
            
            if not original_path.exists():
                print(f"Warning: Image file not found: {original_path}")
                continue
            

            # Rename in place
            new_path = self.image_folder / new_filename
            os.rename(original_path, new_path)
            
            # Update the filename in the image_map and annotation_data
            filename_mapping[original_filename] = new_filename
            self.image_map[image_id]['file_name'] = new_filename
            
            renamed_count += 1
            print(f"Renamed: {base_filename} -> {new_filename}")
        
        # Update the annotation file with new filenames
        for image in self.annotation_data.get('images', []):
            if image['file_name'] in filename_mapping:
                image['file_name'] = filename_mapping[image['file_name']]
        
        # Save the updated annotation file
        with open(Path(self.BasePath) / "V_renamed_annotation.json", 'w') as f:
            json.dump(self.annotation_data, f, indent=4)
        
        print(f"Renamed {renamed_count} files successfully.")
        print(f"Updated annotation file with new filenames.")


def main():
    """Main function to run the file renamer utility."""

    try:
        renamer = FileRenamer()
        renamer.load_annotations()
        renamer.rename_files()
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
