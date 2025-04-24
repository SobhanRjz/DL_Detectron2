import json
from pathlib import Path
import os


class CategoryDeleter:
    def __init__(self, annotation_file):
        """
        Initialize the category deleter.
        
        Args:
            annotation_file (str): Path to COCO annotation JSON file
        """
        
        self.annotation_file = Path(annotation_file)
        self.data = None
        
    def load_annotations(self):
        """Load the COCO annotation file"""
        with open(self.annotation_file, 'r') as f:
            self.data = json.load(f)
            
    def delete_categories(self, categories_to_delete):
        """
        Delete specified categories and their annotations.
        
        Args:
            categories_to_delete (list): List of category names to delete
        """
        if self.data is None:
            self.load_annotations()
            
        # Get IDs of categories to delete
        category_ids_to_delete = set()
        filtered_categories = []
        category_id_mapping = {}  # Map old category IDs to new ones
        
        for cat in self.data['categories']:
            if cat['name'] in categories_to_delete:
                category_ids_to_delete.add(cat['id'])
            else:
                filtered_categories.append(cat)
                
        # Update category IDs to maintain consecutive numbering
        for i, cat in enumerate(filtered_categories, 1):
            category_id_mapping[cat['id']] = i  # Store mapping of old ID to new ID
            cat['id'] = i
            
        # Filter out annotations for deleted categories and update category IDs
        filtered_annotations = []
        for ann in self.data['annotations']:
            if ann['category_id'] not in category_ids_to_delete:
                # Update category ID using the mapping
                ann['category_id'] = category_id_mapping[ann['category_id']]
                filtered_annotations.append(ann)
                
        # Update the data
        self.data['categories'] = filtered_categories
        self.data['annotations'] = filtered_annotations
        
        print(f"Deleted categories: {categories_to_delete}")
        print(f"Remaining categories: {[cat['name'] for cat in filtered_categories]}")
        
    def save_annotations(self, output_file=None):
        """
        Save the modified annotations to file.
        
        Args:
            output_file (str, optional): Path to save modified annotations. 
                                       If None, overwrites input file.
        """
        if output_file is None:
            # Generate new filename by adding '_modified' before extension
            base, ext = os.path.splitext(self.annotation_file)
            output_file = f"{base}_modified{ext}"
            
        with open(output_file, 'w') as f:
            json.dump(self.data, f, indent=4)
        print(f"Saved modified annotations to {output_file}")
        
    def remove_empty_annotations(self):
        """
        Remove annotations without valid segmentation or bbox data and their corresponding images.
        
        Returns:
            tuple: Number of removed annotations and images
        """
        # Keep track of images with valid annotations
        images_with_annotations = set()
        filtered_annotations = []
        
        for ann in self.data['annotations']:
            if 'segmentation' in ann and ann['segmentation'] and 'bbox' in ann and ann['bbox']:
                filtered_annotations.append(ann)
                images_with_annotations.add(ann['image_id'])
        
        # Filter out images without valid annotations
        filtered_images = [img for img in self.data['images'] 
                        if img['id'] in images_with_annotations]
        
        removed_annotations = len(self.data['annotations']) - len(filtered_annotations)
        removed_images = len(self.data['images']) - len(filtered_images)
        
        self.data['annotations'] = filtered_annotations
        self.data['images'] = filtered_images
        
        return removed_annotations, removed_images

    def copy_images_to_folder(self):
        
        """
        Copy images that exist in annotations to a new folder.
        
        Args:
            source_image_dir (str): Directory containing source images
            target_image_dir (str): Directory to copy images to
        """
        import shutil
        # Create target directory with new folder name if it doesn't exist
        
        base_dir = os.path.dirname(self.annotation_file)
        source_image_dir = base_dir + "\images"
        target_image_dir = os.path.join(base_dir, "images_modified")

        os.makedirs(target_image_dir, exist_ok=True)
        os.makedirs(os.path.join(target_image_dir, "images"), exist_ok=True)

        # Get list of image filenames from annotations
        image_files = [img['file_name'] for img in self.data['images']]
        
        copied_count = 0
        for image_file in image_files:
            source_path = os.path.join(source_image_dir, image_file)
            target_path = os.path.join(target_image_dir, image_file)
            
            if os.path.exists(source_path):
                # Copy the image file
                shutil.copy2(source_path, target_path)
                copied_count += 1
            else:
                print(f"Warning: Source image not found: {source_path}")
                
        print(f"Copied {copied_count} images to {target_image_dir}")
def main():
    # Example usage
    annotation_file = r"C:\Users\sobha\Desktop\detectron2\Data\RoboFlowData\MergeData(1-8)\Final.json"
    categories_to_delete = ["Vermin", "Junction", "Hole", "Sealing Ring"]
    
    deleter = CategoryDeleter(annotation_file)
    deleter.delete_categories(categories_to_delete)
    deleter.remove_empty_annotations()
    deleter.copy_images_to_folder()
    
    deleter.save_annotations()

if __name__ == "__main__":
    main()
