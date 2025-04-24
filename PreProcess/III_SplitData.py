import os
import json
import random
import shutil
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import train_test_split
from detectron2.structures import BoxMode

class DatasetSplitter:
    def __init__(self, annotation_file, image_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        """
        Initialize the dataset splitter.
        
        Args:
            annotation_file (str): Path to COCO annotation file
            image_dir (str): Path to directory containing images
            output_dir (str): Path to output directory for split datasets
            train_ratio (float): Ratio of data for training (default 0.7)
            val_ratio (float): Ratio of data for validation (default 0.15) 
            test_ratio (float): Ratio of data for testing (default 0.15)
        """
        self.annotation_file = Path(annotation_file)
        self.image_dir = Path(image_dir)
        self.output_dir = Path(output_dir)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        
        # Load annotations
        with open(self.annotation_file) as f:
            self.data = json.load(f)
            
        # Create category-wise image grouping
        self.category_images = defaultdict(list)
        self._group_images_by_category()
        
    def _group_images_by_category(self):
        """Group images by their categories to ensure balanced splitting"""
        image_categories = defaultdict(set)
        
        # Map images to their categories
        for ann in self.data['annotations']:
            image_categories[ann['image_id']].add(ann['category_id'])
            
        # Group images by categories
        for img_id, categories in image_categories.items():
            for cat_id in categories:
                self.category_images[cat_id].append(img_id)
                
    def split_dataset(self):
        """
        Split the dataset while maintaining category balance as much as possible
        """
        train_images = set()
        val_images = set()
        test_images = set()
        
        # Split each category's images separately
        for cat_id, images in self.category_images.items():
            # First split: train vs rest
            train_imgs, temp_imgs = train_test_split(
                images, 
                train_size=self.train_ratio,
                random_state=42
            )
            
            # Second split: val vs test from remaining
            val_ratio_adjusted = self.val_ratio / (self.val_ratio + self.test_ratio)
            val_imgs, test_imgs = train_test_split(
                temp_imgs,
                train_size=val_ratio_adjusted,
                random_state=42
            )
            
            train_images.update(train_imgs)
            val_images.update(val_imgs)
            test_images.update(test_imgs)
            
        # Create split datasets
        self._create_split_dataset('test', test_images)
        self._create_split_dataset('train', train_images)
        self._create_split_dataset('valid', val_images)
        
        
    def _create_split_dataset(self, split_name, image_ids):
        """
        Create a new dataset for the given split
        
        Args:
            split_name (str): Name of the split (train/val/test)
            image_ids (set): Set of image IDs for this split
        """
        # Create output directories
        split_dir = self.output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        (split_dir).mkdir(exist_ok=True)
        
        # Filter annotations and images
        split_annotations = [ann for ann in self.data['annotations'] 
                           if ann['image_id'] in image_ids]
        split_images = [img for img in self.data['images'] 
                       if img['id'] in image_ids]
        
        # Update image IDs and corresponding annotations
        id_map = {}
        for new_id, img in enumerate(split_images, 1):
            id_map[img['id']] = new_id
            img['id'] = new_id

        # Update annotation image IDs and annotation IDs
        for new_id, ann in enumerate(split_annotations, 1):
            ann['image_id'] = id_map[ann['image_id']]
            ann['id'] = new_id
            ann['bbox_mode'] = BoxMode.XYXY_ABS


        # Create new annotation file with updated IDs
        split_data = {
            'images': split_images,
            'annotations': split_annotations,
            'categories': self.data['categories']
        }
        
        # Remove images without annotations
        images_with_annotations = {ann['image_id'] for ann in split_annotations}
        split_images = [img for img in split_images if img['id'] in images_with_annotations]
        split_data['images'] = split_images
        # Save annotation file
        with open(split_dir / '_annotations.coco.json', 'w') as f:
            json.dump(split_data, f, indent=4)
            
        # Copy images
        for img in split_images:
            src = self.image_dir / img['file_name']
            dst = split_dir / img['file_name']
            shutil.copy2(src, dst)
            
        print(f"{split_name} split created with {len(split_images)} images and {len(split_annotations)} annotations")

def main():
    annotation_file = r"C:\Users\sobha\Desktop\detectron2\Data\RoboFlowData\MergeData(1-8)\Final_modified.json"
    image_dir = r"C:\Users\sobha\Desktop\detectron2\Data\RoboFlowData\MergeData(1-8)\images_modified"
    output_dir = r"C:\Users\sobha\Desktop\detectron2\Data\RoboFlowData\MergeData(1-8)\split_dataset"
    
    splitter = DatasetSplitter(annotation_file, image_dir, output_dir)
    splitter.split_dataset()

if __name__ == "__main__":
    main()
