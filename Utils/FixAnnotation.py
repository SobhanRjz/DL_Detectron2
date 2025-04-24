import os
import json
from pathlib import Path
from dataclasses import dataclass
from I_mergeTrainTestFile import DatasetPaths
import numpy as np

@dataclass
class ImageCleaner:
    paths: DatasetPaths
    data: dict
    images_folder: Path
    
    def __init__(self):
        self.paths = DatasetPaths()
        self.images_folder = self.paths.output / "images"
        self.output_folder = self.paths.output
        #self._load_annotations()
            
    def _load_annotations(self, filename="updated_annotations.coco.json"):
        with open(self.output_folder / filename, "r") as f:
            self.data = json.load(f)
            
    def get_annotated_image_ids(self):
        return {ann["image_id"] for ann in self.data["annotations"]}
    
    def remove_categories(self, categories_to_remove):
        # Get the IDs of the categories to remove and create new ID mapping
        category_ids_to_remove = set(cat["id"] for cat in self.data["categories"] 
                                   if cat["name"] in categories_to_remove)
        
        filtered_categories = [cat for cat in self.data["categories"] 
                             if cat["id"] not in category_ids_to_remove]
        new_category_id_mapping = {cat["id"]: i + 1 
                                 for i, cat in enumerate(filtered_categories)}
        
        # Update category IDs and annotations in one pass
        for cat in filtered_categories:
            cat["id"] = new_category_id_mapping[cat["id"]]
        
        self.data["categories"] = filtered_categories
        self.data["annotations"] = [
            {**ann, "category_id": new_category_id_mapping[ann["category_id"]]}
            for ann in self.data["annotations"]
            if ann["category_id"] in new_category_id_mapping
        ]
        
        print(f"Removed categories: {categories_to_remove}")

    def remove_unannotated_images(self, data=None, ShouldDeletePic = False):
        data = data or self.data
        annotated_ids = self.get_annotated_image_ids()
        
        # Split images in one pass
        images_to_keep = []
        images_to_delete = []
        for img in data["images"]:
            (images_to_keep if img["id"] in annotated_ids else images_to_delete).append(img)

        self._delete_images(images_to_delete)
        data["images"] = images_to_keep

        if ShouldDeletePic : 
            # Remove unannotated files efficiently
            annotated_images = {img["file_name"] for img in data["images"]}
            for image_file in os.listdir(self.images_folder):
                if image_file not in annotated_images:
                    (self.images_folder / image_file).unlink()
                    print(f"Removed unannotated image: {image_file}")
                

        self._save_annotations()
        print("Unannotated images removed")
        
    def remove_duplicate_images(self):
        # Group images by base name
        image_base_names = {}
        for img in self.data["images"]:
            base_name = img["file_name"].split(".rf.")[0]
            image_base_names.setdefault(base_name, []).append(img)
            
        # Keep first image, delete rest
        images_to_keep = [duplicates[0] for duplicates in image_base_names.values()]
        duplicates_to_delete = [img for duplicates in image_base_names.values() 
                              for img in duplicates[1:]]
                
        self._delete_images(duplicates_to_delete)
        self.data["images"] = images_to_keep
        self._save_annotations()
        print("Duplicate images removed")
        
    def convert_rectangles_to_polygons(self):
        coco_data = self.data.copy()

        for annotation in coco_data['annotations']:
            if 'bbox' in annotation:
                x, y, width, height = annotation['bbox']
                annotation['segmentation'] = [[
                    x, y,                    # Top-left
                    x + width, y,            # Top-right
                    x + width, y + height,   # Bottom-right
                    x, y + height            # Bottom-left
                ]]

        output_file = self.output_folder / "updated_annotations_polygons.coco.json"
        self._save_json(output_file, coco_data)
        print(f"Converted annotations saved to {output_file}")
        
    def _delete_images(self, images):
        for img in images:
            img_path = self.images_folder / img["file_name"]
            if img_path.exists():
                img_path.unlink()
                print(f"Deleted: {img_path}")
                
    def _save_annotations(self):
        self._save_json(self.paths.output / "updated_annotations.coco.json", self.data)
        
    def _save_json(self, path, data):
        with open(path, "w") as f:
            json.dump(data, f)
        print(f"Saved annotations to {path}")
        
    def update_annotations_with_bbox(self):
        """Update annotations with bounding boxes derived from segmentations if ManualAnnotation.json exists"""
        manual_annotation_path = self.paths.output / "ManualAnnotation.json"
        if not manual_annotation_path.exists():
            return
            
        print("Found ManualAnnotation.json - updating with bounding boxes...")
        coco_data = self._load_json(manual_annotation_path)

        # Update annotations with bounding boxes
        for annotation in coco_data.get('annotations', []):
            if 'segmentation' in annotation:
                segmentation = np.array(annotation['segmentation'][0]).reshape(-1, 2)
                x_min, y_min = segmentation.min(axis=0)
                x_max, y_max = segmentation.max(axis=0)
                annotation['bbox'] = [float(x_min), float(y_min), 
                                    float(x_max - x_min), float(y_max - y_min)]

        self._save_json(self.paths.output / "ManualAnnotation_with_bbox.json", coco_data)
    
    def update_uniqueCategory(self):
        # Track duplicate categories and map category names to IDs
        category_name_to_id = {}
        duplicate_categories = {}
        
        # Find duplicate categories
        for category in self.data['categories']:
            name = category["name"]
            if name in category_name_to_id:
                duplicate_categories[category["id"]] = category_name_to_id[name]
            else:
                category_name_to_id[name] = category["id"]

        # Remove duplicate categories
        unique_categories = [cat for cat in self.data['categories'] 
                           if cat["id"] not in duplicate_categories]

        # Update annotations with the new category IDs
        for ann in self.data["annotations"]:
            if ann["category_id"] in duplicate_categories:
                ann["category_id"] = duplicate_categories[ann["category_id"]]

        # Update the dataset with unique categories
        self.data["categories"] = unique_categories
        
        print(f"Unique categories: {[cat['name'] for cat in unique_categories]}")

    def reindex_image_ids(self):
        """
        Reindex all image_id values in annotations to ensure they match the image IDs.
        This ensures consistency between image IDs and their references in annotations.
        """
        print("Reindexing image IDs in annotations...")
        
        # Create a mapping from old image IDs to new sequential IDs
        old_to_new_image_id = {}
        for new_id, image in enumerate(self.data['images'], 1):
            old_to_new_image_id[image['id']] = new_id
            image['id'] = new_id
        
        # Update all annotations with the new image IDs
        # Filter out annotations with non-existent image IDs and update valid ones
        valid_annotations = []
        for annotation in self.data['annotations']:
            old_id = annotation['image_id']
            if old_id in old_to_new_image_id:
                annotation['image_id'] = old_to_new_image_id[old_id]
                valid_annotations.append(annotation)
            else:
                print(f"Removing annotation with non-existent image ID: {old_id}")
        
        self.data['annotations'] = valid_annotations
        
        self._save_annotations()
        
        print(f"Reindexed {len(old_to_new_image_id)} image IDs in {len(self.data['annotations'])} annotations")
    def _load_json(self, path):
        with open(path) as f:
            return json.load(f)

if __name__ == "__main__":
    cleaner = ImageCleaner()
    basic_annotation_path = cleaner.paths.output / "I_Basic_annotations.coco.json"
    manual_annotation_path = cleaner.paths.output / "final.json" # Should be annotatated by your own
    manual_annotation_path = Path(r"C:\Users\sobha\Desktop\detectron2\Data\RoboFlowData\8.sewer-final.v3i.coco\combined\final.json")
    if not manual_annotation_path.exists():
        cleaner.data = cleaner._load_json(basic_annotation_path)
        cleaner.update_uniqueCategory()
        cleaner.remove_categories(["ddd"])
        cleaner.remove_unannotated_images(ShouldDeletePic=True)
        cleaner.remove_duplicate_images()
        cleaner.convert_rectangles_to_polygons()
    else:
        cleaner.data = cleaner._load_json(manual_annotation_path)
        cleaner.reindex_image_ids()
        #cleaner.remove_unannotated_images(ShouldDeletePic=False)
        # cleaner.update_annotations_with_bbox()

