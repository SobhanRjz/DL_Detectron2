import os
import shutil
import json
from collections import defaultdict
from mergeTrainTestFile import DatasetPaths
from pathlib import Path
from PIL import Image
RootPath = 'C:/Users/sobha/Desktop/detectron2/Data/RoboFlowData'
# Define paths and folders
base_folders = [
    Path(RootPath) / '1.petro.v7i.coco',
    Path(RootPath) / '2. PipeMonitor.v1i.coco', 
    Path(RootPath) / '3.culv_data_6_class.v5i.coco',
    Path(RootPath) / '4.aaaaaaaa.v2i.coco'
]

# Output paths
output_folder = Path(RootPath) / 'MergeData(1-4)'
output_images_folder = output_folder / 'images'
output_annotation_file = output_folder / 'Final.json'

# Track duplicate categories and map category names to IDs
category_name_to_id = {}
duplicate_categories = {}

# Find duplicate categories
for folder in base_folders:
    annotation_file = folder / 'combined' / 'Final.json'
    with open(annotation_file) as f:
        data = json.load(f)
        for category in data['categories']:
            name = category["name"]
            if name not in category_name_to_id:
                category_name_to_id[name] = category["id"]

# Create output folders if they don't exist
os.makedirs(output_images_folder, exist_ok=True)

# Initialize merged COCO annotation structure
merged_annotations = {
    "info": {},
    "licenses": [],
    "images": [],
    "annotations": [],
    "categories": []
}

# Dictionary to map category names to IDs
category_name_to_id = {}
next_category_id = 1

# Function to merge categories
def get_category_id(category_name):
    global next_category_id
    if category_name not in category_name_to_id:
        category_name_to_id[category_name] = next_category_id
        next_category_id += 1
    return category_name_to_id[category_name]

# Iterate through each folder
for folder in base_folders:
    # Path to the current folder's images and annotation file
    images_folder = os.path.join(folder, 'combined','images')
    annotation_file = os.path.join(folder, 'combined','Final.json')

    # Copy images to the output folder
    for image_name in os.listdir(images_folder):
        src_image_path = os.path.join(images_folder, image_name)
        dst_image_path = os.path.join(output_images_folder, image_name)
        if not os.path.exists(dst_image_path):
            shutil.copy(src_image_path, dst_image_path)

    # Load the annotation file
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)

    # Merge categories
    for category in annotations['categories']:
        category_name = category['name']
        category_id = get_category_id(category_name)
        if category_id > len(merged_annotations['categories']):
            merged_annotations['categories'].append({
                "id": category_id,
                "name": category_name,
                "supercategory": category.get('supercategory', '')
            })

    # Merge images and annotations
    for image in annotations['images']:
        # Update image file path and get dimensions
        image_path = os.path.join(images_folder, os.path.basename(image['file_name']))
        img = Image.open(image_path)
        width, height = img.size
        
        # Update image properties
        image['file_name'] = os.path.join('images', os.path.basename(image['file_name']))
        image['width'] = width
        image['height'] = height
        image['id'] = len(merged_annotations['images']) + 1
        
        merged_annotations['images'].append(image)

    for annotation in annotations['annotations']:
        # Update category ID
        category_name = next(cat['name'] for cat in annotations['categories'] if cat['id'] == annotation['category_id'])
        annotation['category_id'] = get_category_id(category_name)
        merged_annotations['annotations'].append(annotation)

# Save the merged annotations to the output file
with open(output_annotation_file, 'w') as f:
    json.dump(merged_annotations, f, indent=4)

print(f"Merged images and annotations saved to {output_folder}")