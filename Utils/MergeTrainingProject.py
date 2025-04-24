import os
import shutil
import json
from collections import defaultdict
from pathlib import Path
from PIL import Image
RootPath = 'C:/Users/sobha/Desktop/detectron2/Data/RoboFlowData'
# Define paths and folders
base_folders = [
    Path(RootPath) / '1.petro.v7i.coco',
    Path(RootPath) / '2. PipeMonitor.v1i.coco', 
    Path(RootPath) / '3.culv_data_6_class.v5i.coco',
    Path(RootPath) / '4.aaaaaaaa.v2i.coco',
    Path(RootPath) / '5.pipeline.v1i.coco-segmentation',
    Path(RootPath) / '6.Concrete Crack Ver.1.v3i.coco-segmentation',
    Path(RootPath) / '7.divide-data_1X.v1-camere_pipe_75-25.coco',
    Path(RootPath) / '8.sewer-final.v3i.coco'
]

# Output paths
output_folder = Path(RootPath) / 'MergeData(1-8)'
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
    # Get paths for current folder
    images_folder = folder / 'combined' / 'images'
    annotation_file = folder / 'combined' / 'Final.json'

    # Copy new images to output folder
    for image_path in images_folder.iterdir():
        dest_path = output_images_folder / image_path.name
        if not dest_path.exists():
            shutil.copy(image_path, dest_path)

    # Load and process annotations
    with open(annotation_file) as f:
        annotations = json.load(f)

    # Process categories
    for category in annotations['categories']:
        category_name = category['name']
        category_id = get_category_id(category_name)
        
        if category_id > len(merged_annotations['categories']):
            merged_annotations['categories'].append({
                "id": category_id,
                "name": category_name,
                "supercategory": category.get('supercategory', '')
            })

    # Process images
    image_id_map = {}  # Map old image IDs to new ones
    for image in annotations['images']:
        # Get image dimensions
        img_path = images_folder / image['file_name']
        with Image.open(img_path) as img:
            width, height = img.size
        
        # Create new image entry
        new_image_id = len(merged_annotations['images']) + 1
        image_id_map[image['id']] = new_image_id
        
        merged_annotations['images'].append({
            'id': new_image_id,
            'file_name': image['file_name'],
            'width': width,
            'height': height
        })

    # Process annotations
    for annotation in annotations['annotations']:
        # Map category ID
        old_category_id = annotation['category_id']
        category_name = next(cat['name'] for cat in annotations['categories'] 
                           if cat['id'] == old_category_id)
        new_category_id = get_category_id(category_name)
        
        # Create new annotation with mapped IDs
        annotation['category_id'] = new_category_id
        annotation['image_id'] = image_id_map[annotation['image_id']]
        merged_annotations['annotations'].append(annotation)

# Save the merged annotations to the output file
with open(output_annotation_file, 'w') as f:
    json.dump(merged_annotations, f)

print(f"Merged images and annotations saved to {output_folder}")