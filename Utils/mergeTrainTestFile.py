import os
import json
import shutil

# Paths to your folders
validation_path = "C:\\Users\\sobha\\Desktop\\detectron2\\Data\\RoboFlowData\\petro.v7i.coco\\valid"
train_path = "C:\\Users\\sobha\\Desktop\\detectron2\\Data\\RoboFlowData\\petro.v7i.coco\\train"
test_path = "C:\\Users\\sobha\\Desktop\\detectron2\\Data\\RoboFlowData\\petro.v7i.coco\\test"
output_folder = "C:\\Users\\sobha\\Desktop\\detectron2\\Data\\RoboFlowData\\petro.v7i.coco\\combined"

# Create output folders for images and annotations
output_images_folder = os.path.join(output_folder, "images")
output_annotation_file = os.path.join(output_folder, "annotations.json")
os.makedirs(output_images_folder, exist_ok=True)

# Initialize combined COCO JSON structure
combined_annotations = {
    "images": [],
    "annotations": [],
    "categories": []
}

# To keep track of ID offsets
image_id_offset = 0
annotation_id_offset = 0

def merge_folder(folder_path, json_file, image_id_offset, annotation_id_offset):
    """
    Merge the content of a COCO folder into the combined structure.
    """
    # Load the COCO JSON annotation file
    with open(json_file, "r") as f:
        data = json.load(f)

    # Copy categories only once
    if not combined_annotations["categories"]:
        combined_annotations["categories"] = data["categories"]

    # Update image and annotation IDs while copying
    for image in data["images"]:
        old_image_id = image["id"]
        image["id"] += image_id_offset
        combined_annotations["images"].append(image)

        # Copy image files to the output folder
        image_src = os.path.join(folder_path, image["file_name"])
        image_dst = os.path.join(output_images_folder, image["file_name"])
        shutil.copy(image_src, image_dst)

    for annotation in data["annotations"]:
        annotation["id"] += annotation_id_offset
        annotation["image_id"] += image_id_offset
        combined_annotations["annotations"].append(annotation)

    # Return updated offsets
    return image_id_offset + len(data["images"]), annotation_id_offset + len(data["annotations"])

# Process each folder
folders = [
    {"path": validation_path, "json_file": os.path.join(validation_path, "annotations.json")},
    {"path": train_path, "json_file": os.path.join(train_path, "annotations.json")},
    {"path": test_path, "json_file": os.path.join(test_path, "annotations.json")},
]

for folder in folders:
    image_id_offset, annotation_id_offset = merge_folder(folder["path"], folder["json_file"], image_id_offset, annotation_id_offset)

# Save the combined annotations
with open(output_annotation_file, "w") as f:
    json.dump(combined_annotations, f, indent=4)

print(f"Merged annotations saved to {output_annotation_file}")
print(f"All images saved to {output_images_folder}")
