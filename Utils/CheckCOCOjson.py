import json

def validate_coco_json(file_path):
    """
    Validates a COCO format JSON file and returns validation results.
    
    Args:
        file_path (str): Path to the COCO JSON file
        
    Returns:
        dict: Dictionary containing validation results
    """
    try:
        with open(file_path, "r") as file:
            data = json.load(file)
    except Exception as e:
        return {"error": f"Failed to load JSON file: {str(e)}"}

    # Check for required keys
    required_keys = ["images", "annotations", "categories"]
    missing_keys = [key for key in required_keys if key not in data]

    # Initialize counters and validation flags
    segmentation_count = bbox_count = 0
    annotations_valid = images_valid = categories_valid = True

    # Validate annotations and check image coverage
    images_with_annotations = set()
    if "annotations" in data:
        for annotation in data["annotations"]:
            if "segmentation" in annotation and annotation["segmentation"]:
                segmentation_count += 1
            else:
                annotations_valid = False
            if "bbox" in annotation and annotation["bbox"]:
                bbox_count += 1
            else:
                annotations_valid = False
            if "image_id" in annotation:
                images_with_annotations.add(annotation["image_id"])

    # Validate images and check for missing annotations
    images_without_annotations = []
    if "images" in data:
        required_image_keys = ["height", "width", "file_name", "id"] 
        images_valid = all(
            all(key in image for key in required_image_keys)
            for image in data["images"]
        )
        
        for image in data["images"]:
            if image["id"] not in images_with_annotations:
                images_without_annotations.append(image["file_name"])

    # Validate categories
    if "categories" in data:
        required_category_keys = ["id", "name"]
        categories_valid = all(
            all(key in category for key in required_category_keys)
            for category in data["categories"]
        )

    return {
        "missing_keys": missing_keys,
        "segmentation_count": segmentation_count,
        "bbox_count": bbox_count,
        "annotations_valid": annotations_valid,
        "images_valid": images_valid,
        "categories_valid": categories_valid,
        "images_without_annotations": images_without_annotations,
        "total_images": len(data["images"]) if "images" in data else 0,
        "images_with_annotations": len(images_with_annotations)
    }

# File paths to validate
file_paths = [
    r"C:\Users\sobha\Desktop\detectron2\Data\RoboFlowData\MergeData(1-4)\split_dataset\train\_annotations.coco.json",
    r"C:\Users\sobha\Desktop\detectron2\Data\RoboFlowData\MergeData(1-4)\split_dataset\test\_annotations.coco.json", 
    r"C:\Users\sobha\Desktop\detectron2\Data\RoboFlowData\MergeData(1-4)\split_dataset\valid\_annotations.coco.json"
]

# Validate each file
results = {}
for path in file_paths:
    results[path] = validate_coco_json(path)

results
