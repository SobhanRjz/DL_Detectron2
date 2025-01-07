import json
import cv2
import matplotlib.pyplot as plt
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from mergeTrainTestFile import DatasetPaths

paths = DatasetPaths()

# Load COCO annotations
with open(paths.output / 'updated_annotations.coco.json', 'r') as f:
    coco_data = json.load(f)

# Load image
import random
import os
image_files = os.listdir(paths.output / "images")
image_path = paths.output / "images" / random.choice(image_files)
image = cv2.imread(str(image_path))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for matplotlib

# Example annotation from COCO
annotations = coco_data['annotations']

import numpy as np

# Load the SAM model
sam = sam_model_registry["default"](str(paths.output / "sam_vit_h_4b8939.pth"))
mask_generator = SamAutomaticMaskGenerator(sam)

def overlay_mask(image, mask, color=(0, 255, 0), alpha=0.5):
    """
    Overlay a binary mask on an image.
    :param image: Original image (numpy array).
    :param mask: Binary mask (same height and width as the image).
    :param color: Color of the mask overlay (BGR).
    :param alpha: Transparency of the overlay (0 to 1).
    :return: Image with the mask overlay.
    """
    overlay = image.copy()
    colored_mask = np.zeros_like(image)
    colored_mask[mask == 1] = color  # Apply color to mask area
    overlay = cv2.addWeighted(colored_mask, alpha, overlay, 1 - alpha, 0)
    return overlay

# Function to process COCO annotations with SAM
def process_coco_with_sam(image, annotations):
    processed_image = image.copy()
    all_masks = []
    
    for annotation in annotations:
        bbox = annotation['bbox']  # COCO bbox format: [x, y, width, height]
        x, y, w, h = map(int, bbox)

        # Crop region within the bounding box
        cropped_region = image[y:y+h, x:x+w]

        # Generate mask using SAM
        masks = mask_generator.generate(cropped_region)

        # Visualize the first mask (simplified for one mask)
        if len(masks) > 0:
            mask = masks[0]['segmentation']  # SAM output is a binary mask
            
            # Convert boolean mask to uint8 before resizing
            mask_uint8 = mask.astype(np.uint8) * 255  # Convert to 0-255 range
            resized_mask = cv2.resize(mask_uint8, (w, h), interpolation=cv2.INTER_NEAREST)
            # Convert back to binary mask
            resized_mask = (resized_mask > 127).astype(np.uint8)

            # Apply mask to cropped region
            cropped_region_with_mask = cv2.bitwise_and(cropped_region, cropped_region, 
                                                      mask=resized_mask.astype(np.uint8))

            # Replace the processed region in the original image
            processed_image[y:y+h, x:x+w] = cropped_region_with_mask
            
            # Store full-size mask for visualization
            full_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            full_mask[y:y+h, x:x+w] = resized_mask
            all_masks.append(full_mask)

    return processed_image, all_masks

# Process the image and get masks
processed_image, masks = process_coco_with_sam(image, annotations)

# Plotting
plt.figure(figsize=(15, 5))

# Original image
plt.subplot(131)
plt.imshow(image)
plt.title('Original Image')
plt.axis('off')

# Processed image
plt.subplot(132)
plt.imshow(processed_image)
plt.title('Processed Image')
plt.axis('off')

# Combined masks
plt.subplot(133)
combined_mask = np.zeros(image.shape[:2], dtype=np.uint8)
for mask in masks:
    combined_mask = cv2.bitwise_or(combined_mask, mask)
plt.imshow(combined_mask, cmap='gray')
plt.title('Combined Masks')
plt.axis('off')

plt.tight_layout()
plt.show()

# Save the processed image
output_image_path = paths.output / "output_with_sam.jpg"
processed_image_bgr = cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)
cv2.imwrite(str(output_image_path), processed_image_bgr)
