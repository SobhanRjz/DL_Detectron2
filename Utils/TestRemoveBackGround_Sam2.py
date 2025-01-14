import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import json
import random
from pathlib import Path
from mergeTrainTestFile import DatasetPaths
import sys

# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )

np.random.seed(3)

def show_segmentation(segmentation, ax, random_color=False, borders=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    
    # Create empty mask
    h, w = ax.get_figure().get_size_inches() * ax.get_figure().dpi
    mask = np.zeros((int(h), int(w)), dtype=np.uint8)
    
    # Convert polygon points to numpy array
    points = np.array(segmentation).reshape(-1, 2)
    points = points.astype(np.int32)
    
    # Draw filled polygon
    cv2.fillPoly(mask, [points], 1)
    
    mask_image = mask.reshape(mask.shape[0], mask.shape[1], 1) * color.reshape(1, 1, -1)
    
    if borders:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2)
    
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))    

def show_annotations(image, annotation, point_coords=None, input_labels=None, borders=True):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    
    # Show segmentation if it exists
    if 'segmentation' in annotation:
        show_segmentation(annotation['segmentation'][0], plt.gca(), borders=borders)
    
    # Show bounding box
    bbox = annotation['bbox']
    input_box = np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
    show_box(input_box, plt.gca())
    
    if point_coords is not None:
        assert input_labels is not None
        show_points(point_coords, input_labels, plt.gca())
        
    plt.axis('off')
    plt.draw()
    plt.pause(0.001)

input_points = []
input_labels = []
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  # Left click for positive points
        input_points.append([x, y])
        input_labels.append(1)
        print(f"Added positive point at ({x}, {y})")
    elif event == cv2.EVENT_RBUTTONDOWN:  # Right click for negative points
        input_points.append([x, y])
        input_labels.append(0)
        print(f"Added negative point at ({x}, {y})")
    elif event == cv2.EVENT_MBUTTONDOWN:  # Middle click to clear points
        input_points.clear()
        input_labels.clear()
        print("Cleared all points")

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Define paths
paths = DatasetPaths()

# Load COCO annotations
with open(paths.output / 'updated_annotations.coco.json', 'r') as f:
    coco_data = json.load(f)

# Initialize model and predictor
sam2_checkpoint = r"C:\Users\sobha\Desktop\detectron2\Code\sam2\checkpoints\sam2.1_hiera_base_plus.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
predictor = SAM2ImagePredictor(sam2_model)

while True:
    # Load random image
    image_files = os.listdir(paths.output / "images")
    image_path = paths.output / "images" / random.choice(image_files)
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Get annotations for this image
    image_filename = os.path.basename(image_path)
    image_id = None
    for img in coco_data['images']:
        if img['file_name'] == image_filename:
            image_id = img['id']
            break

    image_annotations = []
    if image_id is not None:
        image_annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == image_id]

    predictor.set_image(image)

    # Process each annotation's bounding box
    for annotation in image_annotations:
        bbox = annotation['bbox']  # [x, y, width, height]
        # Convert from COCO format [x,y,w,h] to [x1,y1,x2,y2]
        input_box = np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
        masks, scores, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=False,
        )
        
        # Convert mask to polygon
        mask = masks[0]  # Take first mask
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Get the largest contour
        if len(contours) > 0:
            largest_contour = max(contours, key=cv2.contourArea)
            # Convert contour to polygon format
            polygon = []
            for point in largest_contour.reshape(-1, 2):
                polygon.extend([float(point[0]), float(point[1])])
            
            # Update annotation with polygon segmentation
            annotation['segmentation'] = [polygon]

        show_annotations(image, annotation)
        # Create a figure and connect key press event
        fig = plt.gcf()
        def on_key(event):
            if event.key == 'enter':
                plt.close('all')
            elif event.key == 'escape':
                plt.close('all')
                sys.exit()  # Use sys.exit() instead of break
                
        fig.canvas.mpl_connect('key_press_event', on_key)
        fig.canvas.mpl_connect('button_press_event', lambda event: click_event(
            cv2.EVENT_LBUTTONDOWN if event.button == 1 else 
            cv2.EVENT_RBUTTONDOWN if event.button == 3 else
            cv2.EVENT_MBUTTONDOWN if event.button == 2 else None,
            int(event.xdata) if event.xdata is not None else 0,
            int(event.ydata) if event.ydata is not None else 0,
            None, None))
        plt.show()

print("Done")
