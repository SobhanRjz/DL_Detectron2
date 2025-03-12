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
import sys
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

class SAM2PolygonConverter:
    def __init__(self):
        self.device = self._get_device()
        self._setup_device()
        self.input_points = []
        self.input_labels = []
        
        # Initialize model and predictor
        self.sam2_checkpoint = r"C:\Users\sobha\Desktop\detectron2\Code\sam2\checkpoints\sam2.1_hiera_large.pt"
        self.model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        self.sam2_model = build_sam2(self.model_cfg, self.sam2_checkpoint, device=self.device)
        self.predictor = SAM2ImagePredictor(self.sam2_model)
        
        # Set paths
        self.folder_images = Path(r"C:\Users\sobha\Desktop\detectron2\Data\RoboFlowData\5.pipeline.v1i.coco-segmentation\combined")
        self.final_json_modify_path = self.folder_images / "IV_Sam2_annotations.coco.json"
        self.final_json_path = self.folder_images / "III_AiDetection_annotations.coco.json"

    def _get_device(self):
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _setup_device(self):
        print(f"using device: {self.device}")
        if self.device.type == "cuda":
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        elif self.device.type == "mps":
            print("\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
                  "give numerically different outputs and sometimes degraded performance on MPS. "
                  "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion.")

    @staticmethod
    def show_segmentation(segmentation, ax, random_color=False, borders=True):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])
        
        h, w = ax.get_figure().get_size_inches() * ax.get_figure().dpi
        mask = np.zeros((int(h), int(w)), dtype=np.uint8)
        
        points = np.array(segmentation).reshape(-1, 2).astype(np.int32)
        cv2.fillPoly(mask, [points], 1)
        
        mask_image = mask.reshape(mask.shape[0], mask.shape[1], 1) * color.reshape(1, 1, -1)
        
        if borders:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2)
        
        ax.imshow(mask_image)

    def process_images(self):
        import time
        with open(self.final_json_path, 'r') as f:
            final_annotations = json.load(f)

        image_files = list((self.folder_images / "images").iterdir())
        total_images = len(image_files)
        
        # Process first image to estimate time
        start_time = time.time()
        first_image = image_files[0]
        
        # Process first image
        image = cv2.imread(str(first_image))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_id = next((img['id'] for img in final_annotations['images'] 
                       if img['file_name'] == first_image.name), None)
        if image_id is not None:
            image_annotations = [ann for ann in final_annotations['annotations'] 
                               if ann['image_id'] == image_id]
            self.predictor.set_image(image)
            self._process_annotations(image_annotations, final_annotations)
            
        # Calculate estimated total time
        time_per_image = time.time() - start_time
        total_estimated_time = time_per_image * total_images
        print(f"Estimated total time: {total_estimated_time/60:.1f} minutes")

        # Process remaining images
        for idx, image_file in enumerate(image_files, 1):
            print(f"Processing {image_file.name}... ({idx}/{total_images})")
            estimated_remaining = (total_images - idx) * time_per_image
            print(f"Estimated time remaining: {estimated_remaining/60:.1f} minutes")
            
            # Load and process image
            image = cv2.imread(str(image_file))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Get image ID and annotations
            image_id = next((img['id'] for img in final_annotations['images'] 
                           if img['file_name'] == image_file.name), None)
            
            if image_id is None:
                print(f"No annotations found for {image_file.name}")
                continue

            image_annotations = [ann for ann in final_annotations['annotations'] 
                               if ann['image_id'] == image_id]

            self.predictor.set_image(image)
            self._process_annotations(image_annotations, final_annotations)

        # Save updated annotations
        with open(self.final_json_modify_path, 'w') as f:
            json.dump(final_annotations, f)
        
        print("Done updating annotations")

    def _process_annotations(self, image_annotations, final_annotations):
        for annotation in image_annotations:
            bbox = annotation['bbox']
            input_box = np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
            
            masks, scores, _ = self.predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_box[None, :],
                multimask_output=False,
            )
            
            mask = masks[0]
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, 
                                         cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                polygon = [float(x) for point in largest_contour.reshape(-1, 2) 
                          for x in point]
                
                matching_annotation = next((final_ann for final_ann in final_annotations['annotations']
                                            if final_ann['image_id'] == annotation['image_id'] and 
                                            final_ann['id'] == annotation['id']), None)
                if matching_annotation is not None:
                    matching_annotation['segmentation'] = [polygon]

if __name__ == "__main__":
    np.random.seed(3)
    converter = SAM2PolygonConverter()
    converter.process_images()
