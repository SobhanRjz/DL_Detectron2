import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import cv2
import json
import random
from pathlib import Path
import sys
import time
import logging
from typing import List, Dict, Tuple, Optional, Union, Any
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('sam2_polygon_converter.log')
    ]
)
logger = logging.getLogger('SAM2PolygonConverter')

class SAM2PolygonConverter:
    """
    A professional tool for converting bounding box annotations to polygon segmentations
    using SAM2 (Segment Anything Model 2) with interactive point selection capabilities.
    
    This converter supports:
    - Interactive point selection on images
    - Automatic checkpoint saving and resuming
    - Ctrl+S for manual saves
    - Visualization of segmentation results
    """
    
    # Modern color scheme
    COLORS = {
        'foreground': {'color': '#FF5757', 'alpha': 0.8},  # Vibrant red
        'background': {'color': '#4A7CFF', 'alpha': 0.8},  # Vibrant blue
        'bbox': {'color': '#32CD32', 'alpha': 0.8},        # Lime green
        'segmentation': {'color': '#FF9E3D', 'alpha': 0.5},# Orange
        'highlight': {'color': '#FFDE59', 'alpha': 0.9},   # Yellow highlight
    }
    
    def __init__(self) -> None:
        """Initialize the SAM2PolygonConverter with model, paths, and state tracking."""
        self.device = self._get_device()
        self._setup_device()
        self.input_points: List[List[int]] = []
        self.input_labels: List[int] = []
        self.current_annotation: Optional[Dict[str, Any]] = None
        self.current_image: Optional[np.ndarray] = None
        self.fig: Optional[plt.Figure] = None
        self.ax: Optional[plt.Axes] = None
        self.all_annotations_obj: Optional[Dict[str, Any]] = None
        self.all_annotations: Optional[List[Dict[str, Any]]] = None
        self.all_images: Optional[List[Dict[str, Any]]] = None
        self.category_map: Optional[Dict[int, str]] = None
        self.checkpoint: Optional[Dict[str, Any]] = None
        
        # Initialize model and predictor
        logger.info("Initializing SAM2 model and predictor...")
        self.sam2_checkpoint = r"C:\Users\sobha\Desktop\detectron2\Code\sam2\checkpoints\sam2.1_hiera_large.pt"
        self.model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        self.sam2_model = build_sam2(self.model_cfg, self.sam2_checkpoint, device=self.device)
        self.predictor = SAM2ImagePredictor(self.sam2_model)
        
        # Set paths
        self.folder_images = Path(r"C:\Users\sobha\Desktop\detectron2\Data\RoboFlowData\5.pipeline.v1i.coco-segmentation\combined")
        self.final_json_modify_path = self.folder_images / "Final_Pro_Polygon.json"
        self.final_json_path = self.folder_images / "III_AiDetection_annotations.coco.json"
        self.checkpoint_path = self.folder_images / "annotation_checkpoint.txt"
        
        # Point visualization params
        self.point_size = 10
        self.point_border_size = 1.5
        self.hover_point: Optional[int] = None
        self.point_artists: List[Any] = []

    def _get_device(self) -> torch.device:
        """Determine the best available device for processing."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _setup_device(self) -> None:
        """Configure device-specific settings."""
        logger.info(f"Using device: {self.device}")
        if self.device.type == "cuda":
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        elif self.device.type == "mps":
            logger.warning(
                "Support for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
                "give numerically different outputs and sometimes degraded performance on MPS."
            )

    def show_segmentation(self, segmentation: List[List[float]], ax: plt.Axes, 
                          random_color: bool = False, borders: bool = True) -> None:
        """
        Display segmentation polygon on the given matplotlib axis.
        
        Args:
            segmentation: List of coordinates forming the polygon
            ax: Matplotlib axis to draw on
            random_color: Whether to use random colors
            borders: Whether to draw borders around the segmentation
        """
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([self.COLORS['segmentation']['alpha']])], axis=0)
        else:
            # Convert hex to rgba
            hex_color = matplotlib.colors.to_rgba(self.COLORS['segmentation']['color'], 
                                                 self.COLORS['segmentation']['alpha'])
            color = np.array(hex_color)
        
        h, w = ax.get_figure().get_size_inches() * ax.get_figure().dpi
        mask = np.zeros((int(h), int(w)), dtype=np.uint8)
        
        points = np.array(segmentation).reshape(-1, 2).astype(np.int32)
        cv2.fillPoly(mask, [points], 1)
        
        mask_image = mask.reshape(mask.shape[0], mask.shape[1], 1) * color.reshape(1, 1, -1)
        
        if borders:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            border_color = (1, 1, 1, 0.8)  # White borders with high alpha
            line_width = 2
            mask_image = cv2.drawContours(mask_image, contours, -1, border_color, thickness=line_width)
        
        ax.imshow(mask_image)

    def _save_checkpoint(self, image_file: Path) -> None:
        """
        Save progress checkpoint to resume later.
        
        Args:
            image_file: Path to the currently processed image
        """
        # Create or update checkpoint with current information
        checkpoint = {
            'image_names': []
        }
        
        # Try to read existing checkpoint to preserve history if needed
        if self.checkpoint_path.exists():
            try:
                with open(self.checkpoint_path, 'r') as f:
                    existing = json.load(f)
                    if 'image_names' in existing:
                        checkpoint['image_names'] = existing['image_names']
            except Exception as e:
                logger.error(f"Error reading existing checkpoint: {e}")
        
        # Add current image to history if not already there
        if image_file.name not in checkpoint['image_names']:
            checkpoint['image_names'].append(image_file.name)
        
        # Write updated checkpoint
        with open(self.checkpoint_path, 'w') as f:
            json.dump(checkpoint, f)
        logger.info(f"Checkpoint saved: Image {image_file.name}")
    
    def _load_checkpoint(self) -> Dict[str, Any]:
        """
        Load checkpoint if exists.
        
        Returns:
            Dictionary with checkpoint data or empty dict if no checkpoint exists
        """
        if not self.checkpoint_path.exists():
            logger.info("No checkpoint found. Starting from beginning.")
            return {'image_names': []}
        
        try:
            with open(self.checkpoint_path, 'r') as f:
                checkpoint = json.load(f)
            if 'image_names' in checkpoint and checkpoint['image_names']:
                logger.info(f"Checkpoint found. {len(checkpoint['image_names'])} images already processed.")
                logger.info(f"Resuming after image: {checkpoint['image_names'][-1]}")
            return checkpoint
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            return {'image_names': []}
    
    def process_images(self) -> None:
        """Process all images in the dataset, applying SAM2 to generate polygon segmentations."""
        logger.info("Starting image processing...")
        with open(self.final_json_path, 'r') as f:
            final_annotations = json.load(f)

        # Store reference to full annotations for manual saving
        self.all_annotations_obj = final_annotations
        self.all_annotations = final_annotations['annotations']
        self.all_images = final_annotations['images']

        # Create a mapping of category IDs to category names
        self.category_map = {cat['id']: cat['name'] for cat in final_annotations['categories']}

        image_files = list((self.folder_images / "images").iterdir())
        total_images = len(image_files)
        
        # Check for checkpoint
        self.checkpoint = self._load_checkpoint()
        if not self.checkpoint:
            self.checkpoint = {'image_names': []}

        # Process images
        start_time = time.time()
        processed_count = 0
        
        for idx, image_file in enumerate(image_files):
            logger.info(f"Processing {image_file.name}... ({idx+1}/{total_images})")
            
            if 'image_names' in self.checkpoint and image_file.name in self.checkpoint['image_names']:
                logger.info(f"Skipping {image_file.name} as it's already processed")
                continue
                
            # Load and process image
            image = cv2.imread(str(image_file))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.current_image = image

            # Get image ID and annotations
            image_id = next((img['id'] for img in final_annotations['images'] 
                           if img['file_name'] == image_file.name), None)
            
            if image_id is None:
                logger.warning(f"No annotations found for {image_file.name}")
                continue

            image_annotations = [ann for ann in final_annotations['annotations'] 
                               if ann['image_id'] == image_id]

            self.predictor.set_image(image)

            self._process_image_annotations(image_file, image_annotations, final_annotations)
            self._save_checkpoint(image_file)
            processed_count += 1
            
            # Calculate and log remaining time
            elapsed_time = time.time() - start_time
            if processed_count > 0:
                avg_time_per_image = elapsed_time / processed_count
                remaining_images = total_images - idx - 1
                remaining_time = avg_time_per_image * remaining_images
                logger.info(f"Estimated remaining time: {remaining_time/60:.1f} minutes "
                           f"({remaining_time/3600:.1f} hours)")

        # Save updated annotations
        with open(self.final_json_modify_path, 'w') as f:
            json.dump(final_annotations, f)
        
        # Clean up checkpoint if completed successfully
        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()
            logger.info("Checkpoint file removed as processing completed successfully")
            
        logger.info(f"Processing complete! {processed_count} images processed.")
    def _process_image_annotations(self, image_file: Path, 
                                  image_annotations: List[Dict[str, Any]], 
                                  final_annotations: Dict[str, Any]) -> None:
        """
        Process all annotations for a single image, with checkpoint support.
        
        Args:
            image_file: Path to the image file
            image_annotations: List of annotations for this image
            final_annotations: Complete annotations dictionary to update
        """
        total_annotations = len(image_annotations)
        logger.info(f"Processing {total_annotations} annotations for {image_file.name}")
        
        # Process each annotation
        for ann_idx, annotation in enumerate(image_annotations):
            logger.info(f"Processing annotation {ann_idx+1}/{total_annotations}")
            
            # Check if annotation already has segmentation
            if 'segmentation' in annotation and annotation['segmentation']:
                # Check if segmentation is the same as bbox (rectangle)
                bbox = annotation['bbox']
                x1, y1, w, h = [int(v) for v in bbox]
                x2, y2 = x1 + w, y1 + h
                
                # Create a rectangle polygon from the bounding box
                bbox_polygon = [float(x1), float(y1), float(x2), float(y1), 
                              float(x2), float(y2), float(x1), float(y2)]
                
                # If segmentation is already the same as the bounding box, skip
                if annotation['segmentation'] == [bbox_polygon]:
                    logger.info(f"Annotation {ann_idx+1} already has segmentation identical to bbox, skipping")
                    continue
                
                # Check if segmentation has too many points (over 100), skip if so
                if isinstance(annotation['segmentation'], list) and len(annotation['segmentation']) > 0:
                    # Check if it's a list of lists (multiple polygons) or a single polygon
                    if isinstance(annotation['segmentation'][0], list):
                        total_points = sum(len(poly) // 2 for poly in annotation['segmentation'])
                    else:
                        total_points = len(annotation['segmentation'][0]) // 2  # Each point has x,y coords
                    
                    if total_points > 50:
                        logger.info(f"Annotation {ann_idx+1} already has complex segmentation with {total_points} points, skipping")
                        continue
            
            bbox = annotation['bbox']
            input_box = np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
            
            result_accepted = False
            stored_points = []
            stored_labels = []
            
            while not result_accepted:
                # Get user input points
                point_coords = None
                point_labels = None
                
                # Use stored points if available from previous rejected attempt
                if stored_points and stored_labels:
                    self.input_points = stored_points.copy()
                    self.input_labels = stored_labels.copy()
                
                points, labels = self._select_points(annotation)
                
                # If no points selected, use bounding box directly as the mask
                if not points:
                    logger.info("No points selected, using bounding box as mask")
                    x1, y1, w, h = [int(v) for v in bbox]
                    x2, y2 = x1 + w, y1 + h
                    
                    # Create a rectangle polygon from the bounding box
                    polygon = [float(x1), float(y1), float(x2), float(y1), 
                              float(x2), float(y2), float(x1), float(y2)]
                    
                    # Show the result
                    fig, ax = plt.subplots(figsize=(10, 10))
                    ax.imshow(self.current_image)
                    self.show_segmentation([polygon], ax)
                    
                    # Add category name
                    category_id = annotation['category_id']
                    class_name = self.category_map.get(category_id, f"Unknown (ID: {category_id})")
                    
                    # Add a colored box with the class name
                    textstr = f"Class: {class_name}\nUsing bounding box"
                    props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7)
                    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
                            verticalalignment='top', bbox=props)
                    
                    plt.title("Bounding Box Result (Enter=Accept, R=Reject and reselect points)")
                    plt.tight_layout()
                    
                    # Variable to track user decision
                    user_decision = {'action': None}
                    
                    # Set up event handlers
                    def on_key_press(event):
                        if event.key == 'enter':
                            user_decision['action'] = 'accept'
                            plt.close(fig)
                        elif event.key == 'r':
                            user_decision['action'] = 'reject'
                            plt.close(fig)
                    
                    fig.canvas.mpl_connect('key_press_event', on_key_press)
                    plt.show(block=True)
                    
                    if user_decision['action'] == 'accept':
                        # User accepted the result, save it
                        for final_ann in final_annotations['annotations']:
                            if (final_ann['image_id'] == annotation['image_id'] and 
                                final_ann['id'] == annotation['id']):
                                final_ann['segmentation'] = [polygon]
                                break
                        result_accepted = True
                        logger.info("Bounding box segmentation accepted")
                    else:
                        logger.info("Bounding box segmentation rejected, returning to point selection")
                else:
                    # Use SAM with the selected points
                    point_coords = np.array(points)
                    point_labels = np.array(labels)
                    logger.info(f"Using {len(points)} user-selected points")
                    
                    # Call predict with points
                    masks, scores, _ = self.predictor.predict(
                        point_coords=point_coords,
                        point_labels=point_labels,
                        box=input_box[None, :],
                        multimask_output=False,
                    )
                    
                    mask = masks[0]
                    
                    # Force mask to be within the bounding box
                    mask_bbox = np.zeros_like(mask)
                    x1, y1, w, h = [int(v) for v in bbox]
                    x2, y2 = x1 + w, y1 + h
                    # Create a mask for the bounding box area
                    mask_bbox[y1:y2, x1:x2] = 1
                    # Intersect the predicted mask with the bounding box mask
                    mask = np.logical_and(mask, mask_bbox).astype(np.uint8)
                    
                    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, 
                                                 cv2.CHAIN_APPROX_SIMPLE)
                    
                    if contours:
                        largest_contour = max(contours, key=cv2.contourArea)
                        polygon = [float(x) for point in largest_contour.reshape(-1, 2) 
                                  for x in point]
                        
                        # Show the result
                        fig, ax = plt.subplots(figsize=(10, 10))
                        ax.imshow(self.current_image)
                        self.show_segmentation([polygon], ax)
                        
                        # Add category name and confidence score
                        category_id = annotation['category_id']
                        class_name = self.category_map.get(category_id, f"Unknown (ID: {category_id})")
                        confidence = scores[0] if scores is not None else 0.0
                        
                        # Add a colored box with the class name and confidence
                        textstr = f"Class: {class_name}\nConfidence: {confidence:.2f}"
                        props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7)
                        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
                                verticalalignment='top', bbox=props)
                        
                        plt.title("Segmentation Result (Enter=Accept, R=Reject and reselect points)")
                        plt.tight_layout()
                        
                        # Variable to track user decision
                        user_decision = {'action': None}
                        
                        # Set up event handlers
                        def on_key_press(event):
                            if event.key == 'enter':
                                user_decision['action'] = 'accept'
                                plt.close(fig)
                            elif event.key == 'r':
                                user_decision['action'] = 'reject'
                                plt.close(fig)
                        
                        fig.canvas.mpl_connect('key_press_event', on_key_press)
                        plt.show(block=True)
                        
                        if user_decision['action'] == 'accept':
                            # User accepted the result, save it
                            for final_ann in final_annotations['annotations']:
                                if (final_ann['image_id'] == annotation['image_id'] and 
                                    final_ann['id'] == annotation['id']):
                                    final_ann['segmentation'] = [polygon]
                                    break
                            result_accepted = True
                            logger.info("Segmentation accepted")
                        else:
                            # User rejected, store points for reuse
                            stored_points = self.input_points.copy()
                            stored_labels = self.input_labels.copy()
                            logger.info("Segmentation rejected, returning to point selection")
                    else:
                        logger.warning("No contours found, please try different points")
                        
                        # Store points for reuse
                        stored_points = self.input_points.copy()
                        stored_labels = self.input_labels.copy()


    def _select_points(self, annotation: Dict[str, Any]) -> Tuple[List[List[int]], List[int]]:
        """
        Allow user to select points on the image for the current annotation.
        
        Args:
            annotation: The annotation to process
            
        Returns:
            Tuple of (points, labels) selected by the user
        """
        self.input_points = []
        self.input_labels = []
        self.current_annotation = annotation
        self.point_artists = []
        
        # Get the class name for this annotation
        category_id = annotation['category_id']
        class_name = self.category_map.get(category_id, f"Unknown (ID: {category_id})")
        
        # Setup plot for interaction
        plt.close('all')
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        
        # Set background color to light gray for better contrast
        self.fig.patch.set_facecolor('#F5F5F5')
        self.ax.set_facecolor('#F5F5F5')
        
        # Use high-quality image display
        self.ax.imshow(self.current_image, interpolation='antialiased')
        
        # Draw the bounding box for reference with modern style
        bbox = annotation['bbox']
        rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], 
                         fill=False, 
                         edgecolor=self.COLORS['bbox']['color'], 
                         linewidth=2,
                         linestyle='--',
                         alpha=self.COLORS['bbox']['alpha'])
        self.ax.add_patch(rect)
        
        # Add default center point as foreground with modern style
        center_x = int(bbox[0] + bbox[2] / 2)
        center_y = int(bbox[1] + bbox[3] / 2)
        self.input_points.append([center_x, center_y])
        self.input_labels.append(1)
        self._draw_point(center_x, center_y, 1)
        logger.info(f"Added default foreground point at center ({center_x}, {center_y})")
        
        # Add title with instructions and class name - modern styling
        title_text = f'Class: {class_name}'
        self.ax.set_title(title_text, fontsize=14, fontweight='bold', pad=10)
        
        # Add instruction text at the bottom
        instruction_text = (
            "Left-click: Add foreground point  |  "
            "Right-click: Add background point  |  "
            "Middle-click: Delete point  |  "
            "Press Enter: Accept"
        )
        plt.figtext(0.5, 0.01, instruction_text, ha='center', 
                    bbox=dict(boxstyle='round,pad=0.5', 
                             facecolor='white', 
                             edgecolor='#CCCCCC',
                             alpha=0.9))
        
        # Add keyboard shortcut reminder for saving
        save_text = "Ctrl+S: Save progress"
        plt.figtext(0.5, 0.04, save_text, ha='center', fontsize=9,
                   bbox=dict(boxstyle='round,pad=0.3', 
                            facecolor='#E8F4F8', 
                            edgecolor='#CCCCCC',
                            alpha=0.9))
        
        # Add toolbar with category name
        category_box = plt.text(0.05, 0.97, f"Category: {class_name}", 
                              transform=self.ax.transAxes,
                              fontsize=10, fontweight='bold',
                              bbox=dict(boxstyle='round,pad=0.4', 
                                       facecolor=self.COLORS['highlight']['color'],
                                       alpha=0.9))
        
        # Connect event handlers
        self.fig.canvas.mpl_connect('button_press_event', self._onclick)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)
        
        plt.tight_layout()
        # Remove axis ticks for cleaner look
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        
        plt.subplots_adjust(bottom=0.08)  # Adjust to make room for the instruction text
        plt.show(block=True)
        
        return self.input_points, self.input_labels

    def _draw_point(self, x: int, y: int, label: int) -> None:
        """
        Draw a point with modern styling.
        
        Args:
            x: X coordinate
            y: Y coordinate
            label: 1 for foreground, 0 for background
        """
        color = self.COLORS['foreground']['color'] if label == 1 else self.COLORS['background']['color']
        alpha = self.COLORS['foreground']['alpha'] if label == 1 else self.COLORS['background']['alpha']
        
        # Draw point with a pulsing effect
        point = Circle((x, y), radius=self.point_size, 
                       color=color, alpha=alpha, 
                       zorder=100)  # Ensure points are on top
        
        # Add white border
        border = Circle((x, y), radius=self.point_size + self.point_border_size, 
                        facecolor='none', edgecolor='white', linewidth=1.5, 
                        alpha=0.9, zorder=99)
        
        # Add to the plot
        self.ax.add_patch(point)
        self.ax.add_patch(border)
        
        # Add to artists list for hover effects
        self.point_artists.append((len(self.input_points)-1, point, border))
        
        return point, border

    def _onclick(self, event) -> None:
        """
        Handle mouse clicks for point selection.
        
        Args:
            event: Mouse event
        """
        if event.inaxes != self.ax:
            return
            
        x, y = int(event.xdata), int(event.ydata)
        
        # Left button = foreground point (label 1)
        if event.button == 1:
            self.input_points.append([x, y])
            self.input_labels.append(1)
            self._draw_point(x, y, 1)
            logger.info(f"Added foreground point at ({x}, {y})")
            
        # Right button = background point (label 0)
        elif event.button == 3:
            self.input_points.append([x, y])
            self.input_labels.append(0)
            self._draw_point(x, y, 0)
            logger.info(f"Added background point at ({x}, {y})")
            
        # Middle button = delete specific point or clear all points
        elif event.button == 2:
            # Check if click is near an existing point
            if self.input_points:
                points = np.array(self.input_points)
                distances = np.sqrt(np.sum(np.square(points - np.array([x, y])), axis=1))
                min_dist_idx = np.argmin(distances)
                min_dist = distances[min_dist_idx]
                
                # If click is close to a point (within 15 pixels), delete that point
                if min_dist < 15:
                    deleted_point = self.input_points.pop(min_dist_idx)
                    deleted_label = self.input_labels.pop(min_dist_idx)
                    point_type = "foreground" if deleted_label == 1 else "background"
                    logger.info(f"Deleted {point_type} point at ({deleted_point[0]}, {deleted_point[1]})")
                    
                    # Redraw plot with updated points
                    self._redraw_plot()
                    return
            
            # If not near a point, clear all and reset
            self._reset_plot()
        
        self.fig.canvas.draw()

    def _on_mouse_move(self, event) -> None:
        """
        Handle mouse movement for hover effects.
        
        Args:
            event: Mouse event
        """
        if event.inaxes != self.ax or not self.input_points:
            return
            
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return
            
        # Check if mouse is near a point
        points = np.array(self.input_points)
        if len(points) == 0:
            return
            
        distances = np.sqrt(np.sum(np.square(points - np.array([x, y])), axis=1))
        min_dist_idx = np.argmin(distances)
        min_dist = distances[min_dist_idx]
        
        # If hovering near a point, highlight it
        if min_dist < 15:
            if self.hover_point != min_dist_idx:
                # Reset previous hover
                if self.hover_point is not None and self.hover_point < len(self.point_artists):
                    idx, point, border = self.point_artists[self.hover_point]
                    label = self.input_labels[idx]
                    color = self.COLORS['foreground']['color'] if label == 1 else self.COLORS['background']['color']
                    alpha = self.COLORS['foreground']['alpha']
                    point.set_color(color)
                    point.set_alpha(alpha)
                    border.set_linewidth(1.5)
                
                # Highlight new point
                if min_dist_idx < len(self.point_artists):
                    idx, point, border = self.point_artists[min_dist_idx]
                    point.set_color(self.COLORS['highlight']['color'])
                    point.set_alpha(1.0)
                    border.set_linewidth(2.5)
                    self.hover_point = min_dist_idx
                    self.fig.canvas.draw_idle()
        elif self.hover_point is not None:
            # Reset hover when moving away
            if self.hover_point < len(self.point_artists):
                idx, point, border = self.point_artists[self.hover_point]
                label = self.input_labels[idx]
                color = self.COLORS['foreground']['color'] if label == 1 else self.COLORS['background']['color']
                alpha = self.COLORS['foreground']['alpha'] if label == 1 else self.COLORS['background']['alpha']
                point.set_color(color)
                point.set_alpha(alpha)
                border.set_linewidth(1.5)
                self.hover_point = None
                self.fig.canvas.draw_idle()

    def _reset_plot(self) -> None:
        """Clear all points and reset the plot."""
        # Clear the plot and redraw
        self.ax.clear()
        self.ax.imshow(self.current_image)
        
        # Redraw the bounding box
        bbox = self.current_annotation['bbox']
        rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], 
                         fill=False, 
                         edgecolor=self.COLORS['bbox']['color'], 
                         linewidth=2,
                         linestyle='--',
                         alpha=self.COLORS['bbox']['alpha'])
        self.ax.add_patch(rect)
        
        # Clear points
        self.input_points = []
        self.input_labels = []
        self.point_artists = []
        self.hover_point = None
        
        # Add default center point
        center_x = int(bbox[0] + bbox[2] / 2)
        center_y = int(bbox[1] + bbox[3] / 2)
        self.input_points.append([center_x, center_y])
        self.input_labels.append(1)
        self._draw_point(center_x, center_y, 1)
        
        # Update title and styles
        category_id = self.current_annotation['category_id']
        class_name = self.category_map.get(category_id, f"Unknown (ID: {category_id})")
        
        # Add title with class name
        title_text = f'Class: {class_name}'
        self.ax.set_title(title_text, fontsize=14, fontweight='bold', pad=10)
        
        # Remove axis ticks for cleaner look
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        
        logger.info("Reset all points to default center point")
        self.fig.canvas.draw()

    def _redraw_plot(self) -> None:
        """Redraw the plot with current points."""
        # Clear the plot and redraw
        self.ax.clear()
        self.ax.imshow(self.current_image)
        
        # Redraw the bounding box
        bbox = self.current_annotation['bbox']
        rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], 
                         fill=False, 
                         edgecolor=self.COLORS['bbox']['color'], 
                         linewidth=2,
                         linestyle='--',
                         alpha=self.COLORS['bbox']['alpha'])
        self.ax.add_patch(rect)
        
        # Clear point artists
        self.point_artists = []
        
        # Redraw all points
        for i, (point, label) in enumerate(zip(self.input_points, self.input_labels)):
            self._draw_point(point[0], point[1], label)
        
        # Update title
        category_id = self.current_annotation['category_id']
        class_name = self.category_map.get(category_id, f"Unknown (ID: {category_id})")
        
        # Add title with class name
        title_text = f'Class: {class_name}'
        self.ax.set_title(title_text, fontsize=14, fontweight='bold', pad=10)
        
        # Remove axis ticks for cleaner look
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        
        self.fig.canvas.draw()

    def _on_key_press(self, event) -> None:
        """
        Handle key presses for navigation and saving.
        
        Args:
            event: Keyboard event
        """
        if event.key == 'enter':
            plt.close(self.fig)
        elif event.key == 'ctrl+s':
            # Save current progress
            self._save_all_annotations()
            # Show a message on the plot to confirm save
            save_msg = plt.figtext(0.5, 0.5, "All annotations saved!", ha="center", va="center",
                        bbox={"facecolor":"green", "edgecolor":"darkgreen", "alpha":0.9, "pad":10,
                              "boxstyle":"round,pad=0.5"},
                        color="white", fontsize=14, fontweight="bold")
            self.fig.canvas.draw()
            # Flash message for 1 second then remove
            plt.pause(1)
            save_msg.remove()
            self.fig.canvas.draw()
            logger.info("Manual save triggered with Ctrl+S")

    def _save_all_annotations(self) -> None:
        """Save all annotations when manually triggered with Ctrl+S."""
        # Save checkpoint for current position
        if self.current_annotation:
            # Get current image file name from annotation
            image_id = self.current_annotation['image_id']
            image_file_name = next((img['file_name'] for img in self.all_images 
                                 if img['id'] == image_id), "unknown_image.jpg")
            
            # Create a Path object for the image file
            image_file = self.folder_images / "images" / image_file_name
            
            # Save checkpoint
            self._save_checkpoint(image_file)
        
        # # Save modified annotations to output file
        # with open(self.final_json_modify_path, 'w') as f:
        #     json.dump(self.all_annotations_obj, f, indent=2)
        
        # Create a timestamped backup
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        backup_path = self.folder_images / f"Final_Pro_Polygon_backup_{timestamp}.json"
        with open(backup_path, 'w') as f:
            json.dump(self.all_annotations_obj, f, indent=2)
        
        logger.info(f"All annotations saved at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Backup created at: {backup_path}")


if __name__ == "__main__":
    np.random.seed(3)
    logger.info("Starting SAM2PolygonConverter")
    converter = SAM2PolygonConverter()
    converter.process_images()
