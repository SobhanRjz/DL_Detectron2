"""FastFlow Inference Engine - Object-Oriented Design"""

import os
import time
import torch
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional
from pathlib import Path
import matplotlib.pyplot as plt
from torchvision import transforms

from ..models.fastflow import FastFlow
from ..config.config_manager import ConfigManager


class FastFlowInference:
    """FastFlow Anomaly Detection Inference Engine"""
    
    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        device: str = 'cuda',
        threshold: Optional[float] = None
    ):
        """Initialize FastFlow inference engine
        
        Args:
            config_path: Path to model config YAML file
            checkpoint_path: Path to trained model checkpoint
            device: Device to run inference on ('cuda' or 'cpu')
            threshold: Anomaly threshold for classification (optional)
        """
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.threshold = threshold
        
        # Load configuration
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.to_dict()
        
        # Build and load model
        self.model = self._build_model()
        self._load_checkpoint()
        
        # Setup preprocessing
        self.transform = self._get_transform()
        
        self._print_initialization_info()
    
    def _print_initialization_info(self):
        """Print initialization information"""
        print("="*70)
        print("FastFlow Inference Engine Initialized")
        print("="*70)
        print(f"Config: {self.config_path}")
        print(f"Checkpoint: {self.checkpoint_path}")
        print(f"Device: {self.device}")
        print(f"Input Size: {self.config['input_size']}")
        print(f"Backbone: {self.config['backbone_name']}")
        if self.threshold:
            print(f"Threshold: {self.threshold}")
        print("="*70)
    
    def _build_model(self) -> FastFlow:
        """Build FastFlow model"""
        model = FastFlow(
            backbone_name=self.config["backbone_name"],
            flow_steps=self.config["flow_step"],
            input_size=self.config["input_size"],
            conv3x3_only=self.config["conv3x3_only"],
            hidden_ratio=self.config["hidden_ratio"],
            pretrained_backbone_path=self.config.get("pretrained_backbone_path", None),
        )
        model.to(self.device)
        model.eval()
        return model
    
    def _load_checkpoint(self):
        """Load model weights from checkpoint"""
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        print(f"✓ Model loaded from: {self.checkpoint_path}")
    
    def _get_transform(self) -> transforms.Compose:
        """Get image preprocessing transform"""
        return transforms.Compose([
            transforms.Resize((self.config["input_size"], self.config["input_size"])),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    
    def _load_image(self, image_path: str) -> Tuple[torch.Tensor, Image.Image]:
        """Load and preprocess image
        
        Args:
            image_path: Path to image file
            
        Returns:
            Tuple of (preprocessed_tensor, original_image)
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0)
        return image_tensor, image
    
    def _predict_tensor(self, image_tensor: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """Run inference on preprocessed tensor
        
        Args:
            image_tensor: Preprocessed image tensor [B, C, H, W]
            
        Returns:
            Tuple of (anomaly_maps, anomaly_scores)
        """
        image_tensor = image_tensor.to(self.device)
        batch_size = image_tensor.shape[0]
        
        with torch.no_grad():
            ret = self.model(image_tensor)
        
        anomaly_maps = ret["anomaly_map"].cpu().numpy()
        
        if batch_size == 1:
            anomaly_map = anomaly_maps.squeeze()
            anomaly_score = np.max(anomaly_map)
            return anomaly_map, anomaly_score
        else:
            anomaly_maps = anomaly_maps.squeeze(1)
            anomaly_scores = np.array([np.max(am) for am in anomaly_maps])
            return anomaly_maps, anomaly_scores
    
    def predict_single(self, image_path: str, return_map: bool = True) -> dict:
        """Predict anomaly for a single image
        
        Args:
            image_path: Path to image file
            return_map: Whether to return anomaly map
            
        Returns:
            Dictionary containing prediction results
        """
        start_time = time.time()
        
        image_tensor, original_image = self._load_image(image_path)
        anomaly_map, anomaly_score = self._predict_tensor(image_tensor)
        
        inference_time = time.time() - start_time
        
        result = {
            'image_path': image_path,
            'anomaly_score': float(anomaly_score),
            'inference_time': inference_time,
        }
        
        if self.threshold is not None:
            result['is_anomaly'] = anomaly_score > self.threshold
        
        if return_map:
            result['anomaly_map'] = anomaly_map
            result['original_image'] = original_image
        
        return result
    
    def predict_batch(
        self,
        image_paths: List[str],
        batch_size: int = 8,
        return_maps: bool = False,
        verbose: bool = True
    ) -> List[dict]:
        """Predict anomalies for a batch of images
        
        Args:
            image_paths: List of image file paths
            batch_size: Number of images per GPU batch
            return_maps: Whether to return anomaly maps
            verbose: Whether to print progress
            
        Returns:
            List of result dictionaries
        """
        batch_start_time = time.time()
        results = []
        num_images = len(image_paths)
        
        if verbose:
            print(f"\nRunning batched inference on {num_images} images (batch_size={batch_size})")
            print("="*70)
        
        for start in range(0, num_images, batch_size):
            end = min(start + batch_size, num_images)
            batch_paths = image_paths[start:end]
            
            batch_tensors = []
            batch_originals = []
            batch_valid_indices = []
            
            for idx, p in enumerate(batch_paths):
                try:
                    img_tensor, orig = self._load_image(p)
                    batch_tensors.append(img_tensor)
                    batch_originals.append(orig)
                    batch_valid_indices.append(idx)
                except Exception as e:
                    results.append({
                        "image_path": p,
                        "error": str(e),
                        "anomaly_score": None,
                        "inference_time": None,
                    })
                    if verbose:
                        print(f"[{start + idx + 1}/{num_images}] {os.path.basename(p)} - Error: {e}")
            
            if not batch_tensors:
                continue
            
            batch_tensor = torch.cat(batch_tensors, dim=0)
            
            inference_start = time.time()
            anomaly_maps, anomaly_scores = self._predict_tensor(batch_tensor)
            inference_time = time.time() - inference_start
            per_image_time = inference_time / len(batch_tensors)
            
            for i, valid_idx in enumerate(batch_valid_indices):
                p = batch_paths[valid_idx]
                
                if len(batch_tensors) == 1:
                    score = float(anomaly_scores)
                    anom_map = anomaly_maps
                else:
                    score = float(anomaly_scores[i])
                    anom_map = anomaly_maps[i]
                
                result = {
                    "image_path": p,
                    "anomaly_score": score,
                    "inference_time": per_image_time,
                }
                
                if self.threshold is not None:
                    result["is_anomaly"] = score > self.threshold
                
                if return_maps:
                    result["anomaly_map"] = anom_map
                    result["original_image"] = batch_originals[i]
                
                results.append(result)
                
                if verbose:
                    status = "ANOMALY" if result.get("is_anomaly", False) else "NORMAL"
                    print(f"[{len(results)}/{num_images}] {os.path.basename(p)} "
                          f"→ score={score:.6f} - {status} - Time: {per_image_time:.3f}s")
        
        batch_total_time = time.time() - batch_start_time
        
        if verbose and len(results) > 0:
            successful_results = [r for r in results if r.get('inference_time') is not None]
            if successful_results:
                print("="*70)
                print(f"Batch processing complete!")
                print(f"Total time: {batch_total_time:.3f}s")
                print(f"Average time per image: {batch_total_time / len(successful_results):.3f}s")
                print(f"Throughput: {len(successful_results) / batch_total_time:.2f} images/second")
        
        return results
    
    def predict_folder(
        self,
        folder_path: str,
        batch_size: int = 8,
        extensions: List[str] = ['.jpg', '.jpeg', '.png'],
        return_maps: bool = False,
        verbose: bool = True
    ) -> List[dict]:
        """Predict anomalies for all images in a folder
        
        Args:
            folder_path: Path to folder containing images
            batch_size: Number of images per GPU batch
            extensions: List of image file extensions to process
            return_maps: Whether to return anomaly maps
            verbose: Whether to print progress
            
        Returns:
            List of result dictionaries
        """
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Folder not found: {folder_path}")
        
        image_paths = []
        for ext in extensions:
            image_paths.extend(Path(folder_path).glob(f"*{ext}"))
            image_paths.extend(Path(folder_path).glob(f"*{ext.upper()}"))
        
        image_paths = [str(p) for p in image_paths]
        
        if verbose:
            print(f"\nFound {len(image_paths)} images in {folder_path}")
        
        return self.predict_batch(image_paths, batch_size=batch_size, 
                                  return_maps=return_maps, verbose=verbose)
    
    def visualize_result(
        self,
        result: dict,
        save_path: Optional[str] = None,
        show: bool = True
    ):
        """Visualize anomaly detection result
        
        Args:
            result: Result dictionary from predict_single
            save_path: Path to save visualization (optional)
            show: Whether to display the plot
        """
        if 'anomaly_map' not in result:
            print("Error: Anomaly map not available in result. Use return_map=True")
            return
        
        anomaly_map = result['anomaly_map']
        original_image = result['original_image']
        anomaly_score = result['anomaly_score']
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(original_image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        im = axes[1].imshow(anomaly_map, cmap='jet')
        title = f'Anomaly Map\nScore: {anomaly_score:.4f}'
        if self.threshold:
            status = "ANOMALY" if result.get('is_anomaly') else "NORMAL"
            title += f'\nStatus: {status}'
        axes[1].set_title(title)
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1])
        
        axes[2].imshow(original_image)
        axes[2].imshow(anomaly_map, cmap='jet', alpha=0.5)
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Visualization saved to: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def get_statistics(self, results: List[dict]) -> dict:
        """Get statistics from batch prediction results
        
        Args:
            results: List of result dictionaries
            
        Returns:
            Dictionary with statistics
        """
        scores = [r['anomaly_score'] for r in results if r.get('anomaly_score') is not None]
        times = [r['inference_time'] for r in results if r.get('inference_time') is not None]
        
        stats = {
            'total_images': len(results),
            'successful': len(scores),
            'failed': len(results) - len(scores),
            'mean_score': np.mean(scores) if scores else 0,
            'std_score': np.std(scores) if scores else 0,
            'min_score': np.min(scores) if scores else 0,
            'max_score': np.max(scores) if scores else 0,
            'total_inference_time': sum(times) if times else 0,
            'mean_inference_time': np.mean(times) if times else 0,
            'images_per_second': len(times) / sum(times) if times else 0,
        }
        
        if self.threshold is not None:
            anomalies = [r for r in results if r.get('is_anomaly', False)]
            stats['num_anomalies'] = len(anomalies)
            stats['num_normal'] = len(scores) - len(anomalies)
            stats['anomaly_rate'] = len(anomalies) / len(scores) if scores else 0
        
        return stats
    
    def print_statistics(self, results: List[dict]):
        """Print statistics from batch prediction results"""
        stats = self.get_statistics(results)
        
        print("\n" + "="*70)
        print("PREDICTION STATISTICS")
        print("="*70)
        print(f"Total Images:           {stats['total_images']}")
        print(f"Successful:             {stats['successful']}")
        print(f"Failed:                 {stats['failed']}")
        print(f"Mean Score:             {stats['mean_score']:.6f}")
        print(f"Std Score:              {stats['std_score']:.6f}")
        print(f"Min Score:              {stats['min_score']:.6f}")
        print(f"Max Score:              {stats['max_score']:.6f}")
        print(f"Total Inference Time:   {stats['total_inference_time']:.3f}s")
        print(f"Mean Inference Time:    {stats['mean_inference_time']:.3f}s")
        print(f"Images/Second:          {stats['images_per_second']:.2f}")
        
        if 'num_anomalies' in stats:
            print(f"\nThreshold:              {self.threshold}")
            print(f"Anomalies:              {stats['num_anomalies']}")
            print(f"Normal:                 {stats['num_normal']}")
            print(f"Anomaly Rate:           {stats['anomaly_rate']:.2%}")
        
        print("="*70)
    
    def save_results(self, results: List[dict], output_path: str):
        """Save prediction results to file
        
        Args:
            results: List of result dictionaries
            output_path: Path to save results (CSV or JSON)
        """
        import json
        import csv
        
        ext = os.path.splitext(output_path)[1].lower()
        
        if ext == '.json':
            save_data = []
            for r in results:
                save_r = {k: v for k, v in r.items() if k not in ['anomaly_map', 'original_image']}
                save_data.append(save_r)
            
            with open(output_path, 'w') as f:
                json.dump(save_data, f, indent=2)
        
        elif ext == '.csv':
            with open(output_path, 'w', newline='') as f:
                fieldnames = ['image_path', 'anomaly_score', 'inference_time']
                if self.threshold is not None:
                    fieldnames.append('is_anomaly')
                if 'error' in results[0]:
                    fieldnames.append('error')
                
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for r in results:
                    row = {k: r.get(k) for k in fieldnames}
                    writer.writerow(row)
        else:
            raise ValueError(f"Unsupported file format: {ext}. Use .json or .csv")
        
        print(f"✓ Results saved to: {output_path}")

