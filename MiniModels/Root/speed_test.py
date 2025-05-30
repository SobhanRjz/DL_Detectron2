"""Simple speed test for few-shot learning model."""

import time
import torch
import numpy as np
from pathlib import Path
from typing import Tuple, List
import statistics

from config import ModelConfig
from models import ModelFactory


class ModelSpeedTester:
    """Test the inference speed of the few-shot learning model."""
    
    def __init__(self, model_path: str = "outputs/best_model/model.pth") -> None:
        """Initialize the speed tester.
        
        Args:
            model_path: Path to the saved model
        """
        self.model_path = Path(model_path)
        self.config = ModelConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Using device: {self.device}")
        
        # Load the model
        self.model = self._load_model()
        self.model.eval()
        
        print(f"Model loaded from: {self.model_path}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def _load_model(self) -> torch.nn.Module:
        """Load the saved model."""
        # Create model architecture
        model = ModelFactory.create_prototypical_network(self.config)
        
        # Load saved weights
        if self.model_path.exists():
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                else:
                    model.load_state_dict(checkpoint)
            else:
                model.load_state_dict(checkpoint)
            
            print("Model weights loaded successfully!")
        else:
            print(f"Warning: Model file not found at {self.model_path}")
            print("Using randomly initialized model for speed testing")
        
        return model.to(self.device)
    
    def _create_dummy_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create dummy data for speed testing.
        
        Returns:
            Tuple of (support_images, support_labels, query_images)
        """
        # Create dummy support set (N_WAY * N_SHOT images)
        n_support = self.config.N_WAY * self.config.N_SHOT
        support_images = torch.randn(
            n_support, 3, *self.config.IMAGE_SIZE, 
            device=self.device
        )
        
        # Create support labels
        support_labels = torch.arange(self.config.N_WAY, device=self.device).repeat(
            self.config.N_SHOT
        )
        
        # Create dummy query set (N_WAY * N_QUERY images)
        n_query = self.config.N_WAY * self.config.N_QUERY
        query_images = torch.randn(
            n_query, 3, *self.config.IMAGE_SIZE,
            device=self.device
        )
        
        return support_images, support_labels, query_images
    
    def warm_up(self, num_warmup: int = 10) -> None:
        """Warm up the model and GPU."""
        print(f"Warming up model with {num_warmup} iterations...")
        
        support_images, support_labels, query_images = self._create_dummy_data()
        
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = self.model(support_images, support_labels, query_images)
        
        # Synchronize GPU
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        print("Warm-up completed!")
    
    def measure_inference_time(self, num_iterations: int = 100) -> Tuple[float, float, float]:
        """Measure inference time for multiple iterations.
        
        Args:
            num_iterations: Number of inference iterations
            
        Returns:
            Tuple of (mean_time, std_time, fps)
        """
        print(f"Running {num_iterations} inference iterations...")
        
        support_images, support_labels, query_images = self._create_dummy_data()
        inference_times = []
        
        with torch.no_grad():
            for i in range(num_iterations):
                # Start timing
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                start_time = time.perf_counter()
                
                # Forward pass
                _ = self.model(support_images, support_labels, query_images)
                
                # End timing
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                end_time = time.perf_counter()
                
                inference_time = end_time - start_time
                inference_times.append(inference_time)
                
                if (i + 1) % 20 == 0:
                    print(f"Completed {i + 1}/{num_iterations} iterations")
        
        # Calculate statistics
        mean_time = statistics.mean(inference_times)
        std_time = statistics.stdev(inference_times) if len(inference_times) > 1 else 0.0
        
        # Calculate FPS (frames per second)
        # For few-shot learning, we consider the query set as "frames"
        n_query = self.config.N_WAY * self.config.N_QUERY
        fps = n_query / mean_time
        
        return mean_time, std_time, fps
    
    def run_speed_test(self, num_iterations: int = 100) -> None:
        """Run complete speed test."""
        print("=" * 60)
        print("FEW-SHOT MODEL SPEED TEST")
        print("=" * 60)
        
        print(f"Model Configuration:")
        print(f"  - N-Way: {self.config.N_WAY}")
        print(f"  - N-Shot: {self.config.N_SHOT}")
        print(f"  - N-Query: {self.config.N_QUERY}")
        print(f"  - Image Size: {self.config.IMAGE_SIZE}")
        print(f"  - Backbone: {self.config.BACKBONE_NAME}")
        print(f"  - Device: {self.device}")
        print()
        
        # Warm up
        self.warm_up()
        print()
        
        # Measure inference time
        mean_time, std_time, fps = self.measure_inference_time(num_iterations)
        
        # Display results
        print("=" * 60)
        print("SPEED TEST RESULTS")
        print("=" * 60)
        print(f"Mean inference time: {mean_time*1000:.2f} ± {std_time*1000:.2f} ms")
        print(f"Throughput (FPS): {fps:.2f} queries/second")
        print(f"Time per query image: {(mean_time/self.config.N_QUERY)*1000:.2f} ms")
        print()
        
        # Additional metrics
        min_time = min([mean_time - std_time, mean_time]) * 1000
        max_time = (mean_time + std_time) * 1000
        print(f"Time range: {min_time:.2f} - {max_time:.2f} ms")
        print(f"Total support images processed: {self.config.N_WAY * self.config.N_SHOT}")
        print(f"Total query images processed: {self.config.N_WAY * self.config.N_QUERY}")
        print("=" * 60)


def main():
    """Main function to run speed test."""
    # Initialize speed tester
    tester = ModelSpeedTester()
    
    # Run speed test
    tester.run_speed_test(num_iterations=100)


if __name__ == "__main__":
    main() 