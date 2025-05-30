"""Simple runner for model speed testing."""

import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from speed_test import ModelSpeedTester


def main():
    """Run speed test with different configurations."""
    print("Starting Few-Shot Model Speed Test...")
    print()
    
    # Test with default model path
    model_path = "outputs/best_model/model.pth"
    
    try:
        # Initialize and run speed test
        tester = ModelSpeedTester(model_path=model_path)
        tester.run_speed_test(num_iterations=50)  # Reduced for quick testing
        
    except FileNotFoundError as e:
        print(f"Error: Model file not found - {e}")
        print("Please ensure the model file exists at the specified path.")
        
    except Exception as e:
        print(f"Error during speed testing: {e}")
        print("Please check your model and dependencies.")


if __name__ == "__main__":
    main() 