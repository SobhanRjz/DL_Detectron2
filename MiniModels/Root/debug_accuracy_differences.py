"""
Debug script to identify accuracy differences between Colab and local environments.

This script analyzes potential causes for accuracy differences including:
- Random seed consistency
- Data loading differences
- Model initialization differences
- Environment-specific configurations
- Hardware differences (GPU vs CPU)
- Library version differences
"""

import os
import sys
import json
import torch
import numpy as np
import random
import platform
import subprocess
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import asdict

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import ModelConfig, DataConfig
from data_manager import DataManager
from models import ModelFactory
from trainer import FewShotTrainer
from evaluator import FewShotEvaluator
from main_few_shot_learning import FewShotLearningPipeline


class EnvironmentDebugger:
    """Debug environment differences that could affect model accuracy."""
    
    def __init__(self, output_dir: str = "debug_results") -> None:
        """Initialize the debugger.
        
        Args:
            output_dir: Directory to save debug results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.debug_info: Dict[str, Any] = {}
        
    def collect_environment_info(self) -> Dict[str, Any]:
        """Collect comprehensive environment information."""
        env_info = {
            "timestamp": datetime.now().isoformat(),
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "python_version": platform.python_version(),
            },
            "hardware": self._get_hardware_info(),
            "libraries": self._get_library_versions(),
            "pytorch": self._get_pytorch_info(),
            "environment_variables": self._get_relevant_env_vars(),
            "cuda": self._get_cuda_info(),
        }
        
        self.debug_info["environment"] = env_info
        return env_info
    
    def _get_hardware_info(self) -> Dict[str, Any]:
        """Get hardware information."""
        hardware_info = {
            "cpu_count": os.cpu_count(),
            "available_memory_gb": self._get_memory_info(),
        }
        
        if torch.cuda.is_available():
            hardware_info.update({
                "gpu_count": torch.cuda.device_count(),
                "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                "gpu_memory": [torch.cuda.get_device_properties(i).total_memory / 1e9 
                              for i in range(torch.cuda.device_count())],
            })
        
        return hardware_info
    
    def _get_memory_info(self) -> float:
        """Get available memory in GB."""
        try:
            import psutil
            return psutil.virtual_memory().total / 1e9
        except ImportError:
            return -1
    
    def _get_library_versions(self) -> Dict[str, str]:
        """Get versions of key libraries."""
        libraries = {}
        
        # Core libraries
        try:
            import torch
            libraries["torch"] = torch.__version__
        except ImportError:
            libraries["torch"] = "Not installed"
        
        try:
            import torchvision
            libraries["torchvision"] = torchvision.__version__
        except ImportError:
            libraries["torchvision"] = "Not installed"
        
        try:
            import numpy
            libraries["numpy"] = numpy.__version__
        except ImportError:
            libraries["numpy"] = "Not installed"
        
        try:
            import optuna
            libraries["optuna"] = optuna.__version__
        except ImportError:
            libraries["optuna"] = "Not installed"
        
        try:
            import easyfsl
            libraries["easyfsl"] = easyfsl.__version__
        except ImportError:
            libraries["easyfsl"] = "Not installed"
        
        return libraries
    
    def _get_pytorch_info(self) -> Dict[str, Any]:
        """Get PyTorch-specific information."""
        pytorch_info = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
            "cudnn_enabled": torch.backends.cudnn.enabled,
            "num_threads": torch.get_num_threads(),
            "num_interop_threads": torch.get_num_interop_threads(),
        }
        
        if torch.cuda.is_available():
            pytorch_info.update({
                "current_device": torch.cuda.current_device(),
                "device_count": torch.cuda.device_count(),
            })
        
        return pytorch_info
    
    def _get_relevant_env_vars(self) -> Dict[str, str]:
        """Get relevant environment variables."""
        relevant_vars = [
            "CUDA_VISIBLE_DEVICES",
            "PYTHONHASHSEED",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "NUMBA_NUM_THREADS",
        ]
        
        return {var: os.environ.get(var, "Not set") for var in relevant_vars}
    
    def _get_cuda_info(self) -> Dict[str, Any]:
        """Get CUDA information."""
        cuda_info = {"available": torch.cuda.is_available()}
        
        if torch.cuda.is_available():
            cuda_info.update({
                "version": torch.version.cuda,
                "device_count": torch.cuda.device_count(),
                "current_device": torch.cuda.current_device(),
                "devices": []
            })
            
            for i in range(torch.cuda.device_count()):
                device_props = torch.cuda.get_device_properties(i)
                cuda_info["devices"].append({
                    "name": device_props.name,
                    "total_memory": device_props.total_memory,
                    "major": device_props.major,
                    "minor": device_props.minor,
                    "multi_processor_count": device_props.multi_processor_count,
                })
        
        return cuda_info


class DataConsistencyChecker:
    """Check data loading consistency between environments."""
    
    def __init__(self, data_config: DataConfig, model_config: ModelConfig) -> None:
        """Initialize data consistency checker."""
        self.data_config = data_config
        self.model_config = model_config
        self.data_manager = DataManager(data_config, model_config)
    
    def check_data_consistency(self) -> Dict[str, Any]:
        """Check data loading consistency."""
        consistency_info = {
            "dataset_info": self._get_dataset_info(),
            "data_loading_test": self._test_data_loading(),
            "random_sampling_test": self._test_random_sampling(),
            "transform_consistency": self._test_transform_consistency(),
        }
        
        return consistency_info
    
    def _get_dataset_info(self) -> Dict[str, Any]:
        """Get basic dataset information."""
        train_set = self.data_manager.train_set
        test_set = self.data_manager.test_set
        
        return {
            "train_size": len(train_set),
            "test_size": len(test_set),
            "train_classes": train_set.classes,
            "test_classes": test_set.classes,
            "train_class_counts": {cls: train_set.targets.count(i) 
                                 for i, cls in enumerate(train_set.classes)},
            "test_class_counts": {cls: test_set.targets.count(i) 
                                for i, cls in enumerate(test_set.classes)},
        }
    
    def _test_data_loading(self) -> Dict[str, Any]:
        """Test data loading consistency."""
        # Test multiple data loader iterations with same seed
        results = []
        
        for seed in [42, 123, 456]:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            
            train_loader = self.data_manager.get_train_loader()
            batch = next(iter(train_loader))
            
            support_images, support_labels, query_images, query_labels, class_ids = batch
            
            # Convert to lists safely
            def safe_to_list(item):
                if hasattr(item, 'tolist'):
                    return item.tolist()
                elif isinstance(item, (list, tuple)):
                    return list(item)
                else:
                    return item
            
            results.append({
                "seed": seed,
                "support_shape": list(support_images.shape),
                "query_shape": list(query_images.shape),
                "support_labels": safe_to_list(support_labels),
                "query_labels": safe_to_list(query_labels),
                "class_ids": safe_to_list(class_ids),
                "support_mean": float(support_images.mean()),
                "support_std": float(support_images.std()),
                "query_mean": float(query_images.mean()),
                "query_std": float(query_images.std()),
            })
        
        return {"loading_tests": results}
    
    def _test_random_sampling(self) -> Dict[str, Any]:
        """Test random sampling consistency."""
        # Test if same seed produces same samples
        seed = 42
        results = []
        
        # Convert to lists safely
        def safe_to_list(item):
            if hasattr(item, 'tolist'):
                return item.tolist()
            elif isinstance(item, (list, tuple)):
                return list(item)
            else:
                return item
        
        for trial in range(3):
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            
            train_loader = self.data_manager.get_train_loader()
            batch = next(iter(train_loader))
            support_images, support_labels, _, _, _ = batch
            
            results.append({
                "trial": trial,
                "first_image_hash": hash(support_images[0].numpy().tobytes()),
                "labels": safe_to_list(support_labels),
            })
        
        # Check if all trials produced same results
        all_same = all(r["first_image_hash"] == results[0]["first_image_hash"] for r in results)
        labels_same = all(r["labels"] == results[0]["labels"] for r in results)
        
        return {
            "sampling_tests": results,
            "consistent_sampling": all_same,
            "consistent_labels": labels_same,
        }
    
    def _test_transform_consistency(self) -> Dict[str, Any]:
        """Test transform consistency."""
        # Get a sample image and apply transforms multiple times
        train_set = self.data_manager.train_set
        sample_image, _ = train_set[0]
        
        # Apply transform multiple times to same image
        results = []
        for i in range(5):
            transformed = train_set.transform(train_set.loader(train_set.samples[0][0]))
            results.append({
                "iteration": i,
                "mean": float(transformed.mean()),
                "std": float(transformed.std()),
                "shape": list(transformed.shape),
            })
        
        return {"transform_tests": results}


class ModelConsistencyChecker:
    """Check model initialization and training consistency."""
    
    def __init__(self, model_config: ModelConfig) -> None:
        """Initialize model consistency checker."""
        self.model_config = model_config
    
    def check_model_consistency(self) -> Dict[str, Any]:
        """Check model consistency across initializations."""
        consistency_info = {
            "initialization_test": self._test_model_initialization(),
            "weight_consistency": self._test_weight_consistency(),
            "forward_pass_test": self._test_forward_pass_consistency(),
        }
        
        return consistency_info
    
    def _test_model_initialization(self) -> Dict[str, Any]:
        """Test model initialization consistency."""
        results = []
        
        for seed in [42, 123, 456]:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            
            model = ModelFactory.create_prototypical_network(self.model_config)
            
            # Get some weight statistics
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            # Get first layer weights for comparison
            first_layer_weights = None
            for param in model.parameters():
                if param.requires_grad:
                    first_layer_weights = param.data.flatten()[:10].tolist()
                    break
            
            results.append({
                "seed": seed,
                "total_params": total_params,
                "trainable_params": trainable_params,
                "first_layer_weights": first_layer_weights,
            })
        
        return {"initialization_tests": results}
    
    def _test_weight_consistency(self) -> Dict[str, Any]:
        """Test weight initialization consistency with same seed."""
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)
        
        model1 = ModelFactory.create_prototypical_network(self.model_config)
        
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)
        
        model2 = ModelFactory.create_prototypical_network(self.model_config)
        
        # Compare weights
        weights_match = True
        for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
            if not torch.equal(param1, param2):
                weights_match = False
                break
        
        return {
            "same_seed_weights_match": weights_match,
            "model1_param_count": sum(p.numel() for p in model1.parameters()),
            "model2_param_count": sum(p.numel() for p in model2.parameters()),
        }
    
    def _test_forward_pass_consistency(self) -> Dict[str, Any]:
        """Test forward pass consistency."""
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)
        
        model = ModelFactory.create_prototypical_network(self.model_config)
        model.eval()
        
        # Create dummy data
        batch_size = self.model_config.N_WAY * self.model_config.N_SHOT
        query_size = self.model_config.N_WAY * self.model_config.N_QUERY
        
        support_images = torch.randn(batch_size, 3, *self.model_config.IMAGE_SIZE)
        support_labels = torch.arange(self.model_config.N_WAY).repeat(self.model_config.N_SHOT)
        query_images = torch.randn(query_size, 3, *self.model_config.IMAGE_SIZE)
        
        # Multiple forward passes with same input
        results = []
        for i in range(3):
            with torch.no_grad():
                output = model(support_images, support_labels, query_images)
                results.append({
                    "pass": i,
                    "output_mean": float(output.mean()),
                    "output_std": float(output.std()),
                    "output_shape": list(output.shape),
                })
        
        # Check consistency
        outputs_consistent = all(
            abs(r["output_mean"] - results[0]["output_mean"]) < 1e-6 
            for r in results
        )
        
        return {
            "forward_pass_tests": results,
            "outputs_consistent": outputs_consistent,
        }


class AccuracyDebugger:
    """Main class to debug accuracy differences."""
    
    def __init__(self, output_dir: str = "debug_results") -> None:
        """Initialize accuracy debugger."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.env_debugger = EnvironmentDebugger(output_dir)
        
        # Initialize configs
        self.model_config = ModelConfig()
        self.data_config = DataConfig()
        
        self.data_checker = DataConsistencyChecker(self.data_config, self.model_config)
        self.model_checker = ModelConsistencyChecker(self.model_config)
        
        self.debug_results: Dict[str, Any] = {}
    
    def run_comprehensive_debug(self) -> Dict[str, Any]:
        """Run comprehensive debugging analysis."""
        print("🔍 Starting comprehensive accuracy debugging...")
        
        # Collect environment information
        print("📊 Collecting environment information...")
        env_info = self.env_debugger.collect_environment_info()
        self.debug_results["environment"] = env_info
        
        # Check data consistency
        print("📁 Checking data consistency...")
        data_info = self.data_checker.check_data_consistency()
        self.debug_results["data_consistency"] = data_info
        
        # Check model consistency
        print("🧠 Checking model consistency...")
        model_info = self.model_checker.check_model_consistency()
        self.debug_results["model_consistency"] = model_info
        
        # Run accuracy comparison tests
        print("🎯 Running accuracy comparison tests...")
        accuracy_info = self._run_accuracy_tests()
        self.debug_results["accuracy_tests"] = accuracy_info
        
        # Analyze potential causes
        print("🔬 Analyzing potential causes...")
        analysis = self._analyze_potential_causes()
        self.debug_results["analysis"] = analysis
        
        # Save results
        self._save_debug_results()
        
        # Generate report
        self._generate_debug_report()
        
        print(f"✅ Debug analysis complete! Results saved to {self.output_dir}")
        return self.debug_results
    
    def _run_accuracy_tests(self) -> Dict[str, Any]:
        """Run multiple accuracy tests with different configurations."""
        accuracy_tests = {}
        
        # Test 1: Multiple runs with same configuration
        print("  🔄 Testing multiple runs with same configuration...")
        accuracy_tests["multiple_runs"] = self._test_multiple_runs()
        
        # Test 2: Different random seeds
        print("  🎲 Testing different random seeds...")
        accuracy_tests["different_seeds"] = self._test_different_seeds()
        
        # Test 3: Different device configurations
        print("  💻 Testing different device configurations...")
        accuracy_tests["device_tests"] = self._test_device_configurations()
        
        return accuracy_tests
    
    def _test_multiple_runs(self, n_runs: int = 3) -> Dict[str, Any]:
        """Test multiple runs with same configuration."""
        results = []
        
        for run in range(n_runs):
            print(f"    Run {run + 1}/{n_runs}")
            
            # Set same seed for each run
            torch.manual_seed(42)
            np.random.seed(42)
            random.seed(42)
            
            try:
                # Create pipeline with reduced episodes for faster testing
                test_config = ModelConfig()
                test_config.N_TRAINING_EPISODES = 50
                test_config.N_EVALUATION_TASKS = 20
                
                pipeline = FewShotLearningPipeline(test_config, self.data_config)
                
                # Train and evaluate
                pipeline.train()
                accuracy = pipeline.evaluate()
                
                results.append({
                    "run": run,
                    "accuracy": accuracy,
                    "success": True,
                })
                
            except Exception as e:
                results.append({
                    "run": run,
                    "accuracy": 0.0,
                    "success": False,
                    "error": str(e),
                })
        
        # Calculate statistics
        successful_runs = [r for r in results if r["success"]]
        accuracies = [r["accuracy"] for r in successful_runs]
        
        stats = {}
        if accuracies:
            stats = {
                "mean_accuracy": np.mean(accuracies),
                "std_accuracy": np.std(accuracies),
                "min_accuracy": np.min(accuracies),
                "max_accuracy": np.max(accuracies),
                "accuracy_range": np.max(accuracies) - np.min(accuracies),
            }
        
        return {
            "results": results,
            "statistics": stats,
            "successful_runs": len(successful_runs),
            "total_runs": n_runs,
        }
    
    def _test_different_seeds(self) -> Dict[str, Any]:
        """Test with different random seeds."""
        seeds = [42, 123, 456, 789, 999]
        results = []
        
        for seed in seeds:
            print(f"    Testing seed {seed}")
            
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            
            try:
                # Create pipeline with reduced episodes
                test_config = ModelConfig()
                test_config.N_TRAINING_EPISODES = 50
                test_config.N_EVALUATION_TASKS = 20
                test_config.RANDOM_SEED = seed
                
                pipeline = FewShotLearningPipeline(test_config, self.data_config)
                
                # Train and evaluate
                pipeline.train()
                accuracy = pipeline.evaluate()
                
                results.append({
                    "seed": seed,
                    "accuracy": accuracy,
                    "success": True,
                })
                
            except Exception as e:
                results.append({
                    "seed": seed,
                    "accuracy": 0.0,
                    "success": False,
                    "error": str(e),
                })
        
        # Calculate statistics
        successful_results = [r for r in results if r["success"]]
        accuracies = [r["accuracy"] for r in successful_results]
        
        stats = {}
        if accuracies:
            stats = {
                "mean_accuracy": np.mean(accuracies),
                "std_accuracy": np.std(accuracies),
                "min_accuracy": np.min(accuracies),
                "max_accuracy": np.max(accuracies),
                "seed_sensitivity": np.std(accuracies),
            }
        
        return {
            "results": results,
            "statistics": stats,
            "tested_seeds": seeds,
        }
    
    def _test_device_configurations(self) -> Dict[str, Any]:
        """Test different device configurations."""
        device_tests = []
        
        # Test CPU
        if True:  # CPU always available
            print("    Testing CPU...")
            try:
                test_config = ModelConfig()
                test_config.DEVICE = "cpu"
                test_config.N_TRAINING_EPISODES = 30
                test_config.N_EVALUATION_TASKS = 10
                
                torch.manual_seed(42)
                np.random.seed(42)
                random.seed(42)
                
                pipeline = FewShotLearningPipeline(test_config, self.data_config)
                pipeline.train()
                accuracy = pipeline.evaluate()
                
                device_tests.append({
                    "device": "cpu",
                    "accuracy": accuracy,
                    "success": True,
                })
                
            except Exception as e:
                device_tests.append({
                    "device": "cpu",
                    "accuracy": 0.0,
                    "success": False,
                    "error": str(e),
                })
        
        # Test CUDA if available
        if torch.cuda.is_available():
            print("    Testing CUDA...")
            try:
                test_config = ModelConfig()
                test_config.DEVICE = "cuda"
                test_config.N_TRAINING_EPISODES = 30
                test_config.N_EVALUATION_TASKS = 10
                
                torch.manual_seed(42)
                np.random.seed(42)
                random.seed(42)
                
                pipeline = FewShotLearningPipeline(test_config, self.data_config)
                pipeline.train()
                accuracy = pipeline.evaluate()
                
                device_tests.append({
                    "device": "cuda",
                    "accuracy": accuracy,
                    "success": True,
                })
                
            except Exception as e:
                device_tests.append({
                    "device": "cuda",
                    "accuracy": 0.0,
                    "success": False,
                    "error": str(e),
                })
        
        return {"device_tests": device_tests}
    
    def _analyze_potential_causes(self) -> Dict[str, Any]:
        """Analyze potential causes of accuracy differences."""
        analysis = {
            "potential_causes": [],
            "recommendations": [],
            "environment_differences": [],
        }
        
        env_info = self.debug_results.get("environment", {})
        
        # Check for common issues
        
        # 1. Random seed issues
        if "accuracy_tests" in self.debug_results:
            seed_tests = self.debug_results["accuracy_tests"].get("different_seeds", {})
            if seed_tests.get("statistics", {}).get("seed_sensitivity", 0) > 5:
                analysis["potential_causes"].append({
                    "cause": "High seed sensitivity",
                    "description": "Model accuracy varies significantly with different random seeds",
                    "severity": "medium",
                })
                analysis["recommendations"].append(
                    "Use multiple seeds and average results for more stable evaluation"
                )
        
        # 2. Device differences
        if "accuracy_tests" in self.debug_results:
            device_tests = self.debug_results["accuracy_tests"].get("device_tests", {}).get("device_tests", [])
            cpu_results = [t for t in device_tests if t["device"] == "cpu" and t["success"]]
            cuda_results = [t for t in device_tests if t["device"] == "cuda" and t["success"]]
            
            if cpu_results and cuda_results:
                cpu_acc = cpu_results[0]["accuracy"]
                cuda_acc = cuda_results[0]["accuracy"]
                if abs(cpu_acc - cuda_acc) > 5:
                    analysis["potential_causes"].append({
                        "cause": "Device-dependent accuracy",
                        "description": f"CPU accuracy: {cpu_acc:.2f}%, CUDA accuracy: {cuda_acc:.2f}%",
                        "severity": "high",
                    })
                    analysis["recommendations"].append(
                        "Ensure consistent device usage between Colab and local environments"
                    )
        
        # 3. Library version differences
        pytorch_info = env_info.get("pytorch", {})
        if pytorch_info.get("cudnn_enabled") != pytorch_info.get("cuda_available"):
            analysis["potential_causes"].append({
                "cause": "CUDNN configuration mismatch",
                "description": "CUDNN availability doesn't match CUDA availability",
                "severity": "medium",
            })
        
        # 4. Threading differences
        if pytorch_info.get("num_threads", 0) != pytorch_info.get("num_interop_threads", 0):
            analysis["potential_causes"].append({
                "cause": "Threading configuration differences",
                "description": "Different thread counts may affect reproducibility",
                "severity": "low",
            })
            analysis["recommendations"].append(
                "Set consistent thread counts: torch.set_num_threads(1)"
            )
        
        # 5. Data loading differences
        data_consistency = self.debug_results.get("data_consistency", {})
        if not data_consistency.get("random_sampling_test", {}).get("consistent_sampling", True):
            analysis["potential_causes"].append({
                "cause": "Inconsistent data sampling",
                "description": "Data sampling is not reproducible with same seed",
                "severity": "high",
            })
            analysis["recommendations"].append(
                "Check DataLoader num_workers setting and ensure proper seed setting"
            )
        
        return analysis
    
    def _save_debug_results(self) -> None:
        """Save debug results to JSON file."""
        results_file = self.output_dir / f"debug_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(results_file, 'w') as f:
            json.dump(self.debug_results, f, indent=2, default=str)
        
        print(f"📄 Debug results saved to {results_file}")
    
    def _generate_debug_report(self) -> None:
        """Generate a human-readable debug report."""
        report_file = self.output_dir / f"debug_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        with open(report_file, 'w') as f:
            f.write("# Accuracy Debugging Report\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Environment summary
            f.write("## Environment Summary\n\n")
            env_info = self.debug_results.get("environment", {})
            
            f.write("### Platform\n")
            platform_info = env_info.get("platform", {})
            f.write(f"- System: {platform_info.get('system', 'Unknown')}\n")
            f.write(f"- Python: {platform_info.get('python_version', 'Unknown')}\n")
            
            f.write("\n### Hardware\n")
            hardware_info = env_info.get("hardware", {})
            f.write(f"- CPU cores: {hardware_info.get('cpu_count', 'Unknown')}\n")
            f.write(f"- Memory: {hardware_info.get('available_memory_gb', 'Unknown'):.1f} GB\n")
            if hardware_info.get("gpu_count", 0) > 0:
                f.write(f"- GPUs: {hardware_info.get('gpu_count', 0)}\n")
                for i, name in enumerate(hardware_info.get("gpu_names", [])):
                    f.write(f"  - GPU {i}: {name}\n")
            
            f.write("\n### Libraries\n")
            libraries = env_info.get("libraries", {})
            for lib, version in libraries.items():
                f.write(f"- {lib}: {version}\n")
            
            # Analysis results
            f.write("\n## Analysis Results\n\n")
            analysis = self.debug_results.get("analysis", {})
            
            potential_causes = analysis.get("potential_causes", [])
            if potential_causes:
                f.write("### Potential Causes\n\n")
                for cause in potential_causes:
                    f.write(f"**{cause['cause']}** (Severity: {cause['severity']})\n")
                    f.write(f"{cause['description']}\n\n")
            
            recommendations = analysis.get("recommendations", [])
            if recommendations:
                f.write("### Recommendations\n\n")
                for i, rec in enumerate(recommendations, 1):
                    f.write(f"{i}. {rec}\n")
            
            # Accuracy test results
            f.write("\n## Accuracy Test Results\n\n")
            accuracy_tests = self.debug_results.get("accuracy_tests", {})
            
            if "multiple_runs" in accuracy_tests:
                multiple_runs = accuracy_tests["multiple_runs"]
                stats = multiple_runs.get("statistics", {})
                if stats:
                    f.write("### Multiple Runs (Same Configuration)\n")
                    f.write(f"- Mean accuracy: {stats.get('mean_accuracy', 0):.2f}%\n")
                    f.write(f"- Standard deviation: {stats.get('std_accuracy', 0):.2f}%\n")
                    f.write(f"- Range: {stats.get('accuracy_range', 0):.2f}%\n")
                    f.write(f"- Successful runs: {multiple_runs.get('successful_runs', 0)}/{multiple_runs.get('total_runs', 0)}\n\n")
            
            if "different_seeds" in accuracy_tests:
                seed_tests = accuracy_tests["different_seeds"]
                stats = seed_tests.get("statistics", {})
                if stats:
                    f.write("### Different Seeds\n")
                    f.write(f"- Mean accuracy: {stats.get('mean_accuracy', 0):.2f}%\n")
                    f.write(f"- Seed sensitivity: {stats.get('seed_sensitivity', 0):.2f}%\n")
                    f.write(f"- Range: {stats.get('max_accuracy', 0) - stats.get('min_accuracy', 0):.2f}%\n\n")
        
        print(f"📋 Debug report saved to {report_file}")


def main() -> None:
    """Main function to run accuracy debugging."""
    print("🚀 Starting Accuracy Debugging Analysis")
    print("=" * 50)
    
    try:
        debugger = AccuracyDebugger()
        results = debugger.run_comprehensive_debug()
        
        print("\n" + "=" * 50)
        print("🎉 Debugging Analysis Complete!")
        
        # Print summary
        analysis = results.get("analysis", {})
        potential_causes = analysis.get("potential_causes", [])
        
        if potential_causes:
            print(f"\n🔍 Found {len(potential_causes)} potential causes:")
            for cause in potential_causes:
                print(f"  • {cause['cause']} ({cause['severity']} severity)")
        
        recommendations = analysis.get("recommendations", [])
        if recommendations:
            print(f"\n💡 Recommendations:")
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec}")
        
        print(f"\n📁 Detailed results saved to: debug_results/")
        
    except Exception as e:
        print(f"❌ Error during debugging: {e}")
        raise


if __name__ == "__main__":
    main()
