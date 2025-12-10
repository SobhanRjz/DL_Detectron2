"""
FastFlow Anomaly Detection - Training Script
Professional OOP implementation with modular architecture
"""

import argparse
import os
import torch

# Disable proxy to allow direct connection to HuggingFace
for key in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 
            'CURL_CA_BUNDLE', 'REQUESTS_CA_BUNDLE']:
    if key in os.environ:
        del os.environ[key]

import src.config.constants as const
from src.config import ConfigManager
from src.models import FastFlowModel
from src.data import DataLoaderManager
from src.training import FastFlowTrainer


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Train FastFlow on anomaly detection dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model configuration
    parser.add_argument(
        "-cfg", "--config",
        type=str,
        default=r"MiniModels\FastFlow_AnomalyDetection\configs\resnet18.yaml",
        help="Path to model config file"
    )
    
    # Data configuration
    parser.add_argument(
        "--data",
        type=str,
        default="data",
        help="Path to dataset root folder"
    )
    parser.add_argument(
        "-cat", "--category",
        type=str,
        choices=const.MVTEC_CATEGORIES,
        default="pipe_anomaly",
        help="Dataset category name"
    )
    
    # Training configuration
    parser.add_argument(
        "--epochs",
        type=int,
        default=const.NUM_EPOCHS,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=const.BATCH_SIZE,
        help="Batch size for training"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=const.LR,
        help="Learning rate"
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=const.WEIGHT_DECAY,
        help="Weight decay"
    )
    
    # Checkpoint configuration
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=const.CHECKPOINT_DIR,
        help="Directory to save checkpoints"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from"
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=const.EVAL_INTERVAL,
        help="Evaluation interval in epochs"
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=const.CHECKPOINT_INTERVAL,
        help="Checkpoint save interval in epochs"
    )
    
    # Device configuration
    parser.add_argument(
        "--device",
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help="Device to train on"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of data loader workers"
    )
    
    return parser.parse_args()


def main():
    """Main training function"""
    args = parse_args()
    
    # Print header
    print("\n" + "="*70)
    print("FastFlow Anomaly Detection - Training")
    print("="*70)
    
    # Check device availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU")
        args.device = 'cpu'
    
    # Load configuration
    print("\n[1/4] Loading Configuration...")
    config_manager = ConfigManager(args.config)
    print(f"✓ Config loaded from: {args.config}")
    print(f"  - Backbone: {config_manager['backbone_name']}")
    print(f"  - Input Size: {config_manager['input_size']}")
    print(f"  - Flow Steps: {config_manager['flow_step']}")
    
    # Setup data loaders
    print("\n[2/4] Setting up Data Loaders...")
    data_manager = DataLoaderManager(
        data_root=args.data,
        category=args.category,
        input_size=config_manager['input_size'],
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    train_loader = data_manager.get_train_loader()
    test_loader = data_manager.get_test_loader()
    print(f"✓ Train batches: {len(train_loader)}")
    print(f"✓ Test batches: {len(test_loader)}")
    
    # Build model
    print("\n[3/4] Building Model...")
    model = FastFlowModel(config_manager, device=args.device)
    print(f"✓ Model built successfully")
    print(f"  - Device: {args.device}")
    
    # Setup trainer
    print("\n[4/4] Setting up Trainer...")
    trainer = FastFlowTrainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    print("\n✓ Trainer ready")
    
    # Start training
    print("\n" + "="*70)
    print("Starting Training Loop")
    print("="*70)
    
    try:
        trainer.train(
            num_epochs=args.epochs,
            eval_interval=args.eval_interval,
            checkpoint_interval=args.checkpoint_interval
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        print("Saving checkpoint...")
        trainer.save_checkpoint(trainer.current_epoch)
        print("✓ Checkpoint saved")
    
    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)
    print(f"Checkpoints saved in: {trainer.checkpoint_dir}")
    print(f"Best AUROC: {trainer.best_auroc:.4f}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

