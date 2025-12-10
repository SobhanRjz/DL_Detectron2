"""Training module for FastFlow"""

import os
import torch
import torch.optim as optim
from tqdm import tqdm
import config.constants as const
from ..utils.helpers import AverageMeter
from ..utils.metrics import ROC_AUC


class FastFlowTrainer:
    """FastFlow model trainer with full training pipeline"""
    
    def __init__(
        self,
        model,
        train_loader,
        test_loader,
        checkpoint_dir=None,
        device='cuda',
        lr=None,
        weight_decay=None,
    ):
        """Initialize trainer
        
        Args:
            model: FastFlowModel instance
            train_loader: Training data loader
            test_loader: Testing data loader
            checkpoint_dir: Directory to save checkpoints
            device: Device to train on
            lr: Learning rate (default: from constants)
            weight_decay: Weight decay (default: from constants)
        """
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        
        # Setup checkpoint directory
        if checkpoint_dir is None:
            checkpoint_dir = const.CHECKPOINT_DIR
        os.makedirs(checkpoint_dir, exist_ok=True)
        exp_id = len(os.listdir(checkpoint_dir))
        self.checkpoint_dir = os.path.join(checkpoint_dir, f"exp{exp_id}")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        print(f"Checkpoint directory: {self.checkpoint_dir}")
        
        # Setup optimizer
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=lr or const.LR,
            weight_decay=weight_decay or const.WEIGHT_DECAY
        )
        
        # Training state
        self.current_epoch = 0
        self.best_auroc = 0.0
        
    def train_one_epoch(self, epoch):
        """Train for one epoch

        Args:
            epoch: Current epoch number
        """
        self.model.train()
        loss_meter = AverageMeter()

        # Create progress bar with epoch information
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch + 1}",
            unit="batch",
            leave=True
        )

        for step, (data, _) in enumerate(pbar):
            # Forward
            data = data.to(self.device)
            ret = self.model(data)
            loss = ret["loss"]

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update metrics
            loss_meter.update(loss.item())

            # Update progress bar with current loss info
            pbar.set_postfix({
                'loss': f"{loss_meter.val:.3f}",
                'avg_loss': f"{loss_meter.avg:.3f}"
            })

            # Optional: print detailed logs at intervals (less frequent with tqdm)
            if (step + 1) % (const.LOG_INTERVAL * 5) == 0 or (step + 1) == len(self.train_loader):
                tqdm.write(
                    f"Epoch {epoch + 1} - Step {step + 1}/{len(self.train_loader)}: "
                    f"loss = {loss_meter.val:.3f}({loss_meter.avg:.3f})"
                )

        # Close progress bar
        pbar.close()
    
    def evaluate(self):
        """Evaluate model on test set

        Returns:
            float: AUROC score
        """
        self.model.eval()
        auroc_metric = ROC_AUC()

        # Create progress bar for evaluation
        pbar = tqdm(
            self.test_loader,
            desc="Evaluating",
            unit="batch",
            leave=False
        )

        for data, targets in pbar:
            data, targets = data.to(self.device), targets.to(self.device)

            with torch.no_grad():
                ret = self.model(data)

            # Get image-level anomaly scores
            anomaly_maps = ret["anomaly_map"].cpu().detach()
            outputs = torch.max(anomaly_maps.view(anomaly_maps.size(0), -1), dim=1)[0]
            targets = targets.cpu().detach()
            auroc_metric.update((outputs, targets))

        # Close progress bar
        pbar.close()

        auroc = auroc_metric.compute()
        print(f"AUROC: {auroc:.4f}")
        return auroc
    
    def save_checkpoint(self, epoch, auroc=None):
        """Save model checkpoint
        
        Args:
            epoch: Current epoch
            auroc: Current AUROC score (optional)
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{epoch}.pt")
        self.model.save_checkpoint(checkpoint_path, epoch, self.optimizer)
        
        # Save best model
        if auroc and auroc > self.best_auroc:
            self.best_auroc = auroc
            best_path = os.path.join(self.checkpoint_dir, "best.pt")
            self.model.save_checkpoint(best_path, epoch, self.optimizer)
            print(f"✓ New best model! AUROC: {auroc:.4f}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint to resume training
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_checkpoint(checkpoint_path)
        
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if "epoch" in checkpoint:
            self.current_epoch = checkpoint["epoch"] + 1
            print(f"Resumed from epoch {self.current_epoch}")
    
    def train(self, num_epochs=None, eval_interval=None, checkpoint_interval=None):
        """Full training loop
        
        Args:
            num_epochs: Number of epochs (default: from constants)
            eval_interval: Evaluation interval (default: from constants)
            checkpoint_interval: Checkpoint save interval (default: from constants)
        """
        num_epochs = num_epochs or const.NUM_EPOCHS
        eval_interval = eval_interval or const.EVAL_INTERVAL
        checkpoint_interval = checkpoint_interval or const.CHECKPOINT_INTERVAL
        
        print("="*70)
        print("Starting Training")
        print("="*70)
        print(f"Epochs: {num_epochs}")
        print(f"Eval Interval: {eval_interval}")
        print(f"Checkpoint Interval: {checkpoint_interval}")
        print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']}")
        print(f"Device: {self.device}")
        print("="*70)
        
        for epoch in range(self.current_epoch, num_epochs):
            print(f"\n{'='*70}")
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"{'='*70}")
            
            # Train
            self.train_one_epoch(epoch)
            
            # Evaluate
            if (epoch + 1) % eval_interval == 0:
                auroc = self.evaluate()
                
                # Save checkpoint with AUROC
                if (epoch + 1) % checkpoint_interval == 0:
                    self.save_checkpoint(epoch, auroc)
            else:
                # Save checkpoint without evaluation
                if (epoch + 1) % checkpoint_interval == 0:
                    self.save_checkpoint(epoch)
        
        print("\n" + "="*70)
        print("Training Complete!")
        print(f"Best AUROC: {self.best_auroc:.4f}")
        print(f"Checkpoints saved in: {self.checkpoint_dir}")
        print("="*70)

