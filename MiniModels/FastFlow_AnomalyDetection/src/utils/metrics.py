"""Metrics for anomaly detection evaluation"""

import numpy as np
from sklearn.metrics import roc_auc_score


class ROC_AUC:
    """Custom ROC_AUC metric for anomaly detection evaluation"""
    
    def __init__(self):
        """Initialize metric"""
        self.reset()

    def reset(self):
        """Reset accumulated predictions and targets"""
        self.all_outputs = []
        self.all_targets = []

    def update(self, outputs_targets):
        """Update metric with new batch
        
        Args:
            outputs_targets: Tuple of (outputs, targets) tensors
        """
        outputs, targets = outputs_targets
        # Move tensors to CPU before converting to numpy
        self.all_outputs.append(outputs.cpu().numpy())
        self.all_targets.append(targets.cpu().numpy())

    def compute(self):
        """Compute final ROC-AUC score
        
        Returns:
            float: ROC-AUC score
        """
        all_outputs = np.concatenate(self.all_outputs)
        all_targets = np.concatenate(self.all_targets)
        return roc_auc_score(all_targets, all_outputs)

