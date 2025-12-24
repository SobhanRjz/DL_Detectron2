#!/usr/bin/env python3
"""Check dataset structure and sample counts for few-shot learning."""

import os
from pathlib import Path
from collections import defaultdict

def check_dataset_structure(dataset_path: str):
    """Check if dataset has sufficient samples for few-shot learning."""
    print(f"Checking dataset: {dataset_path}")

    if not Path(dataset_path).exists():
        print(f"ERROR: Dataset path does not exist: {dataset_path}")
        return False

    train_path = Path(dataset_path) / "Train"
    test_path = Path(dataset_path) / "Test"

    if not train_path.exists():
        print(f"ERROR: Train directory not found: {train_path}")
        return False

    if not test_path.exists():
        print(f"ERROR: Test directory not found: {test_path}")
        return False

    # Check train classes
    train_classes = [d for d in train_path.iterdir() if d.is_dir()]
    print(f"\nTrain classes found: {len(train_classes)}")
    train_counts = {}

    for class_dir in sorted(train_classes):
        files = list(class_dir.glob("*.jpg"))
        train_counts[class_dir.name] = len(files)
        print(f"  {class_dir.name}: {len(files)} samples")

    # Check test classes
    test_classes = [d for d in test_path.iterdir() if d.is_dir()]
    print(f"\nTest classes found: {len(test_classes)}")
    test_counts = {}

    for class_dir in sorted(test_classes):
        files = list(class_dir.glob("*.jpg"))
        test_counts[class_dir.name] = len(files)
        print(f"  {class_dir.name}: {len(files)} samples")

    # Check requirements for few-shot learning (n_shot=1, n_query=1)
    min_samples_required = 1 + 1  # n_shot + n_query
    print(f"\nFew-shot requirements: n_shot=1, n_query=1 -> {min_samples_required} samples per class minimum")

    issues = []

    # Check if all classes are present in both train and test
    train_class_names = set(train_counts.keys())
    test_class_names = set(test_counts.keys())

    if train_class_names != test_class_names:
        issues.append(f"Class mismatch: Train has {train_class_names}, Test has {test_class_names}")

    # Check minimum samples per class
    for class_name, count in test_counts.items():
        if count < min_samples_required:
            issues.append(f"Test class '{class_name}' has only {count} samples (needs {min_samples_required}+)")

    for class_name, count in train_counts.items():
        if count < min_samples_required:
            issues.append(f"Train class '{class_name}' has only {count} samples (needs {min_samples_required}+)")

    if issues:
        print("\nISSUES FOUND:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("\nDataset structure looks good for few-shot learning!")
        return True

if __name__ == "__main__":
    base_path = r"C:\Users\sobha\Desktop\detectron2\Code\Implement Detectron 2\MiniModels\FewShotLearning\datasets"

    # Check the Cr_Fr_Br_Xr_Hol dataset
    dataset_name = "Cr_Fr_Br_Xr_Hol"
    dataset_path = os.path.join(base_path, dataset_name)

    check_dataset_structure(dataset_path)
