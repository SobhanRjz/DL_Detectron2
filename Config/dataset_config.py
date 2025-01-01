# dataset_config.py
"""
Dataset configuration module for data loading and preprocessing.
"""
import os
from Config.basic_config import DATA_PATH
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog
class DatasetConfig:
    def __init__(self):
        # Define dataset types
        self.dataset_types = ['train', 'valid', 'test']
        self.json_paths = {}
        self.images_paths = {}
        #self._cleanup_catalogs()
        
        # Use a loop to set paths
        for dataset in self.dataset_types:
            self.json_paths[dataset] = os.path.join(DATA_PATH, dataset, '_annotations.coco.json')
            self.images_paths[dataset] = os.path.join(DATA_PATH, dataset)

    def register_datasets(self):
        """Register COCO datasets for training, validation, and testing."""
        for dataset in self.dataset_types:
            register_coco_instances(f"my_dataset_{dataset}", {}, self.json_paths[dataset], self.images_paths[dataset])
        print(MetadataCatalog.list())
        print(DatasetCatalog.list())


    def _cleanup_catalogs(self):
        """Remove all existing datasets from MetadataCatalog and DatasetCatalog."""
        # Clean MetadataCatalog
        for dataset in MetadataCatalog.list():
            MetadataCatalog.pop(dataset)
        print("Cleaned MetadataCatalog.")

        # Clean DatasetCatalog
        for dataset in DatasetCatalog.list():
            DatasetCatalog.remove(dataset)
        print("Cleaned DatasetCatalog.")
