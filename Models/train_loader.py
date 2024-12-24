"""
Custom trainers and data mappers for Detectron2.
"""

import os
import copy
import torch
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
import detectron2.data.transforms as T
from detectron2.data import DatasetMapper
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.data import detection_utils as utils


class CustomTrainer(DefaultTrainer):
    """
    Custom trainer with a custom data mapper.
    """

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=cls._custom_mapper)
    
    @staticmethod
    def _custom_mapper(dataset_dict):
        """
        Custom dataset mapper for applying data augmentations during training.

        Args:
            dataset_dict (dict): Dataset dictionary for a single image.

        Returns:
            dict: Transformed dataset dictionary.
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # It will be modified below
        image = utils.read_image(dataset_dict["file_name"], format="BGR")
        transform_list = [
            T.RandomBrightness(0.8, 1.8),
            # T.Resize((800,600)),
            # T.RandomContrast(0.6, 1.3),
            #T.RandomSaturation(0.8, 1.4),
            # T.RandomRotation(angle=[90, 90]),
            # T.RandomLighting(0.7),
            # T.RandomFlip(prob=0.4, horizontal=False, vertical=True),
        ]
        image, transforms = T.apply_transform_gens(transform_list, image)
        dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))

        annos = [
            utils.transform_instance_annotations(obj, transforms, image.shape[:2])
            for obj in dataset_dict.pop("annotations")
            if obj.get("iscrowd", 0) == 0
        ]
        instances = utils.annotations_to_instances(annos, image.shape[:2])
        dataset_dict["instances"] = utils.filter_empty_instances(instances)
        return dataset_dict


class CocoTrainer(DefaultTrainer):
    """
    Custom trainer with COCO evaluation support.
    """
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            os.makedirs("coco_eval", exist_ok=True)
            output_folder = "coco_eval"
        return COCOEvaluator(dataset_name, cfg, False, output_folder)
