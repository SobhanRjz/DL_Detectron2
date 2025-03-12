import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class DatasetPaths:
    BasePath: Path = Path("C:/Users/sobha/Desktop/detectron2/Data/RoboFlowData/5.pipeline.v1i.coco-segmentation")
    validation: Path = BasePath / "valid"
    train: Path = BasePath / "train" 
    test: Path = BasePath / "test"
    output: Path = BasePath / "combined"

    @property
    def output_images(self) -> Path:
        return self.output / "images"
    
    @property
    def output_annotations(self) -> Path:
        return self.output / "I_Basic_annotations.coco.json"

class COCODatasetMerger:
    def __init__(self, paths: DatasetPaths):
        self.paths = paths
        self.combined_annotations = {
            "images": [],
            "annotations": [],
            "categories": []
        }
        self.image_id_offset = 0
        self.annotation_id_offset = 0
        
        # Create output directories
        self.paths.output_images.mkdir(parents=True, exist_ok=True)

    def merge_folder(self, folder_path: Path, json_file: Path) -> Tuple[int, int]:
        """
        Merge images and annotations from a folder into the combined structure.
        
        Args:
            folder_path: Path to the dataset folder
            json_file: Path to the COCO JSON annotation file
            
        Returns:
            Tuple containing updated image and annotation ID offsets
        """
        with json_file.open('r') as f:
            data = json.load(f)

        # Copy categories only once
        if not self.combined_annotations["categories"]:
            self.combined_annotations["categories"] = data["categories"]

        # Process images
        for image in data["images"]:
            old_image_id = image["id"]
            new_image_id = self.image_id_offset + old_image_id
            image["id"] = new_image_id
            self.combined_annotations["images"].append(image)

            # Copy image file
            src = folder_path / image["file_name"]
            dst = self.paths.output_images / image["file_name"]
            shutil.copy(str(src), str(dst))

        # Process annotations
        for annotation in data["annotations"]:
            annotation["id"] += self.annotation_id_offset
            annotation["image_id"] += self.image_id_offset
            self.combined_annotations["annotations"].append(annotation)


        return (
            self.image_id_offset + len(data["images"]),
            self.annotation_id_offset + len(data["annotations"])
        )

    def merge_all(self):
        """Merge all dataset folders and save combined annotations."""
        folders = [
            (self.paths.validation, self.paths.validation / "_annotations.coco.json"),
            (self.paths.train, self.paths.train / "_annotations.coco.json"),
            (self.paths.test, self.paths.test / "_annotations.coco.json")
        ]

        for folder_path, json_file in folders:
            if folder_path.exists() and json_file.exists():
                self.image_id_offset, self.annotation_id_offset = self.merge_folder(folder_path, json_file)

        # Save combined annotations
        with self.paths.output_annotations.open('w') as f:
            json.dump(self.combined_annotations, f, indent=4)

def main():
    paths = DatasetPaths()
    merger = COCODatasetMerger(paths)
    merger.merge_all()
    
    print(f"Merged annotations saved to {paths.output_annotations}")
    print(f"All images saved to {paths.output_images}")

if __name__ == "__main__":
    main()
