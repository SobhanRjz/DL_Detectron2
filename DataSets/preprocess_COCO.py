# preprocess.py
"""
Dataset preprocessing utilities using a class-based design.
"""
import json

class COCOJsonProcessor:
    def __init__(self, list_json):
        """
        Initialize the processor with a list of COCO JSON file paths.

        Args:
            list_json (list): List of paths to JSON files.
        """
        self.list_json = list_json
        self.num_repetitive = 0

    def process_files(self):
        """
        Processes COCO JSON files to ensure annotations have valid segmentation data.
        """
        for pathjson in self.list_json:
            self._process_single_file(pathjson)
        print(f"Number of repetitive corrections: {self.num_repetitive}")

    def _process_single_file(self, pathjson):
        """
        Processes a single COCO JSON file.

        Args:
            pathjson (str): Path to the JSON file.
        """
        with open(pathjson, 'r') as f:
            dataset_dicts = json.load(f)

        for key, value in dataset_dicts.items():
            if key == 'annotations':
                for i in range(len(value)):
                    obj = dataset_dicts[key][i]
                    if 'segmentation' not in obj or obj['segmentation'] == []:
                        self.num_repetitive += 1
                        self._add_segmentation_and_area(obj)

        self._save_file(pathjson, dataset_dicts)

    def _add_segmentation_and_area(self, obj):
        """
        Adds segmentation and area to an annotation object.

        Args:
            obj (dict): Annotation object from COCO JSON.
        """
        bbox = obj['bbox']
        obj['segmentation'] = [
            [bbox[0], bbox[1],
             bbox[0], bbox[1] + bbox[3],
             bbox[0] + bbox[2], bbox[1] + bbox[3],
             bbox[0] + bbox[2], bbox[1],
             bbox[0], bbox[1]]
        ]
        obj['area'] = bbox[2] * bbox[3]

    def _save_file(self, pathjson, dataset_dicts):
        """
        Saves the modified dataset back to the JSON file.

        Args:
            pathjson (str): Path to the JSON file.
            dataset_dicts (dict): Modified dataset dictionary.
        """
        json_object = json.dumps(dataset_dicts, indent=4)
        with open(pathjson, "w") as outfile:
            outfile.write(json_object)
