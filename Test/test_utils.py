from collections import OrderedDict
import os
from detectron2.data import build_detection_test_loader
from detectron2.evaluation import inference_on_dataset
from detectron2.utils import comm
from detectron2.evaluation import (
    inference_on_dataset,
    print_csv_format,
)
from Config.basic_config import detectron2_logger as logger
from Utils.evaluation_utils import get_evaluator

def do_test(cfg, model):
    results = OrderedDict()
    for dataset_name in cfg.DATASETS.TEST:
        data_loader = build_detection_test_loader(cfg, dataset_name)
        evaluator = get_evaluator(
            cfg, dataset_name, os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        )
        results_i = inference_on_dataset(model, data_loader, evaluator)
        results[dataset_name] = results_i
        if comm.is_main_process():
            logger.info("Evaluation results for {} in csv format:".format(dataset_name))
            print_csv_format(results_i)
    if len(results) == 1:
        results = list(results.values())[0]
    return results