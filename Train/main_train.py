
import logging
from detectron2.engine import DefaultTrainer
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils import comm
from torch.nn.parallel import DistributedDataParallel
from Train.train_utils import do_train
from Test.test_utils import do_test
from detectron2.modeling import build_model
from Config.basic_config import detectron2_logger as logger


def mainTrain(Basecfg):
    cfg = Basecfg
    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))

    _eval_only = False
    if _eval_only:
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=False
        )
        return do_test(cfg, model)

    distributed = comm.get_world_size() > 1
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
        )

    do_train(cfg, model, resume=False)
    return do_test(cfg, model)