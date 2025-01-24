import torch
from .base_trainer import BaseTrainer
from Config.basic_config import detectron2_logger as logger

class SimpleDefaultTrainer(BaseTrainer):
    """Default trainer implementation"""

    def __init__(self, cfg,  resumeTrain=False):
        self.resumeTrain = resumeTrain
        super().__init__(cfg)
        
    
    def do_train(self):
        """
        Main training logic implemented inside the trainer class.
        Args:
            resume (bool): whether to attempt to resume from the checkpoint directory.
                         Default: False.
        """
        import os
        import glob
        import shutil
        
        # Remove TensorBoard event files from output directory
        if os.path.exists(self.cfg.OUTPUT_DIR):
            event_files = glob.glob(os.path.join(self.cfg.OUTPUT_DIR, "events.out.tfevents.*"))
            for f in event_files:
                try:
                    os.remove(f)
                    logger.info(f"Removed TensorBoard event file: {f}")
                except OSError as e:
                    logger.warning(f"Error removing file {f}: {e}")
                    
        from detectron2.data import build_detection_train_loader
        from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
        from detectron2.utils.events import EventStorage
        from detectron2.utils import comm
        from detectron2.solver import build_lr_scheduler, build_optimizer
        from detectron2.engine import default_writers
        from torch.utils.tensorboard import SummaryWriter
        
        model = self.model
        cfg = self.cfg
        
        # Build optimizer and scheduler
        optimizer = build_optimizer(cfg, model)
        scheduler = build_lr_scheduler(cfg, optimizer)

        # Create checkpointer
        checkpointer = DetectionCheckpointer(
            model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
        )
        
        # Resume or load model weights
        start_iter = checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=self.resumeTrain).get("iteration", -1) + 1
        max_iter = cfg.SOLVER.MAX_ITER
        
        # Set up periodic checkpointing
        periodic_checkpointer = PeriodicCheckpointer(
            checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
        )

        # Set up writers for logging
        writers = default_writers(cfg.OUTPUT_DIR, max_iter) if comm.is_main_process() else []
        
        # Initialize TensorBoard writer
        tb_writer = SummaryWriter(log_dir=cfg.OUTPUT_DIR) if comm.is_main_process() else None

        # Build data loader
        data_loader = build_detection_train_loader(cfg) # by default use cfg.INPUT.RANDOM_FLIP and cfg.INPUT.MAX_SIZE_TRAIN and cfg.INPUT.CROP.ENABLED and cfg.INPUT.MIN_SIZE_TRAIN
        logger.info("Starting training from iteration {}".format(start_iter))

        # Main training loop
        with EventStorage(start_iter) as storage:
            for data, iteration in zip(data_loader, range(start_iter, max_iter)):
                storage.iter = iteration

                loss_dict = model(data)
                losses = sum(loss_dict.values())
                assert torch.isfinite(losses).all(), loss_dict

                loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
                losses_reduced = sum(loss for loss in loss_dict_reduced.values())
                if comm.is_main_process():
                    storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)
                    
                    # Log metrics to TensorBoard
                    if tb_writer is not None:
                        tb_writer.add_scalar('total_loss', losses_reduced, iteration)
                        for k, v in loss_dict_reduced.items():
                            tb_writer.add_scalar(f'losses/{k}', v, iteration)
                        tb_writer.add_scalar('lr', optimizer.param_groups[0]["lr"], iteration)

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
                storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
                scheduler.step()

                if (
                    cfg.TEST.EVAL_PERIOD > 0
                    and (iteration + 1) % cfg.TEST.EVAL_PERIOD == 0
                    and iteration != max_iter - 1
                ):
                    test_results = self.do_test()
                    # Log evaluation metrics to TensorBoard
                    if comm.is_main_process() and tb_writer is not None:
                        for metric_name, metric_value in test_results.items():
                            if isinstance(metric_value, (int, float)):
                                tb_writer.add_scalar(f'eval/{metric_name}', metric_value, iteration)
                    comm.synchronize()

                if iteration - start_iter > 5 and (
                    (iteration + 1) % 20 == 0 or iteration == max_iter - 1
                ):
                    for writer in writers:
                        writer.write()
                periodic_checkpointer.step(iteration)
            

            self.do_test(dataset_name="my_dataset_test")
            # Close TensorBoard writer
            if tb_writer is not None:
                tb_writer.close()

    def do_test(self, model=None, dataset_name="my_dataset_valid"):
        """
        Run model evaluation on test datasets.
        
        Args:
            model: Model to evaluate. If None, uses self.model
            dataset_name: Name of dataset to evaluate on. Defaults to validation set.
            
        Returns:
            OrderedDict of evaluation results for each dataset
        """
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

        if model is None:
            model = self.model

        results = OrderedDict()
        try:
            # Create data loader for test dataset
            data_loader = build_detection_test_loader(self.cfg, dataset_name)
            
            # Initialize evaluator
            output_folder = os.path.join(self.cfg.OUTPUT_DIR, "inference", dataset_name)
            evaluator = get_evaluator(
                self.cfg, dataset_name, output_folder
            )
            
            # Run inference
            results_i = inference_on_dataset(model, data_loader, evaluator)
            results[dataset_name] = results_i
            
            # Log results
            if comm.is_main_process():
                logger.info(f"Evaluation results for {dataset_name} in csv format:")
                print_csv_format(results_i)
                
        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}")
            raise
            
        # Return single result if only one dataset
        if len(results) == 1:
            results = list(results.values())[0]
            
        return results