from .base_trainer import BaseTrainer
# Add your custom training logic imports here

class CustomTrainer(BaseTrainer):

    def __init__(self, cfg):
        super().__init__(cfg)


    def do_train(self, resume=False):
        import torch
        from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
        from detectron2.data import build_detection_train_loader
        from detectron2.engine import default_writers
        from detectron2.solver import build_lr_scheduler, build_optimizer
        from detectron2.solver import LRMultiplier, WarmupParamScheduler
        from fvcore.common.param_scheduler import MultiStepParamScheduler
        from detectron2.utils import comm
        from detectron2.utils.events import EventStorage
        from Config.basic_config import detectron2_logger as logger
        from torch.utils.tensorboard import SummaryWriter
        from detectron2.data.samplers import RepeatFactorTrainingSampler
        from detectron2.data import DatasetCatalog, MetadataCatalog

        # Build optimizer and scheduler
        optimizer = build_optimizer(self.cfg, self.model)
        max_iter = self.cfg.SOLVER.MAX_ITER
        
        # Configure learning rate scheduler
        if hasattr(self.cfg.SOLVER, 'LR_TEST') and self.cfg.SOLVER.LR_TEST:
            try:
                warmup_factor = self.cfg.SOLVER.BASE_LR / self.cfg.SOLVER.WARMUP_STEPS
                multiplier = WarmupParamScheduler(
                    MultiStepParamScheduler(
                        [1, 0.1, 0.01, 0.001],
                        milestones=[0.5 * max_iter, 0.625 * max_iter, 0.75 * max_iter, 0.95 * max_iter],
                    ),
                    warmup_factor=warmup_factor,
                    warmup_length=0.5,
                )
                scheduler = LRMultiplier(optimizer, multiplier, max_iter=max_iter)
            except Exception as e:
                logger.warning(f"Error configuring custom LR scheduler: {str(e)}. Using default scheduler.")
                scheduler = build_lr_scheduler(self.cfg, optimizer)
        else:
            scheduler = build_lr_scheduler(self.cfg, optimizer)

        # Setup checkpointing
        checkpointer = DetectionCheckpointer(
            self.model, self.cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
        )
        
        start_iter = checkpointer.resume_or_load(
            self.cfg.MODEL.WEIGHTS, resume=resume
        ).get("iteration", -1) + 1
        
        periodic_checkpointer = PeriodicCheckpointer(
            checkpointer,
            self.cfg.SOLVER.CHECKPOINT_PERIOD,
            max_iter=max_iter
        )

        # Setup logging
        writers = (
            default_writers(self.cfg.OUTPUT_DIR, max_iter) 
            if comm.is_main_process() else []
        )

        # Initialize TensorBoard writer
        tb_writer = SummaryWriter(log_dir=self.cfg.OUTPUT_DIR) if comm.is_main_process() else None

        # Initialize training
        # data_loader = build_detection_train_loader(self.cfg, mapper=self._custom_mapper)
        
        # # Identify and oversample minority classes
        # dataset_dicts = DatasetCatalog.get("my_dataset_train")
        # repeat_factors = RepeatFactorTrainingSampler.repeat_factors_from_category_frequency(
        #     dataset_dicts, repeat_thresh=0.05
        # )
        # Create data loader with custom mapper and sampler
        
        data_loader = build_detection_train_loader(
            self.cfg
        )

        logger.info(f"Starting training from iteration {start_iter}")
        
        best_loss = float('inf')
        patience_counter = 0
        losses_history = []
        lr_history = []

        with EventStorage(start_iter) as storage:
            for data, iteration in zip(data_loader, range(start_iter, max_iter)):
                storage.iter = iteration

                loss_dict = self.model(data)
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
                current_lr = optimizer.param_groups[0]["lr"]

                # Record metrics
                losses_history.append(losses_reduced)
                lr_history.append(current_lr)

                # Log learning rate to TensorBoard
                if comm.is_main_process() and tb_writer is not None:
                    tb_writer.add_scalar('learning_rate', current_lr, iteration)

                # Periodic evaluation
                if (
                    self.cfg.TEST.EVAL_PERIOD > 0
                    and (iteration + 1) % self.cfg.TEST.EVAL_PERIOD == 0
                    and iteration != max_iter - 1
                ):
                    test_results = self.do_test()
                    # Log evaluation metrics to TensorBoard
                    if comm.is_main_process() and tb_writer is not None:
                        for metric_name, metric_value in test_results.items():
                            if isinstance(metric_value, (int, float)):
                                tb_writer.add_scalar(f'eval/{metric_name}', metric_value, iteration)
                    comm.synchronize()

                # Write logs
                if iteration - start_iter > 5 and (
                    (iteration + 1) % 20 == 0 or iteration == max_iter - 1
                ):
                    for writer in writers:
                        writer.write()
                
                periodic_checkpointer.step(iteration)

                # Early stopping check
                if losses_reduced < best_loss:
                    best_loss = losses_reduced
                    logger.info(f"New best loss: {best_loss:.4f}")
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter % 100 == 0:
                        logger.info(f"******************* Loss hasn't improved for {patience_counter} iterations *******************")
                    if patience_counter >= self.cfg.SOLVER.PATIENCE:
                        logger.info("******************* Early stopping triggered *******************")
                        checkpointer.save("model_final")
                        break

        # Close TensorBoard writer
        if tb_writer is not None:
            tb_writer.close()

        return {
            'losses': losses_history,
            'learning_rates': lr_history,
            'best_loss': best_loss
        }
    
    def do_test(self, model=None):
        """
        Run model evaluation on test datasets.
        
        Args:
            model: Model to evaluate. If None, uses self.model
            
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
            for dataset_name in self.cfg.DATASETS.TEST:
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

    def _custom_mapper(self, dataset_dict):
        import torch
        from detectron2.data import detection_utils as utils
        import detectron2.data.transforms as T
        import copy

        
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format="BGR")

        # Apply augmentations only to specific classes
        augment_classes = {1, 3, 5}  # Set of underrepresented class IDs
        augment = any(anno["category_id"] in augment_classes for anno in dataset_dict["annotations"])

        transform_list = [
            T.ResizeShortestEdge(short_edge_length=(800, 1024), max_size=1333),
            T.RandomBrightness(0.8, 1.2) if augment else T.NoOpTransform(),
            T.RandomFlip(prob=0.5, horizontal=True, vertical=False) if augment else T.NoOpTransform(),
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
