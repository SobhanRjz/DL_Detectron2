from .base_trainer import BaseTrainer
# Add your custom training logic imports here

class CustomTrainer(BaseTrainer):

    def __init__(self, cfg, resumeTrain = False):
        self.resumeTrain = resumeTrain
        super().__init__(cfg)


    def do_train(self):
        import torch
        from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
        from detectron2.data import build_detection_train_loader
        from detectron2.engine import default_writers
        from detectron2.solver import build_lr_scheduler, build_optimizer
        from detectron2.solver import LRMultiplier, WarmupParamScheduler
        from fvcore.common.param_scheduler import MultiStepParamScheduler, CosineParamScheduler
        from detectron2.utils import comm
        from detectron2.utils.events import EventStorage
        from Config.basic_config import detectron2_logger as logger
        from torch.utils.tensorboard import SummaryWriter
        from detectron2.data.samplers import RepeatFactorTrainingSampler
        from detectron2.data import DatasetCatalog, MetadataCatalog


        # Remove TensorBoard event files from output directory
        import os
        import glob
        if os.path.exists(self.cfg.OUTPUT_DIR):
            event_files = glob.glob(os.path.join(self.cfg.OUTPUT_DIR, "events.out.tfevents.*"))
            for f in event_files:
                try:
                    os.remove(f)
                    logger.info(f"Removed TensorBoard event file: {f}")
                except OSError as e:
                    logger.warning(f"Error removing file {f}: {e}")
                    
        # Build optimizer and scheduler
        optimizer = build_optimizer(self.cfg, self.model)
        max_iter = self.cfg.SOLVER.MAX_ITER
        
        # Configure learning rate scheduler
        IsCustomWarmup = False
        if IsCustomWarmup:
            try:
                warmup_factor = self.cfg.SOLVER.BASE_LR / self.cfg.SOLVER.WARMUP_STEPS
                warmup_length = 0.1  # Shorter warmup phase
                milestones = [0.5 * max_iter, 0.75 * max_iter, 0.9 * max_iter]  # Adjusted milestones

                # Choose between MultiStep or Cosine Annealing
                use_cosine = getattr(self.cfg.SOLVER, 'USE_COSINE_ANNEALING', False)

                if use_cosine:
                    # Cosine Annealing Scheduler
                    multiplier = WarmupParamScheduler(
                        CosineParamScheduler(start_value=1.0, end_value=0.0),  # Cosine decay
                        warmup_factor=warmup_factor,
                        warmup_length=warmup_length,
                    )
                else:
                    # MultiStep Scheduler
                    multiplier = WarmupParamScheduler(
                        MultiStepParamScheduler(
                            [1, 0.1, 0.01],  # 3 decay steps
                            milestones=milestones,
                        ),
                        warmup_factor=warmup_factor,
                        warmup_length=warmup_length,
                    )
                
                # Apply the scheduler
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
            self.cfg.MODEL.WEIGHTS, resume=self.resumeTrain
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
        data_loader = build_detection_train_loader(self.cfg, mapper=self._custom_mapper, )
        
        # Identify and oversample minority classes
        dataset_dicts = DatasetCatalog.get("my_dataset_train")
        repeat_factors = RepeatFactorTrainingSampler.repeat_factors_from_category_frequency(
            dataset_dicts, repeat_thresh= self.cfg.DATALOADER.REPEAT_THRESHOLD
        )
        # Print sampling statistics
        # if comm.is_main_process():
        #     category_freq = self.get_category_frequency(dataset_dicts)
        #     for cat_id, freq in category_freq.items():
        #         # Get the average repeat factor for this category
        #         cat_repeat_factors = []
        #         for dataset_dict in dataset_dicts:
        #             if any(ann["category_id"] == cat_id for ann in dataset_dict["annotations"]):
        #                 idx = dataset_dict["image_id"]  # or appropriate index
        #                 cat_repeat_factors.append(repeat_factors[idx])
                
        #         avg_rep_factor = sum(cat_repeat_factors) / len(cat_repeat_factors) if cat_repeat_factors else 1.0
        #         print(f"Category {cat_id}: freq={freq:.2f}, rep={avg_rep_factor:.2f}")

            # Build sampler
        # Create sampler with correct arguments
        sampler = RepeatFactorTrainingSampler(
            repeat_factors=repeat_factors,  # Only needs repeat_factors
            shuffle=True,     # Optional keyword argument
            seed=42
        )

        data_loader = build_detection_train_loader(self.cfg, dataset=dataset_dicts, mapper=self._custom_mapper, sampler=sampler)
        # Create data loader with custom mapper and sampler
        
        # data_loader = build_detection_train_loader(
        #     self.cfg
        # )

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
                    # if hasattr(self.cfg.SOLVER, 'PATIENCE') and patience_counter >= self.cfg.SOLVER.PATIENCE:
                    #     logger.info("******************* Early stopping triggered *******************")
                    #     checkpointer.save("model_final")
                    #     break

        logger.info(f"Start Test evaluation ===================================>  my_dataset_test")
        self.do_test(dataset_name="my_dataset_test")
        # Close TensorBoard writer
        if tb_writer is not None:
            tb_writer.close()

        return {
            'losses': losses_history,
            'learning_rates': lr_history,
            'best_loss': best_loss
        }
    
    def do_test(self, dataset_name="my_dataset_valid"):
        """
        Run model evaluation on a specific test dataset.
        
        Args:
            dataset_name: Name of dataset to evaluate on. Defaults to "my_dataset_valid"
            
        Returns:
            Evaluation results for the dataset
        """
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

        try:
            # Create data loader for test dataset
            data_loader = build_detection_test_loader(self.cfg, dataset_name)
            
            # Initialize evaluator
            output_folder = os.path.join(self.cfg.OUTPUT_DIR, "inference", dataset_name)
            evaluator = get_evaluator(
                self.cfg, dataset_name, output_folder
            )
            
            # Run inference
            results = inference_on_dataset(self.model, data_loader, evaluator)
            
            # Log results
            if comm.is_main_process():
                logger.info(f"Evaluation results for {dataset_name} in csv format:")
                print_csv_format(results)
                
        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}")
            raise
            
        return results

    def _custom_mapper(self, dataset_dict = "my_dataset_valid"):
        import torch
        from detectron2.data import detection_utils as utils
        import detectron2.data.transforms as T
        import copy

        
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format="BGR")

        transform_list = [
            T.ResizeShortestEdge(short_edge_length=(640, 640), max_size=800),
            T.RandomBrightness(0.75, 1.25),
            # T.RandomContrast(0.75, 1.25),
            T.RandomCrop(crop_type="relative_range", crop_size=(0.8, 0.8)),
            T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
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
        from collections import defaultdict
        """Calculate frequency of each category in the dataset"""
        category_count = defaultdict(int)
        total_annotations = 0
        
        # Count instances of each category
        for dataset_dict in dataset_dicts:
            for annotation in dataset_dict["annotations"]:
                cat_id = annotation["category_id"]
                category_count[cat_id] += 1
                total_annotations += 1
        
        # Calculate frequencies
        category_freq = {
            cat_id: count / total_annotations 
            for cat_id, count in category_count.items()
        }
    
        return category_freq