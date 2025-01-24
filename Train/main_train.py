from detectron2.checkpoint import DetectionCheckpointer
from Config.basic_config import detectron2_logger as logger
from .factory_trainer import TrainerFactory

def mainTrain(cfg, trainer_type="default", IsResume = False):
    """Main training function that handles model training and evaluation
    
    Args:
        cfg: Configuration object containing model and training settings
        trainer_type: Type of trainer to use ("default" or "custom")
        
    Returns:
        Test results after training or evaluation
    """
    logger.info(f"Initializing {trainer_type} trainer...")
    trainer = TrainerFactory.create_trainer(trainer_type, cfg, IsResume)
    
    # Check if evaluation only mode is enabled in config
    eval_only = cfg.get("EVAL_ONLY", IsResume)
    eval_only = False
    if eval_only:
        logger.info("Running evaluation only mode...")
        checkpointer = DetectionCheckpointer(
            trainer.model, 
            save_dir=cfg.OUTPUT_DIR
        )
        checkpointer.resume_or_load(
            cfg.MODEL.WEIGHTS,
            resume=IsResume
        )
        return trainer.do_test()

    # Train and evaluate
    logger.info("Starting training...")
    trainer.do_train()
    
    logger.info("Training completed. Running evaluation...")
