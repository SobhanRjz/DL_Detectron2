from .default_trainer import SimpleDefaultTrainer
from .custom_trainer import CustomTrainer

class TrainerFactory:
    """Factory class to create appropriate trainer instance"""
    
    @staticmethod
    def create_trainer(trainer_type: str, cfg, resumeTrain = False):
        """Create a trainer based on type
        
        Args:
            trainer_type: Type of trainer ("default" or "custom")
            cfg: Configuration object
            
        Returns:
            Trainer instance
        """
        trainer_types = {
            "default": SimpleDefaultTrainer,
            "custom": CustomTrainer
        }
        
        trainer_class = trainer_types.get(trainer_type.lower())
        if trainer_class is None:
            raise ValueError(f"Unknown trainer type: {trainer_type}. Valid types are: {list(trainer_types.keys())}")
            
        return trainer_class(cfg, resumeTrain)
