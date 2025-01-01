from detectron2.utils import comm
from torch.nn.parallel import DistributedDataParallel
from detectron2.modeling import build_model
from Config.basic_config import detectron2_logger as logger
from abc import ABC, abstractmethod

class BaseTrainer(ABC):
    """Base trainer class that defines the training interface"""
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = build_model(cfg)
        logger.info(f"Model:\n{self.model}")
        
        # Set up distributed training if using multiple GPUs
        self.distributed = comm.get_world_size() > 1
        if self.distributed:
            self.model = DistributedDataParallel(
                self.model,
                device_ids=[comm.get_local_rank()],
                broadcast_buffers=False,
                find_unused_parameters=True  # Optimize memory usage
            )
            
    @abstractmethod
    def do_train(self, resume: bool = False) -> None:
        """Training logic to be implemented by child classes"""
        pass
        
    @abstractmethod
    def do_test(self) -> None:
        """Testing logic to be implemented by child classes"""
        pass 