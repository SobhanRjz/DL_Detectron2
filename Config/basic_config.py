# base_config.py
"""
Base configuration module containing paths and general settings.
"""
import os
import torch
import logging

# Define base paths in a more readable and maintainable way
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DATA_PATH = r"C:\Users\sobha\Desktop\detectron2\Data\RoboFlowData\MergeData(1-4)\split_dataset" #os.path.join(BASE_PATH, 'DataSets', 'images')
OUTPUT_PATH = os.path.join(BASE_PATH, 'output')

# General settings
NUM_WORKERS = 4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Logging configuration
LOGGING_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOGGING_LEVEL = 'INFO'

def setup_logger(name="detectron2", log_file=None, level=LOGGING_LEVEL):
    """
    Sets up a logger with a specified name, level, and optional log file.
    Prevents multiple initializations of the same logger.

    Args:
        name (str): The name of the logger.
        log_file (str): Optional file to log messages to.
        level (str): Logging level (e.g., 'INFO', 'DEBUG').

    Returns:
        logging.Logger: Configured logger instance.
    """
    # Get the logger
    logger = logging.getLogger(name)
    
    # If the logger already has handlers, return it to avoid duplicate logging
    if logger.hasHandlers():
        return logger
    
    # Set level
    logger.setLevel(getattr(logging, level))

    # Create formatter
    formatter = logging.Formatter(LOGGING_FORMAT)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

# Initialize the Detectron2 logger only if it hasn't been initialized
detectron2_logger = setup_logger("detectron2", 
                               log_file=os.path.join(OUTPUT_PATH, "detectron2.log"))