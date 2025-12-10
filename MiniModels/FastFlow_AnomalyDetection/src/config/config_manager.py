"""Configuration management for FastFlow"""

import os
import yaml
from typing import Dict, Any, Optional


class ConfigManager:
    """Manages configuration loading and validation"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize config manager
        
        Args:
            config_path: Path to YAML config file (optional)
        """
        self.config_path = config_path
        self.config = {}
        
        if config_path:
            self.load(config_path)
    
    def load(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file
        
        Args:
            config_path: Path to YAML config file
            
        Returns:
            Dictionary containing configuration
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config is invalid
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.config_path = config_path
        self._validate()
        return self.config
    
    def _validate(self):
        """Validate configuration parameters
        
        Raises:
            ValueError: If required parameters are missing or invalid
        """
        required_keys = ['backbone_name', 'input_size', 'flow_step', 
                        'conv3x3_only', 'hidden_ratio']
        
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        return self.config.get(key, default)
    
    def __getitem__(self, key: str) -> Any:
        """Get configuration value using bracket notation"""
        return self.config[key]
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists in configuration"""
        return key in self.config
    
    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary"""
        return self.config.copy()
    
    def __repr__(self) -> str:
        """String representation of config"""
        return f"ConfigManager(config_path='{self.config_path}')"
    
    def __str__(self) -> str:
        """Pretty print configuration"""
        return yaml.dump(self.config, default_flow_style=False)

