import yaml
import logging
from typing import Dict, Any

def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    validate_config(config)
    return config

def validate_config(config: Dict[str, Any]):
    required_keys = ['data', 'preprocessing', 'modeling']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration section: {key}")
    
    data_cfg = config.get('data', {})
    if not data_cfg.get('files'):
        raise ValueError("Configuration must specify 'data.files'.")
    if not data_cfg.get('target_col'):
        raise ValueError("Configuration must specify 'data.target_col'.")
    if not data_cfg.get('date_col'):
        raise ValueError("Configuration must specify 'data.date_col'.")
