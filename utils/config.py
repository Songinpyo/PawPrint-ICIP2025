import yaml
from pathlib import Path
from typing import Dict, Any

def load_config(config_path: str) -> Dict[Any, Any]:
    """
    Load config file with inheritance support
    
    Args:
        config_path: Path to config file
        
    Returns:
        Dict containing merged configuration
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # If config has parent, merge with parent config
    if 'parent' in config:
        parent_path = Path(config_path).parent.parent / f"{config['parent']}.yaml"
        with open(parent_path, 'r') as f:
            parent_config = yaml.safe_load(f)
        
        # Remove parent key from child config
        del config['parent']
        
        # Merge configs (child overrides parent)
        merged_config = deep_merge(parent_config, config)
        return merged_config
    
    return config

def deep_merge(parent: Dict, child: Dict) -> Dict:
    """
    Recursively merge two dictionaries
    Child values override parent values
    """
    merged = parent.copy()
    
    for key, value in child.items():
        if (
            key in parent and 
            isinstance(parent[key], dict) and 
            isinstance(value, dict)
        ):
            merged[key] = deep_merge(parent[key], value)
        else:
            merged[key] = value
            
    return merged 