"""
Configuration Manager for 360FrameTools
Handles saving/loading pipeline configurations as JSON files
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class ConfigManager:
    """
    Manages pipeline configurations for saving/loading user preferences.
    Allows users to save complete pipeline settings and restore them later.
    """
    
    DEFAULT_CONFIG_DIR = Path.home() / '.360toolkit' / 'configs'
    
    def __init__(self):
        """Initialize configuration manager"""
        self.current_config: Dict[str, Any] = {}
        
        # Ensure config directory exists
        self.DEFAULT_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    
    def save_config(self, config: Dict[str, Any], filepath: Optional[Path] = None,
                   config_name: Optional[str] = None) -> bool:
        """
        Save pipeline configuration to JSON file.
        
        Args:
            config: Configuration dictionary to save
            filepath: Optional custom file path (if None, uses default location)
            config_name: Optional config name (used if filepath is None)
        
        Returns:
            True if save successful, False otherwise
        """
        try:
            # Determine save path
            if filepath is None:
                if config_name is None:
                    # Generate timestamp-based name
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    config_name = f"config_{timestamp}"
                
                filepath = self.DEFAULT_CONFIG_DIR / f"{config_name}.json"
            
            # Add metadata
            save_data = {
                'metadata': {
                    'saved_at': datetime.now().isoformat(),
                    'version': '1.0',
                    'config_name': config_name or filepath.stem
                },
                'pipeline_config': config
            }
            
            # Save to JSON
            filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2)
            
            logger.info(f"Configuration saved to: {filepath}")
            self.current_config = config
            return True
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            return False
    
    def load_config(self, filepath: Path) -> Optional[Dict[str, Any]]:
        """
        Load pipeline configuration from JSON file.
        
        Args:
            filepath: Path to configuration file
        
        Returns:
            Configuration dictionary, or None if load failed
        """
        try:
            if not filepath.exists():
                logger.error(f"Configuration file not found: {filepath}")
                return None
            
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract pipeline config
            if 'pipeline_config' in data:
                config = data['pipeline_config']
                metadata = data.get('metadata', {})
                logger.info(f"Configuration loaded: {metadata.get('config_name', filepath.stem)}")
                logger.info(f"Saved at: {metadata.get('saved_at', 'Unknown')}")
            else:
                # Legacy format without metadata
                config = data
                logger.info(f"Configuration loaded (legacy format): {filepath.stem}")
            
            self.current_config = config
            return config
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return None
    
    def list_saved_configs(self) -> list:
        """
        List all saved configurations in default directory.
        
        Returns:
            List of tuples: (filepath, config_name, saved_date)
        """
        try:
            configs = []
            
            if not self.DEFAULT_CONFIG_DIR.exists():
                return configs
            
            for json_file in self.DEFAULT_CONFIG_DIR.glob('*.json'):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    metadata = data.get('metadata', {})
                    config_name = metadata.get('config_name', json_file.stem)
                    saved_at = metadata.get('saved_at', 'Unknown')
                    
                    configs.append((json_file, config_name, saved_at))
                    
                except Exception as e:
                    logger.warning(f"Could not read config {json_file}: {e}")
            
            # Sort by date (newest first)
            configs.sort(key=lambda x: x[2], reverse=True)
            return configs
            
        except Exception as e:
            logger.error(f"Failed to list configurations: {e}")
            return []
    
    def delete_config(self, filepath: Path) -> bool:
        """
        Delete a saved configuration file.
        
        Args:
            filepath: Path to configuration file to delete
        
        Returns:
            True if deletion successful, False otherwise
        """
        try:
            if filepath.exists():
                filepath.unlink()
                logger.info(f"Configuration deleted: {filepath}")
                return True
            else:
                logger.warning(f"Configuration file not found: {filepath}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to delete configuration: {e}")
            return False
    
    def export_config_with_description(self, config: Dict[str, Any], filepath: Path,
                                      description: str = "") -> bool:
        """
        Export configuration with additional description/notes.
        
        Args:
            config: Configuration dictionary
            filepath: Save path
            description: User description/notes
        
        Returns:
            True if export successful
        """
        try:
            export_data = {
                'metadata': {
                    'saved_at': datetime.now().isoformat(),
                    'version': '1.0',
                    'config_name': filepath.stem,
                    'description': description
                },
                'pipeline_config': config
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Configuration exported with description to: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export configuration: {e}")
            return False
    
    def get_default_config(self) -> Dict[str, Any]:
        """
        Get default pipeline configuration.
        
        Returns:
            Default configuration dictionary
        """
        from .defaults import (
            DEFAULT_FPS, DEFAULT_SDK_QUALITY, DEFAULT_EXTRACTION_METHOD,
            DEFAULT_SPLIT_COUNT, DEFAULT_H_FOV, DEFAULT_TRANSFORM_TYPE,
            DEFAULT_OUTPUT_WIDTH, DEFAULT_OUTPUT_HEIGHT,
            DEFAULT_CONFIDENCE_THRESHOLD, DEFAULT_USE_GPU,
            DEFAULT_CUBEMAP_FACE_SIZE, DEFAULT_CUBEMAP_OVERLAP, DEFAULT_CUBEMAP_FOV
        )
        
        return {
            # Stage 1: Extraction
            'stage1_enabled': True,
            'fps_interval': DEFAULT_FPS,
            'extraction_method': DEFAULT_EXTRACTION_METHOD,
            'sdk_quality': DEFAULT_SDK_QUALITY,
            'output_format': 'png',
            
            # Stage 2: Transform
            'stage2_enabled': True,
            'transform_type': DEFAULT_TRANSFORM_TYPE,
            'split_count': DEFAULT_SPLIT_COUNT,
            'h_fov': DEFAULT_H_FOV,
            'output_width': DEFAULT_OUTPUT_WIDTH,
            'output_height': DEFAULT_OUTPUT_HEIGHT,
            'cubemap_face_size': DEFAULT_CUBEMAP_FACE_SIZE,
            'cubemap_overlap': DEFAULT_CUBEMAP_OVERLAP,
            'cubemap_fov': DEFAULT_CUBEMAP_FOV,
            'skip_transform': False,
            
            # Stage 3: Masking
            'stage3_enabled': True,
            'model_size': 'small',
            'confidence_threshold': DEFAULT_CONFIDENCE_THRESHOLD,
            'use_gpu': DEFAULT_USE_GPU,
            'masking_categories': {
                'persons': True,
                'personal_objects': True,
                'animals': True
            },
            
            # Stage 4: Alignment
            'stage4_enabled': True,
            'alignment_tool': 'glomap',
            'use_gpu_colmap': True,
            
            # Stage 5: Export
            'stage5_enabled': True,
            'export_lithcfeld': True,
            'export_realityscan': True,
            'export_colmap': False,
            
            # I/O
            'input_file': '',
            'output_dir': ''
        }
    
    def validate_config(self, config: Dict[str, Any]) -> tuple[bool, list]:
        """
        Validate configuration for completeness and correctness.
        
        Args:
            config: Configuration dictionary to validate
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check required fields
        if 'input_file' not in config or not config.get('input_file'):
            errors.append("Input file not specified")
        
        if 'output_dir' not in config or not config.get('output_dir'):
            errors.append("Output directory not specified")
        
        # Validate numeric ranges
        if config.get('fps_interval', 0) <= 0:
            errors.append("FPS interval must be greater than 0")
        
        if not (1 <= config.get('split_count', 0) <= 12):
            errors.append("Split count must be between 1 and 12")
        
        if not (30 <= config.get('h_fov', 0) <= 150):
            errors.append("Horizontal FOV must be between 30° and 150°")
        
        if not (0.0 <= config.get('confidence_threshold', 0) <= 1.0):
            errors.append("Confidence threshold must be between 0.0 and 1.0")
        
        is_valid = len(errors) == 0
        return is_valid, errors


# Global instance
_config_manager = None

def get_config_manager() -> ConfigManager:
    """Get global ConfigManager instance"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager
