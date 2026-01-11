"""Settings Manager for pipeline configuration persistence.

This module provides persistence and management of pipeline settings
including blur, illumination, OCR, and calibration configurations.

Feature: ocr-enhancement-improvements
Validates: Requirements 9.1, 9.2, 9.3, 9.4, 9.5
"""

import json
import logging
import os
from dataclasses import dataclass, asdict, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class PipelineSettings:
    """Complete pipeline settings for persistence.
    
    Attributes:
        blur_threshold: Blur threshold for deblur decision
        blur_auto_mode: Whether blur threshold uses auto-calibration
        deblur_enabled: Whether deblurring is enabled
        low_light_threshold: Luminance threshold for low-light detection (0-255)
        gamma_value: Gamma correction value (< 1 brightens, > 1 darkens)
        illumination_auto_mode: Whether illumination uses auto-calibration
        ocr_language: Language code for OCR (e.g., 'en')
        ocr_gpu_enabled: Whether OCR uses GPU acceleration
        ocr_frame_interval: OCR frame interval (run OCR every Nth frame, 1-30)
        calibration_sample_size: Number of frames for calibration
        blur_percentile: Percentile for blur threshold calculation (0-100)
        luminance_percentile: Percentile for luminance threshold calculation (0-100)
    """
    # Blur settings
    blur_threshold: float = 100.0
    blur_auto_mode: bool = True
    deblur_enabled: bool = True
    
    # Illumination settings
    low_light_threshold: int = 80
    gamma_value: float = 1.0
    illumination_auto_mode: bool = True
    
    # OCR settings
    ocr_language: str = 'en'
    ocr_gpu_enabled: bool = True
    ocr_frame_interval: int = 5
    
    # Calibration settings
    calibration_sample_size: int = 30
    blur_percentile: float = 50.0
    luminance_percentile: float = 25.0


    def validate(self) -> None:
        """Validate settings values.
        
        Raises:
            ValueError: If any setting value is invalid
        """
        if self.blur_threshold < 0:
            raise ValueError(f"blur_threshold must be non-negative, got {self.blur_threshold}")
        
        if not 0 <= self.low_light_threshold <= 255:
            raise ValueError(f"low_light_threshold must be in [0, 255], got {self.low_light_threshold}")
        
        if self.gamma_value <= 0:
            raise ValueError(f"gamma_value must be positive, got {self.gamma_value}")
        
        if not 1 <= self.ocr_frame_interval <= 30:
            raise ValueError(f"ocr_frame_interval must be in [1, 30], got {self.ocr_frame_interval}")
        
        if self.calibration_sample_size < 1:
            raise ValueError(f"calibration_sample_size must be at least 1, got {self.calibration_sample_size}")
        
        if not 0 <= self.blur_percentile <= 100:
            raise ValueError(f"blur_percentile must be in [0, 100], got {self.blur_percentile}")
        
        if not 0 <= self.luminance_percentile <= 100:
            raise ValueError(f"luminance_percentile must be in [0, 100], got {self.luminance_percentile}")


class SettingsManager:
    """Manages persistence of pipeline settings.
    
    The SettingsManager handles:
    - Loading settings from JSON file
    - Saving settings to JSON file
    - Resetting to default values
    - Partial updates for blur and illumination settings
    
    Settings are stored in JSON format for easy editing and inspection.
    
    Attributes:
        DEFAULT_SETTINGS_PATH: Default path for settings file
    """
    
    DEFAULT_SETTINGS_PATH = "pipeline_settings.json"
    
    def __init__(self, settings_path: Optional[str] = None):
        """Initialize settings manager.
        
        Args:
            settings_path: Path to settings file. If None, uses DEFAULT_SETTINGS_PATH.
        """
        self._settings_path = settings_path or self.DEFAULT_SETTINGS_PATH
        self._current_settings: Optional[PipelineSettings] = None
    
    @property
    def settings_path(self) -> str:
        """Get the current settings file path.
        
        Returns:
            Path to the settings file
        """
        return self._settings_path
    
    def load_settings(self) -> PipelineSettings:
        """Load settings from file.
        
        If the settings file doesn't exist, returns default settings.
        If the file is corrupted, logs a warning and returns default settings.
        
        Returns:
            PipelineSettings with loaded or default values
        """
        if not os.path.exists(self._settings_path):
            logger.info(f"Settings file not found at {self._settings_path}, using defaults")
            self._current_settings = PipelineSettings()
            return self._current_settings
        
        try:
            with open(self._settings_path, 'r') as f:
                data = json.load(f)
            
            # Create settings from loaded data, using defaults for missing fields
            settings = PipelineSettings(
                blur_threshold=data.get('blur_threshold', 100.0),
                blur_auto_mode=data.get('blur_auto_mode', True),
                deblur_enabled=data.get('deblur_enabled', True),
                low_light_threshold=data.get('low_light_threshold', 80),
                gamma_value=data.get('gamma_value', 1.0),
                illumination_auto_mode=data.get('illumination_auto_mode', True),
                ocr_language=data.get('ocr_language', 'en'),
                ocr_gpu_enabled=data.get('ocr_gpu_enabled', True),
                ocr_frame_interval=data.get('ocr_frame_interval', 5),
                calibration_sample_size=data.get('calibration_sample_size', 30),
                blur_percentile=data.get('blur_percentile', 50.0),
                luminance_percentile=data.get('luminance_percentile', 25.0)
            )
            
            # Validate loaded settings
            settings.validate()
            
            self._current_settings = settings
            logger.info(f"Settings loaded from {self._settings_path}")
            return settings
            
        except json.JSONDecodeError as e:
            logger.warning(f"Corrupted settings file at {self._settings_path}: {e}. Using defaults.")
            self._current_settings = PipelineSettings()
            return self._current_settings
        except ValueError as e:
            logger.warning(f"Invalid settings values in {self._settings_path}: {e}. Using defaults.")
            self._current_settings = PipelineSettings()
            return self._current_settings
        except Exception as e:
            logger.warning(f"Error loading settings from {self._settings_path}: {e}. Using defaults.")
            self._current_settings = PipelineSettings()
            return self._current_settings
    
    def save_settings(self, settings: PipelineSettings) -> None:
        """Save settings to file.
        
        Args:
            settings: PipelineSettings to save
            
        Raises:
            IOError: If unable to write to the settings file
            ValueError: If settings are invalid
        """
        # Validate before saving
        settings.validate()
        
        try:
            # Convert to dict for JSON serialization
            data = asdict(settings)
            
            # Ensure directory exists
            settings_dir = os.path.dirname(self._settings_path)
            if settings_dir and not os.path.exists(settings_dir):
                os.makedirs(settings_dir)
            
            with open(self._settings_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            self._current_settings = settings
            logger.info(f"Settings saved to {self._settings_path}")
            
        except PermissionError as e:
            raise IOError(f"Permission denied writing to {self._settings_path}: {e}")
        except Exception as e:
            raise IOError(f"Error saving settings to {self._settings_path}: {e}")
    
    def reset_to_defaults(self) -> PipelineSettings:
        """Reset settings to defaults and save.
        
        Creates a new PipelineSettings with default values and saves it.
        
        Returns:
            PipelineSettings with default values
        """
        default_settings = PipelineSettings()
        self.save_settings(default_settings)
        logger.info("Settings reset to defaults")
        return default_settings
    
    def get_current_settings(self) -> PipelineSettings:
        """Get current in-memory settings.
        
        If settings haven't been loaded yet, loads them first.
        
        Returns:
            Current PipelineSettings
        """
        if self._current_settings is None:
            return self.load_settings()
        return self._current_settings
    
    def update_blur_settings(
        self,
        threshold: Optional[float] = None,
        auto_mode: Optional[bool] = None,
        deblur_enabled: Optional[bool] = None
    ) -> None:
        """Update blur-related settings.
        
        Only updates the specified fields, leaving others unchanged.
        Automatically saves after update.
        
        Args:
            threshold: New blur threshold (optional)
            auto_mode: New auto mode setting (optional)
            deblur_enabled: New deblur enabled setting (optional)
            
        Raises:
            ValueError: If threshold is negative
        """
        settings = self.get_current_settings()
        
        if threshold is not None:
            if threshold < 0:
                raise ValueError(f"threshold must be non-negative, got {threshold}")
            settings.blur_threshold = threshold
        
        if auto_mode is not None:
            settings.blur_auto_mode = auto_mode
        
        if deblur_enabled is not None:
            settings.deblur_enabled = deblur_enabled
        
        self.save_settings(settings)
        logger.debug(f"Blur settings updated: threshold={settings.blur_threshold}, "
                    f"auto_mode={settings.blur_auto_mode}, deblur_enabled={settings.deblur_enabled}")
    
    def update_illumination_settings(
        self,
        low_light_threshold: Optional[int] = None,
        gamma_value: Optional[float] = None,
        auto_mode: Optional[bool] = None
    ) -> None:
        """Update illumination-related settings.
        
        Only updates the specified fields, leaving others unchanged.
        Automatically saves after update.
        
        Args:
            low_light_threshold: New low-light threshold (optional, 0-255)
            gamma_value: New gamma value (optional, must be positive)
            auto_mode: New auto mode setting (optional)
            
        Raises:
            ValueError: If low_light_threshold is not in [0, 255]
            ValueError: If gamma_value is not positive
        """
        settings = self.get_current_settings()
        
        if low_light_threshold is not None:
            if not 0 <= low_light_threshold <= 255:
                raise ValueError(f"low_light_threshold must be in [0, 255], got {low_light_threshold}")
            settings.low_light_threshold = low_light_threshold
        
        if gamma_value is not None:
            if gamma_value <= 0:
                raise ValueError(f"gamma_value must be positive, got {gamma_value}")
            settings.gamma_value = gamma_value
        
        if auto_mode is not None:
            settings.illumination_auto_mode = auto_mode
        
        self.save_settings(settings)
        logger.debug(f"Illumination settings updated: threshold={settings.low_light_threshold}, "
                    f"gamma={settings.gamma_value}, auto_mode={settings.illumination_auto_mode}")
    
    def update_ocr_settings(
        self,
        language: Optional[str] = None,
        gpu_enabled: Optional[bool] = None,
        frame_interval: Optional[int] = None
    ) -> None:
        """Update OCR-related settings.
        
        Only updates the specified fields, leaving others unchanged.
        Automatically saves after update.
        
        Args:
            language: New OCR language code (optional)
            gpu_enabled: New GPU enabled setting (optional)
            frame_interval: New OCR frame interval (optional, 1-30)
            
        Raises:
            ValueError: If frame_interval is not in [1, 30]
        """
        settings = self.get_current_settings()
        
        if language is not None:
            settings.ocr_language = language
        
        if gpu_enabled is not None:
            settings.ocr_gpu_enabled = gpu_enabled
        
        if frame_interval is not None:
            if not 1 <= frame_interval <= 30:
                raise ValueError(f"frame_interval must be in [1, 30], got {frame_interval}")
            settings.ocr_frame_interval = frame_interval
        
        self.save_settings(settings)
        logger.debug(f"OCR settings updated: language={settings.ocr_language}, "
                    f"gpu_enabled={settings.ocr_gpu_enabled}, "
                    f"frame_interval={settings.ocr_frame_interval}")
    
    def update_calibration_settings(
        self,
        sample_size: Optional[int] = None,
        blur_percentile: Optional[float] = None,
        luminance_percentile: Optional[float] = None
    ) -> None:
        """Update calibration-related settings.
        
        Only updates the specified fields, leaving others unchanged.
        Automatically saves after update.
        
        Args:
            sample_size: New calibration sample size (optional, must be >= 1)
            blur_percentile: New blur percentile (optional, 0-100)
            luminance_percentile: New luminance percentile (optional, 0-100)
            
        Raises:
            ValueError: If sample_size < 1
            ValueError: If percentile values are not in [0, 100]
        """
        settings = self.get_current_settings()
        
        if sample_size is not None:
            if sample_size < 1:
                raise ValueError(f"sample_size must be at least 1, got {sample_size}")
            settings.calibration_sample_size = sample_size
        
        if blur_percentile is not None:
            if not 0 <= blur_percentile <= 100:
                raise ValueError(f"blur_percentile must be in [0, 100], got {blur_percentile}")
            settings.blur_percentile = blur_percentile
        
        if luminance_percentile is not None:
            if not 0 <= luminance_percentile <= 100:
                raise ValueError(f"luminance_percentile must be in [0, 100], got {luminance_percentile}")
            settings.luminance_percentile = luminance_percentile
        
        self.save_settings(settings)
        logger.debug(f"Calibration settings updated: sample_size={settings.calibration_sample_size}, "
                    f"blur_percentile={settings.blur_percentile}, "
                    f"luminance_percentile={settings.luminance_percentile}")
