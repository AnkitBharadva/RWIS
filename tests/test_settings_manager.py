"""
Property-based tests for SettingsManager module.

Feature: ocr-enhancement-improvements
Validates: Requirements 5.7, 6.7, 9.1, 9.2, 9.3, 9.4, 9.5
"""

import os
import tempfile
import pytest
from hypothesis import given, strategies as st, settings, assume

from utils.settings_manager import SettingsManager, PipelineSettings


# Strategy for generating valid blur thresholds
blur_threshold_strategy = st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False)

# Strategy for generating valid low-light thresholds (0-255)
low_light_threshold_strategy = st.integers(min_value=0, max_value=255)

# Strategy for generating valid gamma values (positive)
gamma_value_strategy = st.floats(min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False)

# Strategy for generating valid percentiles (0-100)
percentile_strategy = st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False)

# Strategy for generating valid sample sizes
sample_size_strategy = st.integers(min_value=1, max_value=1000)

# Strategy for generating valid OCR frame intervals (1-30)
ocr_frame_interval_strategy = st.integers(min_value=1, max_value=30)

# Strategy for generating valid language codes
language_strategy = st.sampled_from(['en', 'ch_sim', 'ch_tra', 'ja', 'ko', 'de', 'fr', 'es', 'it', 'pt'])


@st.composite
def valid_pipeline_settings(draw):
    """Generate valid PipelineSettings objects."""
    return PipelineSettings(
        blur_threshold=draw(blur_threshold_strategy),
        blur_auto_mode=draw(st.booleans()),
        deblur_enabled=draw(st.booleans()),
        low_light_threshold=draw(low_light_threshold_strategy),
        gamma_value=draw(gamma_value_strategy),
        illumination_auto_mode=draw(st.booleans()),
        ocr_language=draw(language_strategy),
        ocr_gpu_enabled=draw(st.booleans()),
        ocr_frame_interval=draw(ocr_frame_interval_strategy),
        calibration_sample_size=draw(sample_size_strategy),
        blur_percentile=draw(percentile_strategy),
        luminance_percentile=draw(percentile_strategy)
    )


class TestConfigurationRoundTrip:
    """
    Property 8: Configuration Round-Trip
    
    For any valid PipelineSettings object:
    - Saving to file then loading SHALL produce an equivalent settings object
    - All fields SHALL be preserved: blur_threshold, low_light_threshold, gamma_value,
      deblur_enabled, auto_calibration flags
    
    Validates: Requirements 5.7, 6.7, 9.1, 9.2, 9.3, 9.4, 9.5
    """

    @given(settings_obj=valid_pipeline_settings())
    @settings(max_examples=100)
    def test_save_load_round_trip_preserves_all_fields(self, settings_obj):
        """
        Feature: ocr-enhancement-improvements, Property 8: Configuration Round-Trip
        
        Verify that saving and loading settings preserves all fields exactly.
        """
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            manager = SettingsManager(settings_path=temp_path)
            
            # Save the settings
            manager.save_settings(settings_obj)
            
            # Create a new manager to load (simulates restart)
            manager2 = SettingsManager(settings_path=temp_path)
            loaded_settings = manager2.load_settings()
            
            # Verify all fields are preserved
            assert loaded_settings.blur_threshold == settings_obj.blur_threshold, \
                f"blur_threshold mismatch: {loaded_settings.blur_threshold} != {settings_obj.blur_threshold}"
            
            assert loaded_settings.blur_auto_mode == settings_obj.blur_auto_mode, \
                f"blur_auto_mode mismatch: {loaded_settings.blur_auto_mode} != {settings_obj.blur_auto_mode}"
            
            assert loaded_settings.deblur_enabled == settings_obj.deblur_enabled, \
                f"deblur_enabled mismatch: {loaded_settings.deblur_enabled} != {settings_obj.deblur_enabled}"
            
            assert loaded_settings.low_light_threshold == settings_obj.low_light_threshold, \
                f"low_light_threshold mismatch: {loaded_settings.low_light_threshold} != {settings_obj.low_light_threshold}"
            
            # For gamma_value, use approximate comparison due to float serialization
            assert abs(loaded_settings.gamma_value - settings_obj.gamma_value) < 1e-9, \
                f"gamma_value mismatch: {loaded_settings.gamma_value} != {settings_obj.gamma_value}"
            
            assert loaded_settings.illumination_auto_mode == settings_obj.illumination_auto_mode, \
                f"illumination_auto_mode mismatch: {loaded_settings.illumination_auto_mode} != {settings_obj.illumination_auto_mode}"
            
            assert loaded_settings.ocr_language == settings_obj.ocr_language, \
                f"ocr_language mismatch: {loaded_settings.ocr_language} != {settings_obj.ocr_language}"
            
            assert loaded_settings.ocr_gpu_enabled == settings_obj.ocr_gpu_enabled, \
                f"ocr_gpu_enabled mismatch: {loaded_settings.ocr_gpu_enabled} != {settings_obj.ocr_gpu_enabled}"
            
            assert loaded_settings.ocr_frame_interval == settings_obj.ocr_frame_interval, \
                f"ocr_frame_interval mismatch: {loaded_settings.ocr_frame_interval} != {settings_obj.ocr_frame_interval}"
            
            assert loaded_settings.calibration_sample_size == settings_obj.calibration_sample_size, \
                f"calibration_sample_size mismatch: {loaded_settings.calibration_sample_size} != {settings_obj.calibration_sample_size}"
            
            # For percentiles, use approximate comparison due to float serialization
            assert abs(loaded_settings.blur_percentile - settings_obj.blur_percentile) < 1e-9, \
                f"blur_percentile mismatch: {loaded_settings.blur_percentile} != {settings_obj.blur_percentile}"
            
            assert abs(loaded_settings.luminance_percentile - settings_obj.luminance_percentile) < 1e-9, \
                f"luminance_percentile mismatch: {loaded_settings.luminance_percentile} != {settings_obj.luminance_percentile}"
        
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)

    @given(settings_obj=valid_pipeline_settings())
    @settings(max_examples=100)
    def test_multiple_save_load_cycles_preserve_settings(self, settings_obj):
        """
        Feature: ocr-enhancement-improvements, Property 8: Configuration Round-Trip
        
        Verify that multiple save/load cycles preserve settings.
        """
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            # First cycle
            manager1 = SettingsManager(settings_path=temp_path)
            manager1.save_settings(settings_obj)
            loaded1 = manager1.load_settings()
            
            # Second cycle - save loaded settings
            manager2 = SettingsManager(settings_path=temp_path)
            manager2.save_settings(loaded1)
            loaded2 = manager2.load_settings()
            
            # Third cycle - save again
            manager3 = SettingsManager(settings_path=temp_path)
            manager3.save_settings(loaded2)
            loaded3 = manager3.load_settings()
            
            # All loaded settings should be equivalent to original
            assert abs(loaded3.blur_threshold - settings_obj.blur_threshold) < 1e-9
            assert loaded3.blur_auto_mode == settings_obj.blur_auto_mode
            assert loaded3.deblur_enabled == settings_obj.deblur_enabled
            assert loaded3.low_light_threshold == settings_obj.low_light_threshold
            assert abs(loaded3.gamma_value - settings_obj.gamma_value) < 1e-9
            assert loaded3.illumination_auto_mode == settings_obj.illumination_auto_mode
        
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


class TestOCRIntervalPersistence:
    """
    Property 11: Settings Persistence Round-Trip (OCR Interval)
    
    For any OCR interval value V saved to settings, loading settings SHALL return the same value V.
    
    Feature: ocr-visual-enhancements
    Validates: Requirements 4.7
    """

    @given(interval=ocr_frame_interval_strategy)
    @settings(max_examples=100)
    def test_ocr_interval_round_trip(self, interval):
        """
        Feature: ocr-visual-enhancements, Property 11: Settings Persistence Round-Trip
        
        Verify that saving and loading OCR interval preserves the value exactly.
        """
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            # Create settings with the generated interval
            settings_obj = PipelineSettings(ocr_frame_interval=interval)
            
            # Save the settings
            manager = SettingsManager(settings_path=temp_path)
            manager.save_settings(settings_obj)
            
            # Create a new manager to load (simulates restart)
            manager2 = SettingsManager(settings_path=temp_path)
            loaded_settings = manager2.load_settings()
            
            # Verify OCR interval is preserved
            assert loaded_settings.ocr_frame_interval == interval, \
                f"ocr_frame_interval mismatch: {loaded_settings.ocr_frame_interval} != {interval}"
        
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    @given(interval=ocr_frame_interval_strategy)
    @settings(max_examples=100)
    def test_ocr_interval_update_round_trip(self, interval):
        """
        Feature: ocr-visual-enhancements, Property 11: Settings Persistence Round-Trip
        
        Verify that updating OCR interval via update_ocr_settings preserves the value.
        """
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            # Start with default settings
            manager = SettingsManager(settings_path=temp_path)
            manager.save_settings(PipelineSettings())
            
            # Update OCR interval
            manager.update_ocr_settings(frame_interval=interval)
            
            # Create a new manager to load (simulates restart)
            manager2 = SettingsManager(settings_path=temp_path)
            loaded_settings = manager2.load_settings()
            
            # Verify OCR interval is preserved
            assert loaded_settings.ocr_frame_interval == interval, \
                f"ocr_frame_interval mismatch: {loaded_settings.ocr_frame_interval} != {interval}"
        
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


class TestSettingsManagerUnitTests:
    """Unit tests for SettingsManager edge cases and error handling."""

    def test_load_settings_creates_defaults_when_file_missing(self):
        """Test that loading from non-existent file returns defaults."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=True) as f:
            temp_path = f.name
        
        # File doesn't exist now
        manager = SettingsManager(settings_path=temp_path)
        settings = manager.load_settings()
        
        # Should return default settings
        default = PipelineSettings()
        assert settings.blur_threshold == default.blur_threshold
        assert settings.blur_auto_mode == default.blur_auto_mode
        assert settings.deblur_enabled == default.deblur_enabled

    def test_load_settings_handles_corrupted_file(self):
        """Test that loading corrupted file returns defaults."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("not valid json {{{")
            temp_path = f.name
        
        try:
            manager = SettingsManager(settings_path=temp_path)
            settings = manager.load_settings()
            
            # Should return default settings
            default = PipelineSettings()
            assert settings.blur_threshold == default.blur_threshold
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_reset_to_defaults_saves_default_settings(self):
        """Test that reset_to_defaults saves default settings."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            manager = SettingsManager(settings_path=temp_path)
            
            # Save non-default settings first
            custom_settings = PipelineSettings(
                blur_threshold=500.0,
                gamma_value=2.5,
                deblur_enabled=False
            )
            manager.save_settings(custom_settings)
            
            # Reset to defaults
            reset_settings = manager.reset_to_defaults()
            
            # Verify defaults
            default = PipelineSettings()
            assert reset_settings.blur_threshold == default.blur_threshold
            assert reset_settings.gamma_value == default.gamma_value
            assert reset_settings.deblur_enabled == default.deblur_enabled
            
            # Verify file was updated
            manager2 = SettingsManager(settings_path=temp_path)
            loaded = manager2.load_settings()
            assert loaded.blur_threshold == default.blur_threshold
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_update_blur_settings_partial_update(self):
        """Test that update_blur_settings only updates specified fields."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            manager = SettingsManager(settings_path=temp_path)
            
            # Save initial settings
            initial = PipelineSettings(
                blur_threshold=100.0,
                blur_auto_mode=True,
                deblur_enabled=True
            )
            manager.save_settings(initial)
            
            # Update only threshold
            manager.update_blur_settings(threshold=200.0)
            
            settings = manager.get_current_settings()
            assert settings.blur_threshold == 200.0
            assert settings.blur_auto_mode == True  # Unchanged
            assert settings.deblur_enabled == True  # Unchanged
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_update_illumination_settings_partial_update(self):
        """Test that update_illumination_settings only updates specified fields."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            manager = SettingsManager(settings_path=temp_path)
            
            # Save initial settings
            initial = PipelineSettings(
                low_light_threshold=80,
                gamma_value=1.0,
                illumination_auto_mode=True
            )
            manager.save_settings(initial)
            
            # Update only gamma
            manager.update_illumination_settings(gamma_value=1.5)
            
            settings = manager.get_current_settings()
            assert settings.low_light_threshold == 80  # Unchanged
            assert settings.gamma_value == 1.5
            assert settings.illumination_auto_mode == True  # Unchanged
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_invalid_blur_threshold_raises_error(self):
        """Test that negative blur threshold raises ValueError."""
        settings = PipelineSettings(blur_threshold=-1.0)
        with pytest.raises(ValueError):
            settings.validate()

    def test_invalid_low_light_threshold_raises_error(self):
        """Test that invalid low_light_threshold raises ValueError."""
        settings = PipelineSettings(low_light_threshold=300)
        with pytest.raises(ValueError):
            settings.validate()
        
        settings = PipelineSettings(low_light_threshold=-1)
        with pytest.raises(ValueError):
            settings.validate()

    def test_invalid_gamma_value_raises_error(self):
        """Test that non-positive gamma raises ValueError."""
        settings = PipelineSettings(gamma_value=0)
        with pytest.raises(ValueError):
            settings.validate()
        
        settings = PipelineSettings(gamma_value=-1.0)
        with pytest.raises(ValueError):
            settings.validate()

    def test_invalid_calibration_sample_size_raises_error(self):
        """Test that invalid sample size raises ValueError."""
        settings = PipelineSettings(calibration_sample_size=0)
        with pytest.raises(ValueError):
            settings.validate()

    def test_invalid_percentile_raises_error(self):
        """Test that invalid percentile raises ValueError."""
        settings = PipelineSettings(blur_percentile=-1)
        with pytest.raises(ValueError):
            settings.validate()
        
        settings = PipelineSettings(blur_percentile=101)
        with pytest.raises(ValueError):
            settings.validate()
        
        settings = PipelineSettings(luminance_percentile=-1)
        with pytest.raises(ValueError):
            settings.validate()
        
        settings = PipelineSettings(luminance_percentile=101)
        with pytest.raises(ValueError):
            settings.validate()

    def test_get_current_settings_loads_if_not_loaded(self):
        """Test that get_current_settings loads settings if not already loaded."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            manager = SettingsManager(settings_path=temp_path)
            
            # get_current_settings should load defaults since file doesn't exist
            settings = manager.get_current_settings()
            
            default = PipelineSettings()
            assert settings.blur_threshold == default.blur_threshold
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_update_blur_settings_validates_threshold(self):
        """Test that update_blur_settings validates threshold."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            manager = SettingsManager(settings_path=temp_path)
            manager.save_settings(PipelineSettings())
            
            with pytest.raises(ValueError):
                manager.update_blur_settings(threshold=-1.0)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_update_illumination_settings_validates_values(self):
        """Test that update_illumination_settings validates values."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            manager = SettingsManager(settings_path=temp_path)
            manager.save_settings(PipelineSettings())
            
            with pytest.raises(ValueError):
                manager.update_illumination_settings(low_light_threshold=300)
            
            with pytest.raises(ValueError):
                manager.update_illumination_settings(gamma_value=0)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_settings_path_property(self):
        """Test that settings_path property returns correct path."""
        manager = SettingsManager(settings_path="/custom/path.json")
        assert manager.settings_path == "/custom/path.json"
        
        manager2 = SettingsManager()
        assert manager2.settings_path == SettingsManager.DEFAULT_SETTINGS_PATH

    def test_invalid_ocr_frame_interval_raises_error(self):
        """Test that invalid ocr_frame_interval raises ValueError."""
        # Test interval below minimum (0)
        settings = PipelineSettings(ocr_frame_interval=0)
        with pytest.raises(ValueError):
            settings.validate()
        
        # Test interval above maximum (31)
        settings = PipelineSettings(ocr_frame_interval=31)
        with pytest.raises(ValueError):
            settings.validate()

    def test_update_ocr_settings_validates_frame_interval(self):
        """Test that update_ocr_settings validates frame_interval."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            manager = SettingsManager(settings_path=temp_path)
            manager.save_settings(PipelineSettings())
            
            # Test interval below minimum
            with pytest.raises(ValueError):
                manager.update_ocr_settings(frame_interval=0)
            
            # Test interval above maximum
            with pytest.raises(ValueError):
                manager.update_ocr_settings(frame_interval=31)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_update_ocr_settings_partial_update(self):
        """Test that update_ocr_settings only updates specified fields."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            manager = SettingsManager(settings_path=temp_path)
            
            # Save initial settings
            initial = PipelineSettings(
                ocr_language='en',
                ocr_gpu_enabled=True,
                ocr_frame_interval=5
            )
            manager.save_settings(initial)
            
            # Update only frame_interval
            manager.update_ocr_settings(frame_interval=10)
            
            settings = manager.get_current_settings()
            assert settings.ocr_language == 'en'  # Unchanged
            assert settings.ocr_gpu_enabled == True  # Unchanged
            assert settings.ocr_frame_interval == 10
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_default_ocr_frame_interval_is_5(self):
        """Test that default OCR frame interval is 5."""
        settings = PipelineSettings()
        assert settings.ocr_frame_interval == 5
