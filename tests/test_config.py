"""
Property-based tests for configuration loading.

Feature: railway-wagon-inspection, Property 15: Configuration Loading
Validates: Requirements 7.1
"""

import os
import tempfile
import pytest
from hypothesis import given, strategies as st, settings

from config import PipelineConfig, load_config


# Strategy for generating valid blur thresholds where t1 < t2
@st.composite
def valid_blur_thresholds(draw):
    t1 = draw(st.floats(min_value=0.0, max_value=500.0, allow_nan=False, allow_infinity=False))
    t2 = draw(st.floats(min_value=t1 + 0.1, max_value=1000.0, allow_nan=False, allow_infinity=False))
    return t1, t2


# Strategy for generating valid confidence thresholds
valid_confidence = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)

# Strategy for generating valid counting line position
valid_line_position = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)

# Strategy for generating valid batch sizes
valid_batch_size = st.integers(min_value=1, max_value=32)


@st.composite
def valid_config_dict(draw):
    """Generate a valid configuration dictionary."""
    t1, t2 = draw(valid_blur_thresholds())
    
    return {
        "video_source": draw(st.text(min_size=0, max_size=100)),
        "blur_threshold_t1": t1,
        "blur_threshold_t2": t2,
        "wagon_model_path": draw(st.text(min_size=1, max_size=100)),
        "damage_model_path": draw(st.text(min_size=1, max_size=100)),
        "nafnet_model_path": draw(st.text(min_size=1, max_size=100)),
        "wagon_confidence_threshold": draw(valid_confidence),
        "damage_confidence_threshold": draw(valid_confidence),
        "counting_line_position": draw(valid_line_position),
        "ocr_gpu_enabled": draw(st.booleans()),
        "ocr_language": draw(st.sampled_from(["en", "ch", "fr", "de"])),
        "max_batch_size": draw(valid_batch_size),
        "enable_threading": draw(st.booleans()),
        "output_dir": draw(st.text(min_size=1, max_size=50)),
        "enable_debug_frames": draw(st.booleans()),
        "log_format": draw(st.lists(st.sampled_from(["csv", "json"]), min_size=1, max_size=2, unique=True)),
    }


class TestConfigurationLoading:
    """
    Property 15: Configuration Loading
    
    For any configuration file, all threshold values (T1, T2, confidence thresholds,
    counting line position) SHALL be loaded and applied to the pipeline.
    Changing config values SHALL change pipeline behavior accordingly.
    
    Validates: Requirements 7.1
    """

    @given(config_data=valid_config_dict())
    @settings(max_examples=100)
    def test_all_config_values_loaded_correctly(self, config_data):
        """
        Feature: railway-wagon-inspection, Property 15: Configuration Loading
        
        Generate random valid configurations and verify all values are loaded correctly.
        """
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            import json
            json.dump(config_data, f)
            temp_path = f.name
        
        try:
            config = load_config(temp_path)
            
            # Verify all values are loaded correctly
            assert config.video_source == config_data["video_source"]
            assert config.blur_threshold_t1 == config_data["blur_threshold_t1"]
            assert config.blur_threshold_t2 == config_data["blur_threshold_t2"]
            assert config.wagon_model_path == config_data["wagon_model_path"]
            assert config.damage_model_path == config_data["damage_model_path"]
            assert config.nafnet_model_path == config_data["nafnet_model_path"]
            assert config.wagon_confidence_threshold == config_data["wagon_confidence_threshold"]
            assert config.damage_confidence_threshold == config_data["damage_confidence_threshold"]
            assert config.counting_line_position == config_data["counting_line_position"]
            assert config.ocr_gpu_enabled == config_data["ocr_gpu_enabled"]
            assert config.ocr_language == config_data["ocr_language"]
            assert config.max_batch_size == config_data["max_batch_size"]
            assert config.enable_threading == config_data["enable_threading"]
            assert config.output_dir == config_data["output_dir"]
            assert config.enable_debug_frames == config_data["enable_debug_frames"]
            assert config.log_format == config_data["log_format"]
        finally:
            os.unlink(temp_path)

    @given(config_data=valid_config_dict())
    @settings(max_examples=100)
    def test_config_round_trip(self, config_data):
        """
        Feature: railway-wagon-inspection, Property 15: Configuration Loading
        
        Verify that saving and loading a config produces equivalent values.
        """
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            import json
            json.dump(config_data, f)
            temp_path = f.name
        
        try:
            # Load config
            config = load_config(temp_path)
            
            # Save to new file
            save_path = temp_path + ".saved.json"
            config.save(save_path)
            
            # Load again
            config2 = load_config(save_path)
            
            # Verify all values match
            assert config.to_dict() == config2.to_dict()
            
            os.unlink(save_path)
        finally:
            os.unlink(temp_path)
