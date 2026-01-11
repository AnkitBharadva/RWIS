"""
Property-based tests for dashboard app module.

Feature: streamlit-dashboard
Validates: Requirements 6.1, 6.2, 6.4
"""

import pytest
from hypothesis import given, strategies as st, settings

from dashboard.app import get_damage_indicator_state


class TestVisualFeedbackConsistency:
    """
    Property 5: Visual Feedback Consistency
    
    For any damage detection state:
    - If damage_detected is True, the visual indicator SHALL be red/alert
    - If damage_detected is False, the visual indicator SHALL be green/normal
    - The indicator state SHALL match the current damage_detected value
    
    Validates: Requirements 6.1, 6.2, 6.4
    """

    @given(damage_detected=st.booleans())
    @settings(max_examples=100)
    def test_indicator_color_matches_damage_state(self, damage_detected: bool):
        """
        Feature: streamlit-dashboard, Property 5: Visual Feedback Consistency
        
        Generate random damage_detected states.
        Verify indicator color matches state (True=red, False=green).
        """
        state = get_damage_indicator_state(damage_detected)
        
        if damage_detected:
            assert state["color"] == "red", \
                f"Expected red color when damage detected, got {state['color']}"
        else:
            assert state["color"] == "green", \
                f"Expected green color when no damage, got {state['color']}"

    @given(damage_detected=st.booleans())
    @settings(max_examples=100)
    def test_css_class_matches_damage_state(self, damage_detected: bool):
        """
        Feature: streamlit-dashboard, Property 5: Visual Feedback Consistency
        
        Verify CSS class matches damage detection state.
        """
        state = get_damage_indicator_state(damage_detected)
        
        if damage_detected:
            assert state["css_class"] == "status-alert", \
                f"Expected status-alert class when damage detected, got {state['css_class']}"
        else:
            assert state["css_class"] == "status-normal", \
                f"Expected status-normal class when no damage, got {state['css_class']}"

    @given(damage_detected=st.booleans())
    @settings(max_examples=100)
    def test_text_matches_damage_state(self, damage_detected: bool):
        """
        Feature: streamlit-dashboard, Property 5: Visual Feedback Consistency
        
        Verify status text matches damage detection state.
        """
        state = get_damage_indicator_state(damage_detected)
        
        if damage_detected:
            assert state["text"] == "DAMAGE DETECTED", \
                f"Expected 'DAMAGE DETECTED' text when damage detected, got {state['text']}"
        else:
            assert state["text"] == "NORMAL", \
                f"Expected 'NORMAL' text when no damage, got {state['text']}"

    @given(damage_detected=st.booleans())
    @settings(max_examples=100)
    def test_delta_color_matches_damage_state(self, damage_detected: bool):
        """
        Feature: streamlit-dashboard, Property 5: Visual Feedback Consistency
        
        Verify delta color for st.metric matches damage detection state.
        """
        state = get_damage_indicator_state(damage_detected)
        
        if damage_detected:
            assert state["delta_color"] == "inverse", \
                f"Expected 'inverse' delta_color when damage detected, got {state['delta_color']}"
        else:
            assert state["delta_color"] == "normal", \
                f"Expected 'normal' delta_color when no damage, got {state['delta_color']}"

    @given(damage_detected=st.booleans())
    @settings(max_examples=100)
    def test_all_state_properties_consistent(self, damage_detected: bool):
        """
        Feature: streamlit-dashboard, Property 5: Visual Feedback Consistency
        
        Verify all state properties are internally consistent.
        """
        state = get_damage_indicator_state(damage_detected)
        
        # All properties should indicate the same state
        is_alert = (
            state["color"] == "red" and
            state["css_class"] == "status-alert" and
            state["text"] == "DAMAGE DETECTED" and
            state["delta_color"] == "inverse"
        )
        
        is_normal = (
            state["color"] == "green" and
            state["css_class"] == "status-normal" and
            state["text"] == "NORMAL" and
            state["delta_color"] == "normal"
        )
        
        # State should be either all alert or all normal
        assert is_alert or is_normal, \
            f"State properties are inconsistent: {state}"
        
        # And it should match the damage_detected flag
        if damage_detected:
            assert is_alert, \
                f"Expected alert state when damage_detected=True, got {state}"
        else:
            assert is_normal, \
                f"Expected normal state when damage_detected=False, got {state}"


class TestUIUpdateIsolation:
    """
    Property 6: UI Update Isolation
    
    For any frame update cycle, only the designated placeholder containers
    (metrics, video, log) SHALL be updated. The sidebar and page header
    SHALL NOT re-render during the update loop.
    
    Validates: Requirements 3.6, 4.4, 7.4
    """

    @given(
        fps=st.floats(min_value=0.0, max_value=120.0, allow_nan=False, allow_infinity=False),
        inference_ms=st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
        object_count=st.integers(min_value=0, max_value=100),
        damage_count=st.integers(min_value=0, max_value=50),
        damage_detected=st.booleans()
    )
    @settings(max_examples=100)
    def test_metrics_update_uses_placeholder_pattern(
        self,
        fps: float,
        inference_ms: float,
        object_count: int,
        damage_count: int,
        damage_detected: bool
    ):
        """
        Feature: streamlit-dashboard, Property 6: UI Update Isolation
        
        Verify that render_metrics_row accepts a placeholder parameter,
        enabling isolated updates without full page re-render.
        
        The placeholder pattern ensures:
        - Only the metrics container is updated
        - Sidebar remains unchanged
        - Header remains unchanged
        """
        from unittest.mock import MagicMock, patch
        from dashboard.app import MissionControlDashboard
        
        # Create a mock placeholder that tracks calls
        mock_placeholder = MagicMock()
        mock_container = MagicMock()
        mock_placeholder.container.return_value.__enter__ = MagicMock(return_value=mock_container)
        mock_placeholder.container.return_value.__exit__ = MagicMock(return_value=False)
        
        # Patch streamlit to avoid actual UI rendering
        with patch('dashboard.app.st') as mock_st:
            mock_st.columns.return_value = [MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock()]
            mock_st.metric = MagicMock()
            mock_st.markdown = MagicMock()
            mock_st.session_state = MagicMock()
            mock_st.session_state.get = MagicMock(return_value=True)
            
            # Create dashboard instance (without page config)
            with patch.object(MissionControlDashboard, '__init__', lambda x: None):
                dashboard = MissionControlDashboard()
                dashboard.video_manager = MagicMock()
                dashboard.metrics_calculator = MagicMock()
                
                # Call render_metrics_row with the mock placeholder
                dashboard.render_metrics_row(
                    mock_placeholder,
                    fps=fps,
                    inference_ms=inference_ms,
                    object_count=object_count,
                    damage_count=damage_count,
                    damage_detected=damage_detected
                )
                
                # Verify placeholder.container() was called (isolated update pattern)
                mock_placeholder.container.assert_called_once()

    @given(frame_data=st.binary(min_size=100, max_size=1000))
    @settings(max_examples=100)
    def test_video_feed_update_uses_placeholder_pattern(self, frame_data: bytes):
        """
        Feature: streamlit-dashboard, Property 6: UI Update Isolation
        
        Verify that render_video_feed accepts a placeholder parameter,
        enabling isolated updates without full page re-render.
        """
        from unittest.mock import MagicMock, patch
        from dashboard.app import MissionControlDashboard
        import numpy as np
        
        # Create a mock placeholder
        mock_placeholder = MagicMock()
        mock_placeholder.container.return_value.__enter__ = MagicMock()
        mock_placeholder.container.return_value.__exit__ = MagicMock(return_value=False)
        
        # Create a simple test frame (3-channel BGR image)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Patch streamlit and cv2 (cv2 is imported inside the method)
        with patch('dashboard.app.st') as mock_st, \
             patch('cv2.cvtColor') as mock_cvtColor:
            mock_st.columns.return_value = [MagicMock(), MagicMock(), MagicMock()]
            mock_st.image = MagicMock()
            mock_cvtColor.return_value = frame
            
            # Create dashboard instance
            with patch.object(MissionControlDashboard, '__init__', lambda x: None):
                dashboard = MissionControlDashboard()
                dashboard.video_manager = MagicMock()
                dashboard.metrics_calculator = MagicMock()
                
                # Call render_video_feed with the mock placeholder
                dashboard.render_video_feed(mock_placeholder, frame)
                
                # Verify placeholder.container() was called (isolated update pattern)
                mock_placeholder.container.assert_called_once()

    @given(num_entries=st.integers(min_value=0, max_value=20))
    @settings(max_examples=100)
    def test_detection_log_update_uses_placeholder_pattern(self, num_entries: int):
        """
        Feature: streamlit-dashboard, Property 6: UI Update Isolation
        
        Verify that render_detection_log accepts a placeholder parameter,
        enabling isolated updates without full page re-render.
        """
        from unittest.mock import MagicMock, patch
        from dashboard.app import MissionControlDashboard
        from dashboard.models import DetectionLogEntry
        from datetime import datetime
        
        # Create mock detection log entries
        detection_log = [
            DetectionLogEntry(
                timestamp=datetime.now(),
                wagon_id=i,
                damage_type="crack",
                confidence=0.85,
                frame_index=i * 10
            )
            for i in range(num_entries)
        ]
        
        # Create a mock placeholder
        mock_placeholder = MagicMock()
        mock_placeholder.container.return_value.__enter__ = MagicMock()
        mock_placeholder.container.return_value.__exit__ = MagicMock(return_value=False)
        
        # Patch streamlit to avoid actual rendering
        with patch('dashboard.app.st') as mock_st:
            mock_st.expander.return_value.__enter__ = MagicMock()
            mock_st.expander.return_value.__exit__ = MagicMock(return_value=False)
            mock_st.dataframe = MagicMock()
            mock_st.markdown = MagicMock()
            
            # Create dashboard instance
            with patch.object(MissionControlDashboard, '__init__', lambda x: None):
                dashboard = MissionControlDashboard()
                dashboard.video_manager = MagicMock()
                dashboard.metrics_calculator = MagicMock()
                
                # Call render_detection_log with the mock placeholder
                dashboard.render_detection_log(mock_placeholder, detection_log)
                
                # Verify placeholder.container() was called (isolated update pattern)
                mock_placeholder.container.assert_called_once()

    @given(
        is_running=st.booleans(),
        has_frame=st.booleans()
    )
    @settings(max_examples=100)
    def test_run_method_creates_placeholders_for_isolation(
        self,
        is_running: bool,
        has_frame: bool
    ):
        """
        Feature: streamlit-dashboard, Property 6: UI Update Isolation
        
        Verify that the run() method creates st.empty() placeholders
        for metrics, video, and log to enable isolated updates.
        
        This ensures the sidebar and header are not re-rendered
        during the processing loop.
        """
        from unittest.mock import MagicMock, patch, call
        from dashboard.app import MissionControlDashboard
        import numpy as np
        
        # Track st.empty() calls
        empty_calls = []
        
        def mock_empty():
            mock = MagicMock()
            empty_calls.append(mock)
            return mock
        
        # Patch streamlit
        with patch('dashboard.app.st') as mock_st:
            mock_st.empty = mock_empty
            mock_st.title = MagicMock()
            mock_st.session_state = {
                "is_running": False,  # Don't run the loop
                "detection_log": [],
                "last_frame": None,
                "video_source": "",
                "confidence_threshold": 0.5,
                "frame_skip": 3,
                "connection_status": "Disconnected"
            }
            mock_st.sidebar.__enter__ = MagicMock()
            mock_st.sidebar.__exit__ = MagicMock(return_value=False)
            mock_st.columns.return_value = [MagicMock(), MagicMock(), MagicMock(), MagicMock()]
            mock_st.text_input.return_value = ""
            mock_st.slider.return_value = 0.5
            mock_st.number_input.return_value = 3
            mock_st.button.return_value = False
            mock_st.divider = MagicMock()
            mock_st.header = MagicMock()
            mock_st.subheader = MagicMock()
            mock_st.info = MagicMock()
            mock_st.success = MagicMock()
            mock_st.warning = MagicMock()
            mock_st.error = MagicMock()
            mock_st.metric = MagicMock()
            mock_st.markdown = MagicMock()
            mock_st.image = MagicMock()
            mock_st.expander.return_value.__enter__ = MagicMock()
            mock_st.expander.return_value.__exit__ = MagicMock(return_value=False)
            mock_st.dataframe = MagicMock()
            
            # Create dashboard instance
            with patch.object(MissionControlDashboard, '__init__', lambda x: None):
                dashboard = MissionControlDashboard()
                dashboard.video_manager = MagicMock()
                dashboard.video_manager.is_connected.return_value = False
                dashboard.metrics_calculator = MagicMock()
                dashboard.metrics_calculator.get_fps.return_value = 0.0
                dashboard.metrics_calculator.get_inference_ms.return_value = 0.0
                
                # Mock the render methods to avoid complex setup
                dashboard.render_sidebar = MagicMock(return_value=MagicMock())
                dashboard.render_metrics_row = MagicMock()
                dashboard.render_video_feed = MagicMock()
                dashboard.render_detection_log = MagicMock()
                dashboard._init_session_state = MagicMock()
                
                # Mock new dashboard enhancement components
                dashboard.dual_display = MagicMock()
                dashboard.track_renderer = MagicMock()
                dashboard.frame_saver = MagicMock()
                dashboard.frame_saver.update_config = MagicMock()
                dashboard.ocr_log_display = MagicMock()
                dashboard.metrics_calculator.get_smoothed_latency = MagicMock(return_value=0.0)
                dashboard.metrics_calculator.is_latency_warning = MagicMock(return_value=False)
                
                # Call run()
                dashboard.run()
                
                # Verify st.empty() was called 5 times (metrics, video, log, ocr_log, stop button placeholders)
                assert len(empty_calls) == 5, \
                    f"Expected 5 st.empty() calls for placeholders, got {len(empty_calls)}"


class TestConfigurationPersistence:
    """
    Property 7: Configuration Persistence
    
    For any configuration change in the sidebar:
    - The configuration SHALL persist in session state across reruns
    - Reloading the dashboard SHALL restore the previous configuration values
    
    Validates: Requirements 8.5
    """

    @given(
        frame_save_enabled=st.booleans(),
        save_on_deblur=st.booleans(),
        save_on_illumination=st.booleans(),
        save_on_ocr=st.booleans(),
        output_directory=st.text(
            alphabet=st.characters(whitelist_categories=('L', 'N'), whitelist_characters='/_-'),
            min_size=1,
            max_size=50
        ).filter(lambda x: len(x.strip()) > 0)
    )
    @settings(max_examples=100)
    def test_frame_save_config_persists_in_session_state(
        self,
        frame_save_enabled: bool,
        save_on_deblur: bool,
        save_on_illumination: bool,
        save_on_ocr: bool,
        output_directory: str
    ):
        """
        Feature: dashboard-enhancements, Property 7: Configuration Persistence
        
        Generate random frame save configuration values.
        Verify configuration persists in session state across simulated reruns.
        
        **Validates: Requirements 8.5**
        """
        from unittest.mock import MagicMock, patch
        import streamlit
        from dashboard.models import FrameSaveConfig
        
        # Create a mock session state that persists values
        mock_session_state = MagicMock()
        mock_session_state.get = MagicMock(side_effect=lambda key, default=None: {
            "frame_save_enabled": frame_save_enabled,
            "frame_save_on_deblur": save_on_deblur,
            "frame_save_on_illumination": save_on_illumination,
            "frame_save_on_ocr": save_on_ocr,
            "frame_save_output_directory": output_directory
        }.get(key, default))
        
        # Patch streamlit.session_state at the module level where it's imported
        with patch.object(streamlit, 'session_state', mock_session_state):
            from dashboard.app import get_frame_save_config_from_session
            
            # Get config from session state
            config = get_frame_save_config_from_session()
            
            # Verify all values match what was stored in session state
            assert config.enabled == frame_save_enabled, \
                f"Expected enabled={frame_save_enabled}, got {config.enabled}"
            assert config.save_on_deblur == save_on_deblur, \
                f"Expected save_on_deblur={save_on_deblur}, got {config.save_on_deblur}"
            assert config.save_on_illumination == save_on_illumination, \
                f"Expected save_on_illumination={save_on_illumination}, got {config.save_on_illumination}"
            assert config.save_on_ocr == save_on_ocr, \
                f"Expected save_on_ocr={save_on_ocr}, got {config.save_on_ocr}"
            assert config.output_directory == output_directory, \
                f"Expected output_directory={output_directory}, got {config.output_directory}"

    @given(
        frame_save_enabled=st.booleans(),
        save_on_deblur=st.booleans(),
        save_on_illumination=st.booleans(),
        save_on_ocr=st.booleans()
    )
    @settings(max_examples=100)
    def test_config_values_survive_simulated_rerun(
        self,
        frame_save_enabled: bool,
        save_on_deblur: bool,
        save_on_illumination: bool,
        save_on_ocr: bool
    ):
        """
        Feature: dashboard-enhancements, Property 7: Configuration Persistence
        
        Simulate multiple dashboard reruns and verify configuration persists.
        
        **Validates: Requirements 8.5**
        """
        from unittest.mock import MagicMock, patch
        import streamlit
        
        # Simulate session state that persists across reruns
        persistent_state = {
            "frame_save_enabled": frame_save_enabled,
            "frame_save_on_deblur": save_on_deblur,
            "frame_save_on_illumination": save_on_illumination,
            "frame_save_on_ocr": save_on_ocr,
            "frame_save_output_directory": "outputs/saved_frames"
        }
        
        mock_session_state = MagicMock()
        mock_session_state.get = MagicMock(side_effect=lambda key, default=None: persistent_state.get(key, default))
        
        # Simulate "rerun" - values should still be there
        with patch.object(streamlit, 'session_state', mock_session_state):
            from dashboard.app import get_frame_save_config_from_session
            
            # Get config after simulated rerun
            config = get_frame_save_config_from_session()
            
            # Verify values persisted
            assert config.enabled == frame_save_enabled
            assert config.save_on_deblur == save_on_deblur
            assert config.save_on_illumination == save_on_illumination
            assert config.save_on_ocr == save_on_ocr

    @given(
        initial_enabled=st.booleans(),
        updated_enabled=st.booleans()
    )
    @settings(max_examples=100)
    def test_config_update_persists_new_values(
        self,
        initial_enabled: bool,
        updated_enabled: bool
    ):
        """
        Feature: dashboard-enhancements, Property 7: Configuration Persistence
        
        Verify that updating configuration values persists the new values.
        
        **Validates: Requirements 8.5**
        """
        from unittest.mock import MagicMock, patch
        import streamlit
        
        # Start with initial value
        persistent_state = {
            "frame_save_enabled": initial_enabled,
            "frame_save_on_deblur": True,
            "frame_save_on_illumination": True,
            "frame_save_on_ocr": True,
            "frame_save_output_directory": "outputs/saved_frames"
        }
        
        mock_session_state = MagicMock()
        mock_session_state.get = MagicMock(side_effect=lambda key, default=None: persistent_state.get(key, default))
        
        with patch.object(streamlit, 'session_state', mock_session_state):
            from dashboard.app import get_frame_save_config_from_session
            
            # Verify initial value
            config1 = get_frame_save_config_from_session()
            assert config1.enabled == initial_enabled
            
            # Update the value (simulating user toggle)
            persistent_state["frame_save_enabled"] = updated_enabled
            
            # Verify updated value persists
            config2 = get_frame_save_config_from_session()
            assert config2.enabled == updated_enabled, \
                f"Expected enabled={updated_enabled} after update, got {config2.enabled}"

    @given(
        enabled=st.booleans(),
        deblur=st.booleans(),
        illumination=st.booleans(),
        ocr=st.booleans()
    )
    @settings(max_examples=100)
    def test_frame_save_config_dataclass_matches_session_state(
        self,
        enabled: bool,
        deblur: bool,
        illumination: bool,
        ocr: bool
    ):
        """
        Feature: dashboard-enhancements, Property 7: Configuration Persistence
        
        Verify FrameSaveConfig dataclass correctly represents session state values.
        
        **Validates: Requirements 8.5**
        """
        from dashboard.models import FrameSaveConfig
        
        # Create config directly
        config = FrameSaveConfig(
            enabled=enabled,
            save_on_deblur=deblur,
            save_on_illumination=illumination,
            save_on_ocr=ocr,
            output_directory="test/path"
        )
        
        # Verify all fields are correctly set
        assert config.enabled == enabled
        assert config.save_on_deblur == deblur
        assert config.save_on_illumination == illumination
        assert config.save_on_ocr == ocr
        assert config.output_directory == "test/path"
        
        # Verify config is a valid FrameSaveConfig instance
        assert isinstance(config, FrameSaveConfig)
