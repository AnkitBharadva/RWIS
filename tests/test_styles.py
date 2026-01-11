"""
Unit tests for CSS injection module.

Tests the styles.py module to ensure CSS contains required selectors
and inject_css() function works correctly.

Requirements: 8.1
"""

import sys
import pytest
from unittest.mock import MagicMock

# Mock streamlit before importing any dashboard modules
mock_st = MagicMock()
sys.modules['streamlit'] = mock_st

# Now we can safely import from dashboard.styles
# Import directly to avoid dashboard/__init__.py importing other modules
import importlib.util
spec = importlib.util.spec_from_file_location("styles", "dashboard/styles.py")
styles_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(styles_module)

CUSTOM_CSS = styles_module.CUSTOM_CSS
inject_css = styles_module.inject_css


class TestCustomCSS:
    """Tests for the CUSTOM_CSS constant."""

    def test_css_contains_block_container_padding(self):
        """Test CSS contains reduced top padding for block-container."""
        assert ".block-container" in CUSTOM_CSS
        assert "padding-top: 1rem" in CUSTOM_CSS

    def test_css_contains_metric_container_background(self):
        """Test CSS contains dark grey background for metric containers."""
        assert 'div[data-testid="metric-container"]' in CUSTOM_CSS
        assert "#1E1E1E" in CUSTOM_CSS

    def test_css_contains_metric_container_border(self):
        """Test CSS contains subtle border for metric cards."""
        assert "border: 1px solid #3D3D3D" in CUSTOM_CSS

    def test_css_contains_large_bold_metric_values(self):
        """Test CSS contains large bold font styling for metric values."""
        assert "font-size: 2rem" in CUSTOM_CSS
        assert "font-weight: 700" in CUSTOM_CSS

    def test_css_contains_status_normal_class(self):
        """Test CSS contains green status-normal class."""
        assert ".status-normal" in CUSTOM_CSS
        assert "#00FF00" in CUSTOM_CSS

    def test_css_contains_status_alert_class(self):
        """Test CSS contains red status-alert class."""
        assert ".status-alert" in CUSTOM_CSS
        assert "#FF4444" in CUSTOM_CSS

    def test_css_is_wrapped_in_style_tags(self):
        """Test CSS is properly wrapped in style tags."""
        assert "<style>" in CUSTOM_CSS
        assert "</style>" in CUSTOM_CSS


class TestInjectCSS:
    """Tests for the inject_css() function."""

    def test_inject_css_calls_markdown(self):
        """Test inject_css() calls st.markdown with correct parameters."""
        # Reset the mock before test
        mock_st.reset_mock()
        
        inject_css()
        
        mock_st.markdown.assert_called_once_with(CUSTOM_CSS, unsafe_allow_html=True)

    def test_inject_css_does_not_raise_exception(self):
        """Test inject_css() does not raise any exceptions."""
        mock_st.reset_mock()
        mock_st.markdown = MagicMock()
        
        # Should not raise any exception
        try:
            inject_css()
        except Exception as e:
            pytest.fail(f"inject_css() raised an exception: {e}")

    def test_inject_css_enables_unsafe_html(self):
        """Test inject_css() enables unsafe_allow_html for CSS injection."""
        mock_st.reset_mock()
        
        inject_css()
        
        # Verify unsafe_allow_html=True was passed
        call_args = mock_st.markdown.call_args
        assert call_args[1].get('unsafe_allow_html') is True
