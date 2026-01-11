"""
Property-based tests for processing status indicators module.

Feature: ocr-visual-enhancements
Property 10: Processing Indicators Reflect Current State
Validates: Requirements 7.2, 7.3, 7.4
"""

import pytest
from hypothesis import given, strategies as st, settings

from dashboard.processing_indicators import ProcessingStatusIndicator, ProcessingStatus


class TestProcessingIndicatorsReflectCurrentState:
    """
    Property 10: Processing Indicators Reflect Current State
    
    For any processed frame:
    - The illumination indicator SHALL show "APPLIED" if CLAHE/gamma was applied,
      "NORMAL" if not needed, or "OFF" if disabled.
    - The deblur indicator SHALL show "APPLIED" if deblurring was applied,
      "SKIPPED" if not needed, or "OFF" if disabled.
    
    Validates: Requirements 7.2, 7.3, 7.4
    """

    @given(
        enabled=st.booleans(),
        applied=st.booleans(),
        error=st.booleans()
    )
    @settings(max_examples=100)
    def test_illumination_status_reflects_state(self, enabled, applied, error):
        """
        Feature: ocr-visual-enhancements, Property 10: Processing Indicators Reflect Current State
        
        Verify illumination indicator shows correct status based on state.
        """
        indicator = ProcessingStatusIndicator()
        status = indicator.get_illumination_status(enabled, applied, error)
        
        if error:
            assert status == "ERROR", \
                f"Expected ERROR when error=True, got {status}"
        elif not enabled:
            assert status == "OFF", \
                f"Expected OFF when enabled=False, got {status}"
        elif applied:
            assert status == "APPLIED", \
                f"Expected APPLIED when enabled=True and applied=True, got {status}"
        else:
            assert status == "NORMAL", \
                f"Expected NORMAL when enabled=True and applied=False, got {status}"

    @given(
        enabled=st.booleans(),
        applied=st.booleans(),
        skipped_sharp=st.booleans(),
        error=st.booleans()
    )
    @settings(max_examples=100)
    def test_deblur_status_reflects_state(self, enabled, applied, skipped_sharp, error):
        """
        Feature: ocr-visual-enhancements, Property 10: Processing Indicators Reflect Current State
        
        Verify deblur indicator shows correct status based on state.
        """
        indicator = ProcessingStatusIndicator()
        status = indicator.get_deblur_status(enabled, applied, skipped_sharp, error)
        
        if error:
            assert status == "ERROR", \
                f"Expected ERROR when error=True, got {status}"
        elif not enabled:
            assert status == "OFF", \
                f"Expected OFF when enabled=False, got {status}"
        elif applied:
            assert status == "APPLIED", \
                f"Expected APPLIED when enabled=True and applied=True, got {status}"
        elif skipped_sharp:
            assert status == "SKIPPED", \
                f"Expected SKIPPED when enabled=True, applied=False, skipped_sharp=True, got {status}"
        else:
            assert status == "NORMAL", \
                f"Expected NORMAL when enabled=True, applied=False, skipped_sharp=False, got {status}"

    @given(
        enabled=st.booleans(),
        active=st.booleans(),
        skipped_interval=st.booleans(),
        error=st.booleans()
    )
    @settings(max_examples=100)
    def test_ocr_status_reflects_state(self, enabled, active, skipped_interval, error):
        """
        Feature: ocr-visual-enhancements, Property 10: Processing Indicators Reflect Current State
        
        Verify OCR indicator shows correct status based on state.
        """
        indicator = ProcessingStatusIndicator()
        status = indicator.get_ocr_status(enabled, active, skipped_interval, error)
        
        if error:
            assert status == "ERROR", \
                f"Expected ERROR when error=True, got {status}"
        elif not enabled:
            assert status == "OFF", \
                f"Expected OFF when enabled=False, got {status}"
        elif active:
            assert status == "ACTIVE", \
                f"Expected ACTIVE when enabled=True and active=True, got {status}"
        elif skipped_interval:
            assert status == "SKIPPED", \
                f"Expected SKIPPED when enabled=True, active=False, skipped_interval=True, got {status}"
        else:
            assert status == "NORMAL", \
                f"Expected NORMAL when enabled=True, active=False, skipped_interval=False, got {status}"


class TestStatusColorMapping:
    """
    Tests for status color mapping consistency.
    
    Validates: Requirements 7.2, 7.3, 7.5
    """

    @given(status=st.sampled_from(["APPLIED", "ACTIVE", "NORMAL", "SKIPPED", "OFF", "ERROR"]))
    @settings(max_examples=100)
    def test_all_statuses_have_colors(self, status):
        """
        Feature: ocr-visual-enhancements, Property 10: Processing Indicators Reflect Current State
        
        Verify all status values have defined colors.
        """
        indicator = ProcessingStatusIndicator()
        color = indicator.get_status_color(status)
        
        assert color is not None, f"Status {status} should have a color"
        assert color.startswith("#"), f"Color should be hex format, got {color}"
        assert len(color) == 7, f"Color should be 7 chars (#RRGGBB), got {color}"

    @given(status=st.sampled_from(["APPLIED", "ACTIVE", "NORMAL", "SKIPPED", "OFF", "ERROR"]))
    @settings(max_examples=100)
    def test_all_statuses_have_text_colors(self, status):
        """
        Feature: ocr-visual-enhancements, Property 10: Processing Indicators Reflect Current State
        
        Verify all status values have defined text colors.
        """
        indicator = ProcessingStatusIndicator()
        color = indicator.get_text_color(status)
        
        assert color is not None, f"Status {status} should have a text color"
        assert color.startswith("#"), f"Text color should be hex format, got {color}"

    def test_applied_and_active_are_green(self):
        """Verify APPLIED and ACTIVE statuses use green color."""
        indicator = ProcessingStatusIndicator()
        
        applied_color = indicator.get_status_color("APPLIED")
        active_color = indicator.get_status_color("ACTIVE")
        
        assert applied_color == "#28a745", f"APPLIED should be green, got {applied_color}"
        assert active_color == "#28a745", f"ACTIVE should be green, got {active_color}"

    def test_normal_is_gray(self):
        """Verify NORMAL status uses gray color."""
        indicator = ProcessingStatusIndicator()
        color = indicator.get_status_color("NORMAL")
        assert color == "#6c757d", f"NORMAL should be gray, got {color}"

    def test_skipped_is_yellow(self):
        """Verify SKIPPED status uses yellow color."""
        indicator = ProcessingStatusIndicator()
        color = indicator.get_status_color("SKIPPED")
        assert color == "#ffc107", f"SKIPPED should be yellow, got {color}"

    def test_off_and_error_are_red(self):
        """Verify OFF and ERROR statuses use red color."""
        indicator = ProcessingStatusIndicator()
        
        off_color = indicator.get_status_color("OFF")
        error_color = indicator.get_status_color("ERROR")
        
        assert off_color == "#dc3545", f"OFF should be red, got {off_color}"
        assert error_color == "#dc3545", f"ERROR should be red, got {error_color}"


class TestIndicatorHTMLGeneration:
    """
    Tests for HTML indicator generation.
    
    Validates: Requirements 7.5
    """

    @given(
        label=st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=('L', 'N'))),
        status=st.sampled_from(["APPLIED", "ACTIVE", "NORMAL", "SKIPPED", "OFF", "ERROR"])
    )
    @settings(max_examples=100)
    def test_indicator_html_contains_label_and_status(self, label, status):
        """
        Feature: ocr-visual-enhancements, Property 10: Processing Indicators Reflect Current State
        
        Verify generated HTML contains the label and status text.
        """
        indicator = ProcessingStatusIndicator()
        html = indicator.render_indicator_html(label, status)
        
        assert label in html, f"HTML should contain label '{label}'"
        assert status.upper() in html, f"HTML should contain status '{status.upper()}'"

    @given(status=st.sampled_from(["APPLIED", "ACTIVE", "NORMAL", "SKIPPED", "OFF", "ERROR"]))
    @settings(max_examples=100)
    def test_indicator_html_contains_correct_color(self, status):
        """
        Feature: ocr-visual-enhancements, Property 10: Processing Indicators Reflect Current State
        
        Verify generated HTML contains the correct background color.
        """
        indicator = ProcessingStatusIndicator()
        html = indicator.render_indicator_html("Test", status)
        
        expected_color = indicator.get_status_color(status)
        assert expected_color in html, \
            f"HTML should contain color {expected_color} for status {status}"

    def test_indicator_html_with_tooltip(self):
        """Verify tooltip is included in HTML when provided."""
        indicator = ProcessingStatusIndicator()
        tooltip = "Test tooltip text"
        html = indicator.render_indicator_html("Test", "APPLIED", tooltip=tooltip)
        
        assert f'title="{tooltip}"' in html, "HTML should contain tooltip attribute"

    def test_indicator_html_without_tooltip(self):
        """Verify no tooltip attribute when not provided."""
        indicator = ProcessingStatusIndicator()
        html = indicator.render_indicator_html("Test", "APPLIED", tooltip=None)
        
        assert 'title="' not in html, "HTML should not contain tooltip attribute when None"


class TestMetricsRowHTMLGeneration:
    """
    Tests for metrics row HTML generation.
    
    Validates: Requirements 8.1, 8.2
    """

    @given(
        fps=st.floats(min_value=0, max_value=120, allow_nan=False, allow_infinity=False),
        latency_ms=st.floats(min_value=0, max_value=1000, allow_nan=False, allow_infinity=False),
        object_count=st.integers(min_value=0, max_value=100),
        wagon_count=st.integers(min_value=0, max_value=1000),
        damage_count=st.integers(min_value=0, max_value=50)
    )
    @settings(max_examples=100)
    def test_metrics_row_contains_all_values(self, fps, latency_ms, object_count, wagon_count, damage_count):
        """
        Feature: ocr-visual-enhancements, Property 10: Processing Indicators Reflect Current State
        
        Verify metrics row HTML contains all metric values.
        """
        indicator = ProcessingStatusIndicator()
        html = indicator.render_metrics_row_html(
            fps=fps,
            latency_ms=latency_ms,
            object_count=object_count,
            wagon_count=wagon_count,
            damage_count=damage_count,
            illumination_status="NORMAL",
            deblur_status="NORMAL",
            ocr_status="NORMAL"
        )
        
        # Check that FPS value is present (formatted to 1 decimal)
        assert f"{fps:.1f}" in html, f"HTML should contain FPS value {fps:.1f}"
        
        # Check that latency value is present
        assert f"{latency_ms:.1f}ms" in html, f"HTML should contain latency value {latency_ms:.1f}ms"
        
        # Check that counts are present
        assert f">{object_count}<" in html, f"HTML should contain object count {object_count}"
        assert f">{wagon_count}<" in html, f"HTML should contain wagon count {wagon_count}"
        assert f">{damage_count}<" in html, f"HTML should contain damage count {damage_count}"

    def test_metrics_row_order(self):
        """
        Verify metrics row displays in correct order:
        FPS | Latency | Objects | Wagons | Damage | Illumination | Deblur | OCR
        """
        indicator = ProcessingStatusIndicator()
        html = indicator.render_metrics_row_html(
            fps=30.0,
            latency_ms=50.0,
            object_count=5,
            wagon_count=10,
            damage_count=2,
            illumination_status="APPLIED",
            deblur_status="SKIPPED",
            ocr_status="ACTIVE"
        )
        
        # Find positions of each element
        fps_pos = html.find("FPS")
        latency_pos = html.find("Latency")
        objects_pos = html.find("Objects")
        wagons_pos = html.find("Wagons")
        damage_pos = html.find("Damage")
        illum_pos = html.find("Illum")
        deblur_pos = html.find("Deblur")
        ocr_pos = html.find("OCR")
        
        # Verify order
        assert fps_pos < latency_pos < objects_pos < wagons_pos < damage_pos < illum_pos < deblur_pos < ocr_pos, \
            "Metrics should be in order: FPS | Latency | Objects | Wagons | Damage | Illumination | Deblur | OCR"

    def test_metrics_row_latency_warning(self):
        """Verify latency warning indicator appears when latency_warning is True."""
        indicator = ProcessingStatusIndicator()
        
        # With warning
        html_warning = indicator.render_metrics_row_html(
            fps=30.0,
            latency_ms=150.0,
            object_count=5,
            wagon_count=10,
            damage_count=0,
            illumination_status="NORMAL",
            deblur_status="NORMAL",
            ocr_status="NORMAL",
            latency_warning=True
        )
        assert "HIGH" in html_warning, "Should show HIGH warning when latency_warning=True"
        assert "#ff4b4b" in html_warning, "Should use red color for latency warning"
        
        # Without warning
        html_normal = indicator.render_metrics_row_html(
            fps=30.0,
            latency_ms=50.0,
            object_count=5,
            wagon_count=10,
            damage_count=0,
            illumination_status="NORMAL",
            deblur_status="NORMAL",
            ocr_status="NORMAL",
            latency_warning=False
        )
        assert "HIGH" not in html_normal, "Should not show HIGH warning when latency_warning=False"

    def test_metrics_row_damage_detected(self):
        """Verify damage indicator changes when damage_detected is True."""
        indicator = ProcessingStatusIndicator()
        
        # With damage
        html_damage = indicator.render_metrics_row_html(
            fps=30.0,
            latency_ms=50.0,
            object_count=5,
            wagon_count=10,
            damage_count=2,
            illumination_status="NORMAL",
            deblur_status="NORMAL",
            ocr_status="NORMAL",
            damage_detected=True
        )
        assert "ALERT" in html_damage, "Should show ALERT when damage_detected=True"
        
        # Without damage
        html_normal = indicator.render_metrics_row_html(
            fps=30.0,
            latency_ms=50.0,
            object_count=5,
            wagon_count=10,
            damage_count=0,
            illumination_status="NORMAL",
            deblur_status="NORMAL",
            ocr_status="NORMAL",
            damage_detected=False
        )
        assert "OK" in html_normal, "Should show OK when damage_detected=False"


class TestProcessingStatusEnum:
    """Tests for ProcessingStatus enum."""

    def test_all_status_values_exist(self):
        """Verify all expected status values are defined."""
        expected_statuses = ["APPLIED", "ACTIVE", "NORMAL", "SKIPPED", "OFF", "ERROR"]
        
        for status_name in expected_statuses:
            assert hasattr(ProcessingStatus, status_name), \
                f"ProcessingStatus should have {status_name}"
            assert ProcessingStatus[status_name].value == status_name, \
                f"ProcessingStatus.{status_name}.value should be '{status_name}'"
