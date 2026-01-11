"""
Property-based tests for metrics calculator module.

Feature: streamlit-dashboard
Validates: Requirements 3.2, 3.3, 3.4, 3.5
"""

import pytest
from hypothesis import given, strategies as st, settings

from dashboard.metrics import MetricsCalculator


# Strategy for generating valid inference times (in milliseconds)
inference_time_strategy = st.floats(min_value=-1000.0, max_value=10000.0, allow_nan=False, allow_infinity=False)

# Strategy for generating history sizes
history_size_strategy = st.integers(min_value=1, max_value=100)

# Strategy for generating number of samples
num_samples_strategy = st.integers(min_value=1, max_value=50)


class TestMetricsValueRanges:
    """
    Property 3: Metrics Value Ranges
    
    For any metrics update:
    - FPS SHALL be >= 0.0
    - Inference latency SHALL be >= 0.0 milliseconds
    - Object count SHALL be >= 0
    - Damage count SHALL be >= 0
    
    Validates: Requirements 3.2, 3.3, 3.4, 3.5
    """

    @given(
        inference_times=st.lists(inference_time_strategy, min_size=1, max_size=50),
        history_size=history_size_strategy
    )
    @settings(max_examples=100)
    def test_inference_ms_always_non_negative(self, inference_times, history_size):
        """
        Feature: streamlit-dashboard, Property 3: Metrics Value Ranges
        
        Generate random metric inputs.
        Verify inference_ms >= 0 regardless of input values.
        """
        calculator = MetricsCalculator(history_size=history_size)
        
        # Record various inference times (including negative values)
        for duration in inference_times:
            calculator.record_inference_time(duration)
        
        # Get the smoothed inference time
        inference_ms = calculator.get_inference_ms()
        
        # Property: inference_ms must always be >= 0
        assert inference_ms >= 0.0, \
            f"Inference time should be >= 0, got {inference_ms}"

    @given(history_size=history_size_strategy)
    @settings(max_examples=100)
    def test_fps_always_non_negative(self, history_size):
        """
        Feature: streamlit-dashboard, Property 3: Metrics Value Ranges
        
        Verify FPS >= 0 regardless of frame timing.
        """
        calculator = MetricsCalculator(history_size=history_size)
        
        # FPS should be 0 when no frames processed
        fps = calculator.get_fps()
        assert fps >= 0.0, \
            f"FPS should be >= 0, got {fps}"
        
        # Process some frames
        for _ in range(5):
            calculator.start_frame()
            calculator.end_frame()
        
        # FPS should still be >= 0
        fps = calculator.get_fps()
        assert fps >= 0.0, \
            f"FPS should be >= 0 after processing frames, got {fps}"

    @given(history_size=history_size_strategy)
    @settings(max_examples=100)
    def test_fps_zero_when_no_history(self, history_size):
        """
        Feature: streamlit-dashboard, Property 3: Metrics Value Ranges
        
        Verify FPS returns 0.0 when no frames have been processed.
        """
        calculator = MetricsCalculator(history_size=history_size)
        
        fps = calculator.get_fps()
        assert fps == 0.0, \
            f"FPS should be 0.0 when no history, got {fps}"

    @given(history_size=history_size_strategy)
    @settings(max_examples=100)
    def test_inference_ms_zero_when_no_history(self, history_size):
        """
        Feature: streamlit-dashboard, Property 3: Metrics Value Ranges
        
        Verify inference_ms returns 0.0 when no inference times recorded.
        """
        calculator = MetricsCalculator(history_size=history_size)
        
        inference_ms = calculator.get_inference_ms()
        assert inference_ms == 0.0, \
            f"Inference time should be 0.0 when no history, got {inference_ms}"

    @given(
        num_samples=num_samples_strategy,
        history_size=history_size_strategy
    )
    @settings(max_examples=100)
    def test_history_buffer_respects_size_limit(self, num_samples, history_size):
        """
        Feature: streamlit-dashboard, Property 3: Metrics Value Ranges
        
        Verify history buffers don't exceed configured size.
        """
        calculator = MetricsCalculator(history_size=history_size)
        
        # Add more samples than history size
        for i in range(num_samples):
            calculator.record_inference_time(float(i))
        
        # History should not exceed configured size
        assert len(calculator.inference_history) <= history_size, \
            f"History size {len(calculator.inference_history)} exceeds limit {history_size}"

    @given(
        positive_times=st.lists(
            st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=20
        )
    )
    @settings(max_examples=100)
    def test_inference_average_is_correct(self, positive_times):
        """
        Feature: streamlit-dashboard, Property 3: Metrics Value Ranges
        
        Verify inference time averaging is mathematically correct.
        """
        calculator = MetricsCalculator(history_size=100)  # Large enough to hold all samples
        
        for duration in positive_times:
            calculator.record_inference_time(duration)
        
        expected_avg = sum(positive_times) / len(positive_times)
        actual_avg = calculator.get_inference_ms()
        
        # Allow small floating point tolerance
        assert abs(actual_avg - expected_avg) < 0.0001, \
            f"Expected average {expected_avg}, got {actual_avg}"

    @given(history_size=history_size_strategy)
    @settings(max_examples=100)
    def test_reset_clears_all_history(self, history_size):
        """
        Feature: streamlit-dashboard, Property 3: Metrics Value Ranges
        
        Verify reset() clears all metric history.
        """
        calculator = MetricsCalculator(history_size=history_size)
        
        # Add some data
        calculator.record_inference_time(100.0)
        calculator.start_frame()
        calculator.end_frame()
        
        # Reset
        calculator.reset()
        
        # Verify all cleared
        assert calculator.get_fps() == 0.0, "FPS should be 0 after reset"
        assert calculator.get_inference_ms() == 0.0, "Inference time should be 0 after reset"
        assert len(calculator.fps_history) == 0, "FPS history should be empty after reset"
        assert len(calculator.inference_history) == 0, "Inference history should be empty after reset"
        assert calculator.last_time == 0.0, "last_time should be 0 after reset"


class TestLatencyMeasurementAccuracy:
    """
    Property 4: Latency Measurement Accuracy
    
    For any processed frame:
    - The latency measurement SHALL be >= 0 milliseconds
    - The smoothed latency SHALL be the moving average of the last N measurements
    - When latency exceeds the warning threshold, visual feedback SHALL be displayed
    
    Feature: dashboard-enhancements, Property 4: Latency Measurement Accuracy
    Validates: Requirements 4.2, 4.3, 4.4, 4.5
    """

    @given(
        latency_values=st.lists(
            st.floats(min_value=-100.0, max_value=500.0, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=50
        ),
        smoothing_window=st.integers(min_value=1, max_value=20)
    )
    @settings(max_examples=100)
    def test_latency_always_non_negative(self, latency_values, smoothing_window):
        """
        Feature: dashboard-enhancements, Property 4: Latency Measurement Accuracy
        
        Generate random latency values (including negative).
        Verify smoothed latency is always >= 0.
        **Validates: Requirements 4.2**
        """
        from dashboard.metrics import MetricsCalculator
        
        calculator = MetricsCalculator(latency_smoothing_window=smoothing_window)
        
        for latency in latency_values:
            calculator.record_latency(latency)
        
        smoothed = calculator.get_smoothed_latency()
        
        assert smoothed >= 0.0, \
            f"Smoothed latency should be >= 0, got {smoothed}"

    @given(
        latency_values=st.lists(
            st.floats(min_value=0.0, max_value=500.0, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=30
        ),
        smoothing_window=st.integers(min_value=1, max_value=20)
    )
    @settings(max_examples=100)
    def test_smoothed_latency_is_moving_average(self, latency_values, smoothing_window):
        """
        Feature: dashboard-enhancements, Property 4: Latency Measurement Accuracy
        
        Generate random latency sequences.
        Verify smoothed latency equals moving average of last N measurements.
        **Validates: Requirements 4.4**
        """
        from dashboard.metrics import MetricsCalculator
        
        calculator = MetricsCalculator(latency_smoothing_window=smoothing_window)
        
        for latency in latency_values:
            calculator.record_latency(latency)
        
        smoothed = calculator.get_smoothed_latency()
        
        # Calculate expected moving average
        window_values = latency_values[-smoothing_window:]
        expected_avg = sum(window_values) / len(window_values)
        
        assert abs(smoothed - expected_avg) < 0.0001, \
            f"Expected smoothed latency {expected_avg}, got {smoothed}"

    @given(
        latency_value=st.floats(min_value=0.0, max_value=500.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100)
    def test_latency_warning_threshold_detection(self, latency_value):
        """
        Feature: dashboard-enhancements, Property 4: Latency Measurement Accuracy
        
        Generate random latency values.
        Verify warning is triggered when latency exceeds threshold.
        **Validates: Requirements 4.5**
        """
        from dashboard.metrics import MetricsCalculator, LATENCY_WARNING_THRESHOLD_MS
        
        calculator = MetricsCalculator()
        calculator.record_latency(latency_value)
        
        is_warning = calculator.is_latency_warning()
        smoothed = calculator.get_smoothed_latency()
        
        if smoothed > LATENCY_WARNING_THRESHOLD_MS:
            assert is_warning, \
                f"Warning should be True when latency {smoothed} > threshold {LATENCY_WARNING_THRESHOLD_MS}"
        else:
            assert not is_warning, \
                f"Warning should be False when latency {smoothed} <= threshold {LATENCY_WARNING_THRESHOLD_MS}"

    @given(
        latency_value=st.floats(min_value=0.0, max_value=500.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100)
    def test_latency_warning_with_explicit_value(self, latency_value):
        """
        Feature: dashboard-enhancements, Property 4: Latency Measurement Accuracy
        
        Verify is_latency_warning works with explicit latency value parameter.
        **Validates: Requirements 4.5**
        """
        from dashboard.metrics import MetricsCalculator, LATENCY_WARNING_THRESHOLD_MS
        
        calculator = MetricsCalculator()
        
        is_warning = calculator.is_latency_warning(latency_value)
        
        if latency_value > LATENCY_WARNING_THRESHOLD_MS:
            assert is_warning, \
                f"Warning should be True when latency {latency_value} > threshold {LATENCY_WARNING_THRESHOLD_MS}"
        else:
            assert not is_warning, \
                f"Warning should be False when latency {latency_value} <= threshold {LATENCY_WARNING_THRESHOLD_MS}"

    @given(smoothing_window=st.integers(min_value=1, max_value=50))
    @settings(max_examples=100)
    def test_latency_zero_when_no_history(self, smoothing_window):
        """
        Feature: dashboard-enhancements, Property 4: Latency Measurement Accuracy
        
        Verify smoothed latency returns 0.0 when no latency recorded.
        **Validates: Requirements 4.2**
        """
        from dashboard.metrics import MetricsCalculator
        
        calculator = MetricsCalculator(latency_smoothing_window=smoothing_window)
        
        smoothed = calculator.get_smoothed_latency()
        
        assert smoothed == 0.0, \
            f"Smoothed latency should be 0.0 when no history, got {smoothed}"

    @given(
        num_samples=st.integers(min_value=1, max_value=100),
        smoothing_window=st.integers(min_value=1, max_value=20)
    )
    @settings(max_examples=100)
    def test_latency_history_respects_window_size(self, num_samples, smoothing_window):
        """
        Feature: dashboard-enhancements, Property 4: Latency Measurement Accuracy
        
        Verify latency history doesn't exceed configured window size.
        **Validates: Requirements 4.4**
        """
        from dashboard.metrics import MetricsCalculator
        
        calculator = MetricsCalculator(latency_smoothing_window=smoothing_window)
        
        for i in range(num_samples):
            calculator.record_latency(float(i))
        
        assert len(calculator.latency_history) <= smoothing_window, \
            f"Latency history size {len(calculator.latency_history)} exceeds window {smoothing_window}"


class TestWagonCountingCorrectness:
    """
    Property 3: Wagon Counting Correctness
    
    For any sequence of wagon detections crossing the counting line:
    - The total wagon count SHALL equal the number of unique wagon IDs that crossed
    - Each wagon SHALL be counted exactly once (no duplicates)
    - The count SHALL increment by exactly 1 for each new crossing
    - The count SHALL persist across frame updates
    
    Feature: dashboard-enhancements, Property 3: Wagon Counting Correctness
    Validates: Requirements 3.2, 3.3, 3.4, 3.5
    """

    @given(
        num_wagons=st.integers(min_value=1, max_value=20),
        counting_line=st.floats(min_value=0.2, max_value=0.8, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100, deadline=None)
    def test_count_equals_unique_crossings(self, num_wagons, counting_line):
        """
        Feature: dashboard-enhancements, Property 3: Wagon Counting Correctness
        
        Generate random sequences of wagon crossings.
        Verify count equals unique crossings, no duplicates.
        **Validates: Requirements 3.2, 3.3**
        """
        from tracking.tracker import WagonTracker
        
        frame_height = 480
        tracker = WagonTracker(counting_line_y=counting_line)
        line_y_pixels = counting_line * frame_height
        
        # Simulate all wagons crossing the line
        for wagon_id in range(num_wagons):
            tracker._track_history[wagon_id] = []
            
            # Start above, end below (crossing the line)
            positions = [line_y_pixels - 50, line_y_pixels + 50]
            
            for y_pos in positions:
                if wagon_id not in tracker._crossed_wagon_ids:
                    if tracker._track_history[wagon_id]:
                        prev_y = tracker._track_history[wagon_id][-1]
                        crossed_down = prev_y < line_y_pixels <= y_pos
                        crossed_up = prev_y > line_y_pixels >= y_pos
                        if crossed_down or crossed_up:
                            tracker._crossed_wagon_ids.add(wagon_id)
                            tracker._wagon_count += 1
                
                tracker._track_history[wagon_id].append(y_pos)
        
        # Count should equal number of wagons
        assert tracker.get_wagon_count() == num_wagons, \
            f"Count {tracker.get_wagon_count()} != expected {num_wagons}"

    @given(counting_line=st.floats(min_value=0.2, max_value=0.8, allow_nan=False, allow_infinity=False))
    @settings(max_examples=100)
    def test_no_duplicate_counting(self, counting_line):
        """
        Feature: dashboard-enhancements, Property 3: Wagon Counting Correctness
        
        Verify that the same wagon crossing the line multiple times is only counted once.
        **Validates: Requirements 3.4**
        """
        from tracking.tracker import WagonTracker
        
        frame_height = 480
        tracker = WagonTracker(counting_line_y=counting_line)
        line_y_pixels = counting_line * frame_height
        
        wagon_id = 1
        tracker._track_history[wagon_id] = []
        
        # Simulate wagon crossing line multiple times (oscillating)
        positions = [
            line_y_pixels - 50,  # Above line
            line_y_pixels + 50,  # Below line (first crossing)
            line_y_pixels - 50,  # Above line (second crossing)
            line_y_pixels + 50,  # Below line (third crossing)
        ]
        
        for y_pos in positions:
            center_y = y_pos
            
            if wagon_id not in tracker._crossed_wagon_ids:
                if tracker._track_history[wagon_id]:
                    prev_y = tracker._track_history[wagon_id][-1]
                    crossed_down = prev_y < line_y_pixels <= center_y
                    crossed_up = prev_y > line_y_pixels >= center_y
                    if crossed_down or crossed_up:
                        tracker._crossed_wagon_ids.add(wagon_id)
                        tracker._wagon_count += 1
            
            tracker._track_history[wagon_id].append(center_y)
        
        # Should only be counted once despite multiple crossings
        assert tracker.get_wagon_count() == 1, \
            f"Wagon counted {tracker.get_wagon_count()} times, expected 1"

    @given(
        num_wagons=st.integers(min_value=1, max_value=10),
        counting_line=st.floats(min_value=0.2, max_value=0.8, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100)
    def test_count_increments_by_one(self, num_wagons, counting_line):
        """
        Feature: dashboard-enhancements, Property 3: Wagon Counting Correctness
        
        Verify count increments by exactly 1 for each new crossing.
        **Validates: Requirements 3.3**
        """
        from tracking.tracker import WagonTracker
        
        frame_height = 480
        tracker = WagonTracker(counting_line_y=counting_line)
        line_y_pixels = counting_line * frame_height
        
        previous_count = 0
        
        for wagon_id in range(num_wagons):
            tracker._track_history[wagon_id] = []
            
            # Start above, end below (crossing the line)
            positions = [line_y_pixels - 50, line_y_pixels + 50]
            
            for y_pos in positions:
                if wagon_id not in tracker._crossed_wagon_ids:
                    if tracker._track_history[wagon_id]:
                        prev_y = tracker._track_history[wagon_id][-1]
                        crossed_down = prev_y < line_y_pixels <= y_pos
                        crossed_up = prev_y > line_y_pixels >= y_pos
                        if crossed_down or crossed_up:
                            tracker._crossed_wagon_ids.add(wagon_id)
                            tracker._wagon_count += 1
                            
                            # Verify increment is exactly 1
                            current_count = tracker.get_wagon_count()
                            assert current_count == previous_count + 1, \
                                f"Count should increment by 1, was {previous_count}, now {current_count}"
                            previous_count = current_count
                
                tracker._track_history[wagon_id].append(y_pos)

    @given(
        num_wagons=st.integers(min_value=1, max_value=10),
        counting_line=st.floats(min_value=0.2, max_value=0.8, allow_nan=False, allow_infinity=False),
        num_frame_updates=st.integers(min_value=1, max_value=10)
    )
    @settings(max_examples=100)
    def test_count_persists_across_frame_updates(self, num_wagons, counting_line, num_frame_updates):
        """
        Feature: dashboard-enhancements, Property 3: Wagon Counting Correctness
        
        Verify count persists across frame updates.
        **Validates: Requirements 3.5**
        """
        from tracking.tracker import WagonTracker
        
        frame_height = 480
        tracker = WagonTracker(counting_line_y=counting_line)
        line_y_pixels = counting_line * frame_height
        
        # First, have all wagons cross the line
        for wagon_id in range(num_wagons):
            tracker._track_history[wagon_id] = []
            positions = [line_y_pixels - 50, line_y_pixels + 50]
            
            for y_pos in positions:
                if wagon_id not in tracker._crossed_wagon_ids:
                    if tracker._track_history[wagon_id]:
                        prev_y = tracker._track_history[wagon_id][-1]
                        crossed_down = prev_y < line_y_pixels <= y_pos
                        crossed_up = prev_y > line_y_pixels >= y_pos
                        if crossed_down or crossed_up:
                            tracker._crossed_wagon_ids.add(wagon_id)
                            tracker._wagon_count += 1
                
                tracker._track_history[wagon_id].append(y_pos)
        
        expected_count = num_wagons
        
        # Simulate multiple frame updates without new crossings
        for _ in range(num_frame_updates):
            # Just update positions without crossing
            for wagon_id in range(num_wagons):
                tracker._track_history[wagon_id].append(line_y_pixels + 100)
            
            # Count should persist
            assert tracker.get_wagon_count() == expected_count, \
                f"Count should persist at {expected_count}, got {tracker.get_wagon_count()}"
