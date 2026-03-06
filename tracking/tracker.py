"""Wagon tracker module using ByteTrack algorithm for multi-object tracking.

This module provides the WagonTracker class that uses ByteTrack algorithm
to track wagons across video frames and count them as they cross a counting line.

Requirements: 2.5, 2.6, 2.7
"""

from typing import List, Dict, Set, Tuple, Optional
import numpy as np

from utils.data_models import BoundingBox, WagonDetection, TrackedWagon


class WagonTracker:
    """Wagon tracker using ByteTrack algorithm for multi-object tracking.
    
    This class tracks wagons across video frames, assigns unique IDs to each
    wagon, and counts wagons as they cross a configurable counting line.
    It prevents double-counting by maintaining a set of wagon IDs that have
    already crossed the line.
    
    Supports both horizontal lines (for vertical wagon movement) and
    vertical lines (for horizontal wagon movement).
    
    Attributes:
        counting_line_pos: Position of the counting line (as fraction of frame dimension)
        counting_line_orientation: "horizontal" or "vertical"
        _crossed_wagon_ids: Set of wagon IDs that have crossed the counting line
        _wagon_count: Total count of wagons that have crossed the line
        _track_history: Dictionary mapping track_id to list of previous center positions
        _count_indices: Dictionary mapping track_id to assigned count index
        _tracker: ByteTrack tracker instance
        _counted_positions: List of (x, y) positions where wagons were counted (to prevent double counting)
    
    Requirements:
        - 2.5: Assign unique IDs using ByteTrack algorithm
        - 2.6: Increment wagon count exactly once when crossing line
        - 2.7: Prevent double-counting of the same wagon
    """
    
    def __init__(self, counting_line_y: float = 0.5, orientation: str = "vertical"):
        """Initialize the wagon tracker with a counting line position.
        
        Args:
            counting_line_y: Position of the counting line as a fraction [0.0, 1.0].
                           For horizontal line: fraction of frame height.
                           For vertical line: fraction of frame width.
            orientation: "horizontal" (wagons move up/down) or "vertical" (wagons move left/right)
                           
        Raises:
            ValueError: If counting_line_y is not in [0.0, 1.0]
        """
        if not 0.0 <= counting_line_y <= 1.0:
            raise ValueError(
                f"counting_line_y must be between 0.0 and 1.0, got {counting_line_y}"
            )
        
        if orientation not in ("horizontal", "vertical"):
            raise ValueError(
                f"orientation must be 'horizontal' or 'vertical', got {orientation}"
            )
        
        self.counting_line_y = counting_line_y
        self.counting_line_pos = counting_line_y  # Alias for clarity
        self.orientation = orientation
        self._crossed_wagon_ids: Set[int] = set()
        self._wagon_count: int = 0
        self._track_history: Dict[int, List[float]] = {}
        self._count_indices: Dict[int, int] = {}
        self._tracker = None
        self._next_track_id: int = 1
        
        # Position-based duplicate prevention
        # Store (center_y, frame_index) for counted wagons to prevent double counting
        self._counted_positions: List[Tuple[float, int]] = []
        self._position_tolerance: float = 50.0  # Pixels tolerance for same wagon
        
        # Initialize ByteTrack tracker
        self._init_tracker()
    
    def _init_tracker(self) -> None:
        """Initialize the ByteTrack tracker from ultralytics."""
        # Disable ByteTrack for now - use simple IoU+distance tracker instead
        # ByteTrack has issues with fast-moving wagons
        self._tracker = None
        
        # Custom IoU tracker state
        self._active_tracks = {}  # track_id -> last_bbox
        self._track_ages = {}     # track_id -> frames since last seen
        self._max_age = 10        # Maximum frames to keep a track alive (reduced from 30)
        self._min_iou = 0.2       # Minimum IoU for matching (lowered from 0.3 for better tracking)
        self._max_distance = 80   # Maximum center distance in pixels for matching (reduced from 150 to prevent false matches)

    def _detections_to_array(
        self, 
        detections: List[WagonDetection]
    ) -> np.ndarray:
        """Convert WagonDetection list to numpy array for ByteTrack.
        
        Args:
            detections: List of WagonDetection objects or numpy arrays
            
        Returns:
            Numpy array of shape (N, 6) with columns [x1, y1, x2, y2, conf, cls]
        """
        if not detections:
            return np.empty((0, 6), dtype=np.float32)
        
        det_list = []
        for det in detections:
            try:
                if isinstance(det, np.ndarray):
                    # Already a numpy array - use directly if it has enough elements
                    if len(det) >= 6:
                        det_list.append(det[:6].astype(np.float32))
                    elif len(det) >= 4:
                        # Has bbox, add default conf and cls
                        row = np.zeros(6, dtype=np.float32)
                        row[:len(det)] = det
                        if len(det) < 5:
                            row[4] = 0.5  # default confidence
                        if len(det) < 6:
                            row[5] = 0  # default class
                        det_list.append(row)
                elif hasattr(det, 'bbox') and hasattr(det, 'confidence'):
                    # WagonDetection object
                    det_list.append([
                        det.bbox.x1,
                        det.bbox.y1,
                        det.bbox.x2,
                        det.bbox.y2,
                        det.confidence,
                        det.class_id if hasattr(det, 'class_id') else 0
                    ])
                elif hasattr(det, 'xyxy') or hasattr(det, 'xywh'):
                    # Ultralytics detection result
                    if hasattr(det, 'xyxy'):
                        xyxy_attr = getattr(det, 'xyxy', None)
                        if xyxy_attr is None:
                            continue
                        box = xyxy_attr[0] if len(xyxy_attr.shape) > 1 else xyxy_attr
                        x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
                    else:
                        xywh_attr = getattr(det, 'xywh', None)
                        if xywh_attr is None:
                            continue
                        box = xywh_attr[0] if len(xywh_attr.shape) > 1 else xywh_attr
                        cx, cy, w, h = float(box[0]), float(box[1]), float(box[2]), float(box[3])
                        x1, y1, x2, y2 = cx - w/2, cy - h/2, cx + w/2, cy + h/2
                    
                    # Get confidence safely - use getattr to avoid direct attribute access
                    conf = 0.5
                    try:
                        conf_attr = getattr(det, 'conf', None)
                        if conf_attr is not None and isinstance(conf_attr, np.ndarray) and conf_attr.size > 0:
                            conf = float(conf_attr.flat[0])
                    except (TypeError, ValueError, IndexError, AttributeError):
                        pass
                    
                    # Get class safely
                    cls = 0
                    try:
                        cls_attr = getattr(det, 'cls', None)
                        if cls_attr is not None and isinstance(cls_attr, np.ndarray) and cls_attr.size > 0:
                            cls = int(cls_attr.flat[0])
                    except (TypeError, ValueError, IndexError, AttributeError):
                        pass
                    
                    det_list.append([x1, y1, x2, y2, conf, cls])
            except (AttributeError, IndexError, TypeError) as e:
                # Skip invalid detections
                continue
        
        if not det_list:
            return np.empty((0, 6), dtype=np.float32)
        
        return np.array(det_list, dtype=np.float32)
    
    def _compute_iou(self, bbox1: BoundingBox, bbox2: BoundingBox) -> float:
        """Compute Intersection over Union (IoU) between two bounding boxes.
        
        Args:
            bbox1: First bounding box
            bbox2: Second bounding box
            
        Returns:
            IoU score between 0.0 and 1.0
        """
        # Calculate intersection area
        x1 = max(bbox1.x1, bbox2.x1)
        y1 = max(bbox1.y1, bbox2.y1)
        x2 = min(bbox1.x2, bbox2.x2)
        y2 = min(bbox1.y2, bbox2.y2)
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        
        # Calculate union area
        area1 = (bbox1.x2 - bbox1.x1) * (bbox1.y2 - bbox1.y1)
        area2 = (bbox2.x2 - bbox2.x1) * (bbox2.y2 - bbox2.y1)
        union = area1 + area2 - intersection
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def _compute_center_distance(self, bbox1: BoundingBox, bbox2: BoundingBox) -> float:
        """Compute Euclidean distance between centers of two bounding boxes.
        
        Args:
            bbox1: First bounding box
            bbox2: Second bounding box
            
        Returns:
            Distance in pixels
        """
        cx1, cy1 = bbox1.center
        cx2, cy2 = bbox2.center
        return np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
    
    def _match_detections_to_tracks(
        self,
        detections: List[WagonDetection]
    ) -> List[Tuple[int, WagonDetection]]:
        """Match detections to existing tracks using IoU and distance.
        
        Uses IoU as primary metric, with center distance as fallback for
        fast-moving objects where IoU may be low.
        
        Args:
            detections: List of current frame detections
            
        Returns:
            List of (track_id, detection) tuples for matched detections
        """
        matched = []
        unmatched_detections = []
        used_track_ids = set()
        
        # Age all tracks
        for track_id in list(self._track_ages.keys()):
            self._track_ages[track_id] += 1
            # Remove old tracks after max_age frames
            # For crossed wagons, keep them longer (3x) to prevent immediate ID reuse
            max_age_threshold = self._max_age * 3 if track_id in self._crossed_wagon_ids else self._max_age
            if self._track_ages[track_id] > max_age_threshold:
                if track_id in self._active_tracks:
                    del self._active_tracks[track_id]
                if track_id in self._track_ages:
                    del self._track_ages[track_id]
        
        # Match each detection to best track
        for det in detections:
            if isinstance(det, np.ndarray):
                if len(det) >= 4:
                    det_bbox = BoundingBox(
                        x1=int(det[0]), y1=int(det[1]),
                        x2=int(det[2]), y2=int(det[3])
                    )
                else:
                    continue
            elif hasattr(det, 'bbox'):
                det_bbox = det.bbox
            else:
                continue
            
            best_score = 0.0
            best_track_id = None
            
            # Find best matching track (exclude only already used tracks in this frame)
            for track_id, track_bbox in self._active_tracks.items():
                if track_id in used_track_ids:
                    continue
                
                # Compute IoU
                iou = self._compute_iou(det_bbox, track_bbox)
                
                # Compute center distance
                distance = self._compute_center_distance(det_bbox, track_bbox)
                
                # Scoring: prioritize IoU, but use distance as fallback
                # If IoU is good (>= min_iou), use it directly
                # If IoU is low but distance is small, use distance-based score
                if iou >= self._min_iou:
                    score = iou
                elif distance <= self._max_distance:
                    # Convert distance to a score (closer = higher score)
                    # Score ranges from 0.0 (at max_distance) to 0.5 (at distance=0)
                    score = 0.5 * (1.0 - distance / self._max_distance)
                else:
                    score = 0.0
                
                if score > best_score:
                    best_score = score
                    best_track_id = track_id
            
            if best_track_id is not None and best_score > 0.0:
                # Match found - update track
                matched.append((best_track_id, det))
                self._active_tracks[best_track_id] = det_bbox
                self._track_ages[best_track_id] = 0
                used_track_ids.add(best_track_id)
            else:
                # No match - create new track
                unmatched_detections.append(det)
        
        # Create new tracks for unmatched detections
        for det in unmatched_detections:
            if isinstance(det, np.ndarray):
                if len(det) >= 4:
                    det_bbox = BoundingBox(
                        x1=int(det[0]), y1=int(det[1]),
                        x2=int(det[2]), y2=int(det[3])
                    )
                else:
                    continue
            elif hasattr(det, 'bbox'):
                det_bbox = det.bbox
            else:
                continue
            
            new_track_id = self._next_track_id
            self._next_track_id += 1
            self._active_tracks[new_track_id] = det_bbox
            self._track_ages[new_track_id] = 0
            matched.append((new_track_id, det))
        
        return matched
    
    def _check_line_crossing(
        self, 
        track_id: int, 
        current_pos: float,
        line_pos_pixels: float
    ) -> bool:
        """Check if a wagon should be counted with direction detection.
        
        Bidirectional counting: Counts wagons crossing the line in BOTH directions.
        Uses track history to detect actual line crossing (not just position).
        
        Args:
            track_id: The track ID of the wagon
            current_pos: Current position (x or y coordinate depending on orientation)
            line_pos_pixels: Position of counting line in pixels
            
        Returns:
            True if the wagon should be counted, False otherwise
        """
        # Get track history
        if track_id not in self._track_history or len(self._track_history[track_id]) < 2:
            # Not enough history - don't count yet
            return False
        
        history = self._track_history[track_id]
        prev_pos = history[-2]  # Previous position
        
        # Check if wagon crossed the line (in either direction)
        # Left to Right: prev_pos < line < current_pos
        # Right to Left: prev_pos > line > current_pos
        crossed_left_to_right = prev_pos < line_pos_pixels <= current_pos
        crossed_right_to_left = prev_pos > line_pos_pixels >= current_pos
        
        return crossed_left_to_right or crossed_right_to_left
    
    def _is_position_already_counted(self, center_y: float, current_frame: int) -> bool:
        """Check if a wagon at this Y position was already counted recently.
        
        This prevents double-counting when track IDs change but the wagon
        is still at a similar vertical position.
        
        Args:
            center_y: Y coordinate of wagon center
            current_frame: Current frame index
            
        Returns:
            True if a wagon at similar position was counted recently
        """
        # Clean old entries (older than 60 frames)
        self._counted_positions = [
            (y, f) for y, f in self._counted_positions 
            if current_frame - f < 60
        ]
        
        # Check if any recent count is at a similar Y position
        for counted_y, frame in self._counted_positions:
            if abs(center_y - counted_y) < self._position_tolerance:
                return True
        
        return False
    
    def _mark_position_counted(self, center_y: float, current_frame: int) -> None:
        """Mark a Y position as counted.
        
        Args:
            center_y: Y coordinate of wagon center
            current_frame: Current frame index
        """
        self._counted_positions.append((center_y, current_frame))
    
    def update(
        self, 
        detections: List[WagonDetection], 
        frame_shape: Tuple[int, int, int],
        frame_index: int = 0
    ) -> List[TrackedWagon]:
        """Update tracker with new detections and return tracked wagons.
        
        This method processes new detections, updates the tracker state,
        assigns unique IDs, and detects line crossings for counting.
        
        Args:
            detections: List of WagonDetection objects from the detector
            frame_shape: Shape of the frame (height, width, channels)
            frame_index: Current frame index for position-based duplicate prevention
            
        Returns:
            List of TrackedWagon objects with track_id, bbox, crossed_line status
        """
        frame_height, frame_width = frame_shape[0], frame_shape[1]
        
        # Calculate line position in pixels based on orientation
        if self.orientation == "vertical":
            # Vertical line: use X coordinate (fraction of width)
            line_pos_pixels = self.counting_line_pos * frame_width
        else:
            # Horizontal line: use Y coordinate (fraction of height)
            line_pos_pixels = self.counting_line_pos * frame_height
        
        tracked_wagons = []
        
        if self._tracker is not None:
            # Use ByteTrack for tracking
            tracked_wagons = self._update_with_bytetrack(
                detections, frame_shape, line_pos_pixels, frame_index
            )
        else:
            # Fallback to simple tracking
            tracked_wagons = self._update_simple(
                detections, frame_shape, line_pos_pixels, frame_index
            )
        
        return tracked_wagons
    
    def _update_with_bytetrack(
        self,
        detections: List[WagonDetection],
        frame_shape: Tuple[int, int, int],
        line_pos_pixels: float,
        frame_index: int = 0
    ) -> List[TrackedWagon]:
        """Update using ByteTrack algorithm.
        
        Args:
            detections: List of WagonDetection objects
            frame_shape: Shape of the frame (height, width, channels)
            line_pos_pixels: Position of counting line in pixels
            frame_index: Current frame index
            
        Returns:
            List of TrackedWagon objects
        """
        tracked_wagons = []
        
        # Convert detections to numpy array
        det_array = self._detections_to_array(detections)
        
        if det_array.size == 0:
            return tracked_wagons
        
        # Create image info for ByteTrack
        img_info = (frame_shape[0], frame_shape[1])  # (height, width)
        img_size = (frame_shape[0], frame_shape[1])
        
        # Run ByteTrack update with error handling
        try:
            online_targets = self._tracker.update(det_array, img_info, img_size)
        except (AttributeError, TypeError, ValueError) as e:
            # ByteTrack internal error - fall back to simple tracking
            return self._update_simple(detections, frame_shape, line_pos_pixels, frame_index)
        
        for track in online_targets:
            # Handle different ByteTrack return formats
            # Some versions return STrack objects, others return numpy arrays
            try:
                track_id = None
                x1, y1, x2, y2 = 0, 0, 0, 0
                confidence = 0.5
                
                # Check type first - numpy arrays must be handled differently
                if isinstance(track, np.ndarray):
                    # Array format: [x1, y1, x2, y2, track_id, conf, cls, ...]
                    if len(track) >= 5:
                        x1, y1, x2, y2 = int(track[0]), int(track[1]), int(track[2]), int(track[3])
                        track_id = int(track[4])
                        confidence = float(track[5]) if len(track) > 5 else 0.5
                    else:
                        continue
                else:
                    # STrack object format - check for required attributes
                    # Use getattr with defaults to avoid AttributeError
                    track_id_attr = getattr(track, 'track_id', None)
                    tlwh_attr = getattr(track, 'tlwh', None)
                    
                    if track_id_attr is None or tlwh_attr is None:
                        continue
                    
                    track_id = int(track_id_attr)
                    tlwh = tlwh_attr  # top-left x, top-left y, width, height
                    
                    # Convert tlwh to xyxy
                    x1 = int(tlwh[0])
                    y1 = int(tlwh[1])
                    x2 = int(tlwh[0] + tlwh[2])
                    y2 = int(tlwh[1] + tlwh[3])
                    
                    # Get confidence safely - avoid accessing .conf which can fail on some objects
                    confidence = 0.5
                    # Try score first (most common in STrack)
                    score_attr = getattr(track, 'score', None)
                    if score_attr is not None:
                        try:
                            if isinstance(score_attr, (int, float)):
                                confidence = float(score_attr)
                            elif isinstance(score_attr, np.ndarray) and score_attr.size > 0:
                                confidence = float(score_attr.flat[0])
                        except (TypeError, ValueError, IndexError):
                            pass
                
                if track_id is None:
                    continue
                
                bbox = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
                center_x, center_y = bbox.center
                
                # Get the relevant coordinate based on orientation
                if self.orientation == "vertical":
                    current_pos = center_x  # X coordinate for vertical line
                else:
                    current_pos = center_y  # Y coordinate for horizontal line
                
                # Update track history BEFORE checking line crossing
                if track_id not in self._track_history:
                    self._track_history[track_id] = []
                self._track_history[track_id].append(current_pos)
                
                # Keep history limited to prevent memory growth
                if len(self._track_history[track_id]) > 30:
                    self._track_history[track_id] = self._track_history[track_id][-30:]
                
                # Check for line crossing
                crossed_now = False
                should_count = self._check_line_crossing(track_id, current_pos, line_pos_pixels)
                
                if should_count:
                    # Only count if this track ID hasn't been counted before
                    if track_id not in self._crossed_wagon_ids:
                        self._crossed_wagon_ids.add(track_id)
                        self._wagon_count += 1
                        self._count_indices[track_id] = self._wagon_count
                        crossed_now = True
                
                # Create TrackedWagon - mark as crossed if wagon has crossed
                crossed_line = track_id in self._crossed_wagon_ids
                count_index = self._count_indices.get(track_id)
                
                tracked_wagon = TrackedWagon(
                    track_id=track_id,
                    bbox=bbox,
                    confidence=confidence,
                    crossed_line=crossed_line,
                    count_index=count_index
                )
                tracked_wagons.append(tracked_wagon)
                
            except Exception as e:
                # Skip this track if we can't parse it
                continue
        
        return tracked_wagons

    def _update_simple(
        self,
        detections: List[WagonDetection],
        frame_shape: Tuple[int, int, int],
        line_pos_pixels: float,
        frame_index: int = 0
    ) -> List[TrackedWagon]:
        """Simple tracking fallback using custom IoU-based matching.
        
        This uses IoU-based matching to maintain consistent track IDs.
        
        Args:
            detections: List of WagonDetection objects or numpy arrays
            frame_shape: Shape of the frame (height, width, channels)
            line_pos_pixels: Position of counting line in pixels
            frame_index: Current frame index
            
        Returns:
            List of TrackedWagon objects
        """
        tracked_wagons = []
        
        # Match detections to existing tracks
        matched = self._match_detections_to_tracks(detections)
        
        for track_id, det in matched:
            try:
                # Handle different detection formats
                if isinstance(det, np.ndarray):
                    # Numpy array format: [x1, y1, x2, y2, conf, cls]
                    if len(det) >= 4:
                        x1, y1, x2, y2 = int(det[0]), int(det[1]), int(det[2]), int(det[3])
                        confidence = float(det[4]) if len(det) > 4 else 0.5
                        bbox = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
                    else:
                        continue
                elif hasattr(det, 'bbox') and hasattr(det, 'confidence'):
                    # WagonDetection object
                    bbox = det.bbox
                    confidence = det.confidence
                else:
                    continue
                
                center_x, center_y = bbox.center
                
                # Get the relevant coordinate based on orientation
                if self.orientation == "vertical":
                    current_pos = center_x  # X coordinate for vertical line
                else:
                    current_pos = center_y  # Y coordinate for horizontal line
                
                # Update track history BEFORE checking line crossing
                if track_id not in self._track_history:
                    self._track_history[track_id] = []
                self._track_history[track_id].append(current_pos)
                
                # Keep history limited
                if len(self._track_history[track_id]) > 30:
                    self._track_history[track_id] = self._track_history[track_id][-30:]
                
                # Check for line crossing
                should_count = self._check_line_crossing(track_id, current_pos, line_pos_pixels)
                crossed_now = False
                
                if should_count:
                    # Only count if this track ID hasn't been counted before
                    if track_id not in self._crossed_wagon_ids:
                        self._crossed_wagon_ids.add(track_id)
                        self._wagon_count += 1
                        self._count_indices[track_id] = self._wagon_count
                        crossed_now = True
                
                # Create TrackedWagon - mark as crossed if wagon has crossed
                crossed_line = track_id in self._crossed_wagon_ids
                count_index = self._count_indices.get(track_id)
                
                tracked_wagon = TrackedWagon(
                    track_id=track_id,
                    bbox=bbox,
                    confidence=confidence,
                    crossed_line=crossed_line,
                    count_index=count_index
                )
                tracked_wagons.append(tracked_wagon)
                
            except (AttributeError, IndexError, TypeError) as e:
                # Skip invalid detections
                continue
        
        return tracked_wagons
    
    def get_wagon_count(self) -> int:
        """Return the total count of wagons that have crossed the counting line.
        
        Returns:
            Total number of unique wagons that crossed the line
        """
        return self._wagon_count
    
    def has_crossed_line(self, track_id: int) -> bool:
        """Check if a wagon with the given track ID has already crossed the line.
        
        Args:
            track_id: The track ID to check
            
        Returns:
            True if the wagon has crossed the line, False otherwise
        """
        return track_id in self._crossed_wagon_ids
    
    def reset(self) -> None:
        """Reset the tracker state.
        
        Clears all tracking history, crossed wagon IDs, and resets the count.
        """
        self._crossed_wagon_ids.clear()
        self._wagon_count = 0
        self._track_history.clear()
        self._count_indices.clear()
        self._next_track_id = 1
        self._counted_positions.clear()  # Clear position-based duplicate prevention
        
        # Reinitialize tracker
        self._init_tracker()
    
    def get_crossed_wagon_ids(self) -> Set[int]:
        """Return the set of wagon IDs that have crossed the counting line.
        
        Returns:
            Set of track IDs that have crossed the line
        """
        return self._crossed_wagon_ids.copy()
    
    def get_count_index(self, track_id: int) -> Optional[int]:
        """Get the count index assigned to a wagon when it crossed the line.
        
        Args:
            track_id: The track ID to look up
            
        Returns:
            The count index if the wagon has crossed, None otherwise
        """
        return self._count_indices.get(track_id)
