"""Multi-object tracker with Kalman filtering and temporal persistence."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class TrackState(Enum):
    TENTATIVE = auto()
    CONFIRMED = auto()
    COASTING = auto()
    LOST = auto()


class MotionMode(Enum):
    IDLE = auto()
    MOVING = auto()
    FAST = auto()


@dataclass
class KalmanState:
    """Kalman filter state for 2D bounding box tracking. State: [cx, cy, w, h, vx, vy, vw, vh]."""
    x: np.ndarray = field(default_factory=lambda: np.zeros(8))
    P: np.ndarray = field(default_factory=lambda: np.eye(8) * 50.0)
    
    # Process noise — minimal size drift
    Q: np.ndarray = field(default_factory=lambda: np.diag([
        1.0, 1.0,     # cx, cy — normal position uncertainty
        0.1, 0.1,     # w, h — size barely changes between frames
        5.0, 5.0,     # vx, vy — velocity uncertainty
        0.01, 0.01    # vw, vh — no size velocity expected
    ]))
    
    # Measurement noise — trust position, distrust size
    R: np.ndarray = field(default_factory=lambda: np.diag([
        8.0, 8.0,     # cx, cy — position is reliable from NN
        60.0, 60.0    # w, h — size is noisy, trust our own estimate
    ]))
    
    # State transition (constant velocity, no size velocity model)
    F: np.ndarray = field(default_factory=lambda: np.array([
        [1, 0, 0, 0, 0.7, 0,   0,   0],
        [0, 1, 0, 0, 0,   0.7, 0,   0],
        [0, 0, 1, 0, 0,   0,   0,   0],   # no size velocity coupling
        [0, 0, 0, 1, 0,   0,   0,   0],   # no size velocity coupling
        [0, 0, 0, 0, 0.8, 0,   0,   0],
        [0, 0, 0, 0, 0,   0.8, 0,   0],
        [0, 0, 0, 0, 0,   0,   0,   0],   # vw killed
        [0, 0, 0, 0, 0,   0,   0,   0],   # vh killed
    ], dtype=np.float64))
    
    # Observation matrix (cx, cy, w, h)
    H: np.ndarray = field(default_factory=lambda: np.array([
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
    ], dtype=np.float64))


def bbox_to_center(x1: float, y1: float, x2: float, y2: float) -> Tuple[float, float, float, float]:
    """Convert (x1, y1, x2, y2) to (cx, cy, w, h)."""
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2
    cy = y1 + h / 2
    return cx, cy, w, h


def center_to_bbox(cx: float, cy: float, w: float, h: float) -> Tuple[float, float, float, float]:
    """Convert (cx, cy, w, h) to (x1, y1, x2, y2)."""
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return x1, y1, x2, y2


def iou(box1: Tuple[float, float, float, float], 
        box2: Tuple[float, float, float, float]) -> float:
    """Calculate IoU between two boxes in (x1, y1, x2, y2) format."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h
    
    if inter_area <= 0:
        return 0.0
    
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - inter_area
    
    if union <= 0:
        return 0.0
    return inter_area / union


def overlap_percentage(box1: Tuple[float, float, float, float],
                       box2: Tuple[float, float, float, float]) -> Tuple[float, float]:
    """Overlap percentage of each box relative to its own area."""
    # Intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h
    
    if inter_area <= 0:
        return 0.0, 0.0
    
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    
    pct1 = inter_area / area1 if area1 > 0 else 0.0
    pct2 = inter_area / area2 if area2 > 0 else 0.0
    
    return pct1, pct2


def merge_boxes(box1: Tuple[float, float, float, float],
                box2: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    """Merge two boxes into one bounding box that encompasses both."""
    return (
        min(box1[0], box2[0]),  # x1
        min(box1[1], box2[1]),  # y1
        max(box1[2], box2[2]),  # x2
        max(box1[3], box2[3]),  # y2
    )


@dataclass
class TrackedObject:
    """Single tracked object with Kalman state and lifecycle."""
    track_id: int
    label: str
    confidence: float
    state: TrackState = TrackState.TENTATIVE
    motion_mode: MotionMode = MotionMode.IDLE
    
    kalman: KalmanState = field(default_factory=KalmanState)
    
    hits: int = 1
    hit_streak: int = 1
    time_since_update: int = 0
    age: int = 0
    
    smoothed_confidence: float = 0.0
    
    # Reference size from first detection
    _initial_w: float = 0.0
    _initial_h: float = 0.0
    
    # Confidence buffering
    peak_confidence: float = 0.0
    frames_since_peak: int = 0
    confidence_buffer_frames: int = 20
    
    # Motion history
    position_history: List[Tuple[float, float]] = field(default_factory=list)
    motion_history_size: int = 30
    idle_threshold_px: float = 15.0
    moving_threshold_px: float = 50.0
    frames_idle: int = 0
    
    created_at: float = field(default_factory=time.time)
    last_seen_at: float = field(default_factory=time.time)
    
    def initialize(self, x1: float, y1: float, x2: float, y2: float, confidence: float) -> None:
        """Initialize Kalman state from first detection."""
        cx, cy, w, h = bbox_to_center(x1, y1, x2, y2)
        self.kalman.x = np.array([cx, cy, w, h, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.kalman.P = np.eye(8) * 20.0
        self.kalman.P[0, 0] = 10.0
        self.kalman.P[1, 1] = 10.0
        self.kalman.P[2, 2] = 2.0
        self.kalman.P[3, 3] = 2.0
        self.kalman.P[4, 4] = 50.0
        self.kalman.P[5, 5] = 50.0
        self.kalman.P[6, 6] = 1.0
        self.kalman.P[7, 7] = 1.0
        self.confidence = confidence
        self.smoothed_confidence = confidence
        self._initial_w = w
        self._initial_h = h
        self.peak_confidence = confidence
        self.frames_since_peak = 0
        self.position_history = [(cx, cy)]
        self.frames_idle = 0
        self.motion_mode = MotionMode.IDLE
    
    def _update_motion_mode(self, new_cx: float, new_cy: float) -> None:
        """Update motion mode based on recent movement."""
        self.position_history.append((new_cx, new_cy))
        
        if len(self.position_history) > self.motion_history_size:
            self.position_history = self.position_history[-self.motion_history_size:]
        
        if len(self.position_history) < 2:
            return
        
        # Calculate recent movement
        lookback = min(15, len(self.position_history) - 1)
        old_pos = self.position_history[-lookback - 1]
        new_pos = self.position_history[-1]
        
        total_movement = np.sqrt((new_pos[0] - old_pos[0])**2 + (new_pos[1] - old_pos[1])**2)
        
        # Calculate per-frame movement
        per_frame_movement = total_movement / max(1, lookback)
        
        if per_frame_movement > self.moving_threshold_px / 15:
            self.motion_mode = MotionMode.FAST
            self.frames_idle = 0
        elif per_frame_movement > self.idle_threshold_px / 30:
            self.motion_mode = MotionMode.MOVING
            self.frames_idle = 0
        else:
            self.frames_idle += 1
            if self.frames_idle > 30:
                self.motion_mode = MotionMode.IDLE
            elif self.motion_mode == MotionMode.FAST:
                self.motion_mode = MotionMode.MOVING
    
    def _get_adaptive_kalman_params(self) -> Tuple[float, float, float]:
        """Returns (velocity_factor, position_noise_scale, measurement_noise_scale)."""
        if self.motion_mode == MotionMode.IDLE:
            return (0.3, 0.5, 2.0)
        elif self.motion_mode == MotionMode.MOVING:
            return (0.7, 1.0, 1.0)
        else:
            return (1.0, 2.0, 0.3)

    def predict(self) -> Tuple[float, float, float, float]:
        """Kalman predict step. Returns predicted bbox (x1, y1, x2, y2)."""
        vel_factor, proc_noise_scale, _ = self._get_adaptive_kalman_params()
        
        # Adaptive state transition
        F = self.kalman.F.copy()
        F[0, 4] = vel_factor
        F[1, 5] = vel_factor
        
        # Scale process noise
        Q = self.kalman.Q.copy()
        Q[0, 0] *= proc_noise_scale
        Q[1, 1] *= proc_noise_scale
        Q[4, 4] *= proc_noise_scale
        Q[5, 5] *= proc_noise_scale
        
        # State prediction: x = F @ x
        self.kalman.x = F @ self.kalman.x
        
        # Covariance prediction: P = F @ P @ F.T + Q
        self.kalman.P = F @ self.kalman.P @ F.T + Q
        
        # Clamp size to prevent drift (±25% of initial)
        if self._initial_w > 0:
            min_w = self._initial_w * 0.75
            max_w = self._initial_w * 1.25
            self.kalman.x[2] = np.clip(self.kalman.x[2], min_w, max_w)
        else:
            self.kalman.x[2] = max(10.0, self.kalman.x[2])
            
        if self._initial_h > 0:
            min_h = self._initial_h * 0.75
            max_h = self._initial_h * 1.25
            self.kalman.x[3] = np.clip(self.kalman.x[3], min_h, max_h)
        else:
            self.kalman.x[3] = max(10.0, self.kalman.x[3])
        
        # Zero out tiny velocities in IDLE mode
        if self.motion_mode == MotionMode.IDLE:
            if abs(self.kalman.x[4]) < 0.5:  # Less than 0.5 px/frame
                self.kalman.x[4] = 0.0
            if abs(self.kalman.x[5]) < 0.5:
                self.kalman.x[5] = 0.0
        self.kalman.x[6] = 0.0
        self.kalman.x[7] = 0.0
        
        self.age += 1
        self.time_since_update += 1
        self.frames_since_peak += 1
        
        if self.time_since_update > 0:
            self.smoothed_confidence *= 0.95
        
        return self.get_bbox()
    
    def update(self, x1: float, y1: float, x2: float, y2: float, confidence: float) -> None:
        """Kalman update step with new detection."""
        cx, cy, w, h = bbox_to_center(x1, y1, x2, y2)
        z = np.array([cx, cy, w, h], dtype=np.float64)
        
        self._update_motion_mode(cx, cy)
        
        H = self.kalman.H
        R = self.kalman.R
        x = self.kalman.x
        P = self.kalman.P
        
        # Innovation (measurement residual)
        innovation = z - H @ x
        
        # Ignore tiny movements to reduce jitter
        if abs(innovation[0]) < 2.0:
            innovation[0] = 0.0
        if abs(innovation[1]) < 2.0:
            innovation[1] = 0.0
        # Aggressively ignore size changes — NN size output is noisy
        if abs(innovation[2]) < 8.0:
            innovation[2] = 0.0
        if abs(innovation[3]) < 8.0:
            innovation[3] = 0.0
        
        # Innovation covariance
        S = H @ P @ H.T + R
        
        # Kalman gain
        try:
            K = P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            K = P @ H.T @ np.linalg.pinv(S)
        
        # State update
        self.kalman.x = x + K @ innovation
        
        # Covariance update
        I = np.eye(8)
        self.kalman.P = (I - K @ H) @ P
        
        # Clamp size (±25% of initial)
        if self._initial_w > 0:
            min_w = self._initial_w * 0.75
            max_w = self._initial_w * 1.25
            self.kalman.x[2] = np.clip(self.kalman.x[2], min_w, max_w)
        else:
            self.kalman.x[2] = max(10.0, self.kalman.x[2])
            
        if self._initial_h > 0:
            min_h = self._initial_h * 0.75
            max_h = self._initial_h * 1.25
            self.kalman.x[3] = np.clip(self.kalman.x[3], min_h, max_h)
        else:
            self.kalman.x[3] = max(10.0, self.kalman.x[3])
        
        # Zero tiny velocities
        if self.motion_mode == MotionMode.IDLE:
            if abs(self.kalman.x[4]) < 0.5:
                self.kalman.x[4] = 0.0
            if abs(self.kalman.x[5]) < 0.5:
                self.kalman.x[5] = 0.0
        self.kalman.x[6] = 0.0
        self.kalman.x[7] = 0.0
        
        # Very slow size adaptation — only for high-confidence detections
        if confidence > 0.6 and self._initial_w > 0:
            adapt_rate = 0.02
            self._initial_w = self._initial_w * (1 - adapt_rate) + w * adapt_rate
            self._initial_h = self._initial_h * (1 - adapt_rate) + h * adapt_rate
        
        self.hits += 1
        self.hit_streak += 1
        self.time_since_update = 0
        self.last_seen_at = time.time()
        
        if confidence > self.peak_confidence:
            self.peak_confidence = confidence
            self.frames_since_peak = 0
        
        # Adaptive confidence smoothing
        # Note: time_since_update was set to 0 above, so check hit_streak==1
        # to detect re-acquisition after a coast
        just_reacquired = (self.hit_streak == 1 and self.age > 1)
        
        if just_reacquired and self.motion_mode == MotionMode.IDLE:
            alpha = 0.7
        elif self.motion_mode == MotionMode.IDLE:
            alpha = 0.25
        else:
            alpha = 0.4
        
        self.confidence = confidence
        self.smoothed_confidence = alpha * confidence + (1 - alpha) * self.smoothed_confidence
        
        # Boost confidence back up after coasting
        if just_reacquired and confidence > self.smoothed_confidence:
            self.smoothed_confidence = max(self.smoothed_confidence, confidence * 0.8)
    
    def get_bbox(self) -> Tuple[float, float, float, float]:
        """Get current bounding box estimate (x1, y1, x2, y2)."""
        cx, cy, w, h = self.kalman.x[:4]
        return center_to_bbox(cx, cy, max(10.0, w), max(10.0, h))
    
    def get_display_bbox(self) -> Tuple[float, float, float, float]:
        """Get bbox for display, possibly merged."""
        if hasattr(self, '_merged_bbox') and self._merged_bbox is not None:
            bbox = self._merged_bbox
            self._merged_bbox = None
            return bbox
        return self.get_bbox()
    
    def get_center(self) -> Tuple[float, float]:
        """Get current center position."""
        return float(self.kalman.x[0]), float(self.kalman.x[1])
    
    def get_velocity(self) -> Tuple[float, float]:
        """Get current velocity estimate (vx, vy) in pixels per frame."""
        return float(self.kalman.x[4]), float(self.kalman.x[5])
    
    def get_effective_min_confidence(self, base_threshold: float) -> float:
        """Effective min confidence with buffering (2/3 of peak)."""
        if self.frames_since_peak < self.confidence_buffer_frames:
            buffered_threshold = self.peak_confidence * (2.0 / 3.0)
            return min(base_threshold, buffered_threshold)
        return base_threshold
    
    def mark_missed(self) -> None:
        """Called when no detection matched this track."""
        self.hit_streak = 0
        self.time_since_update += 1
        self.frames_since_peak += 1


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    label: str
    confidence: float
    
    def get_bbox(self) -> Tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass
class TrackerConfig:
    """Tracker configuration."""
    iou_threshold: float = 0.3
    min_detection_confidence: float = 0.15
    min_hits_to_confirm: int = 3
    max_age_tentative: int = 3
    max_age_coasting: int = 15
    min_confidence_display: float = 0.1
    velocity_decay: float = 0.95
    confidence_decay: float = 0.92
    same_class_merge_threshold: float = 0.30
    same_class_discard_threshold: float = 0.85
    # Center-distance fallback: max distance as multiple of box diagonal
    center_distance_factor: float = 1.5


class MultiObjectTracker:
    """Multi-object tracker with Kalman filtering and Hungarian assignment."""
    
    def __init__(self, config: Optional[TrackerConfig] = None):
        self.config = config or TrackerConfig()
        self.tracks: Dict[int, TrackedObject] = {}
        self.next_id: int = 1
        self.frame_count: int = 0
    
    def update(self, detections: List[Detection]) -> List[TrackedObject]:
        """Process new frame detections, return active tracks."""
        self.frame_count += 1
        
        for track in self.tracks.values():
            track.predict()
        
        matched_tracks, matched_detections, unmatched_tracks, unmatched_detections = \
            self._associate(detections, self.config.min_detection_confidence)
        
        for track_id, det_idx in zip(matched_tracks, matched_detections):
            det = detections[det_idx]
            track = self.tracks[track_id]
            track.update(det.x1, det.y1, det.x2, det.y2, det.confidence)
            track.label = det.label
            
            if track.state == TrackState.TENTATIVE and track.hits >= self.config.min_hits_to_confirm:
                track.state = TrackState.CONFIRMED
        
        for track_id in unmatched_tracks:
            track = self.tracks[track_id]
            track.mark_missed()
            
            track.kalman.x[4] *= self.config.velocity_decay
            track.kalman.x[5] *= self.config.velocity_decay
            
            # Adaptive confidence decay
            if track.motion_mode == MotionMode.IDLE:
                confidence_decay = 0.995
            elif track.motion_mode == MotionMode.MOVING:
                confidence_decay = self.config.confidence_decay
            else:
                confidence_decay = self.config.confidence_decay * 0.98
            
            track.smoothed_confidence *= confidence_decay
            
            # Confidence floor for established tracks
            if track.state in (TrackState.CONFIRMED, TrackState.COASTING):
                if track.hits >= 10:
                    min_conf_floor = 0.25
                elif track.hits >= 5:
                    min_conf_floor = 0.18
                else:
                    min_conf_floor = 0.12
                
                track.smoothed_confidence = max(track.smoothed_confidence, min_conf_floor)
            
            # State transitions
            if track.state == TrackState.TENTATIVE:
                if track.time_since_update > self.config.max_age_tentative:
                    track.state = TrackState.LOST
            elif track.state == TrackState.CONFIRMED:
                track.state = TrackState.COASTING
            elif track.state == TrackState.COASTING:
                if track.time_since_update > self.config.max_age_coasting:
                    track.state = TrackState.LOST
        
        for det_idx in unmatched_detections:
            det = detections[det_idx]
            self._create_track(det)
        
        lost_ids = [tid for tid, t in self.tracks.items() if t.state == TrackState.LOST]
        for tid in lost_ids:
            del self.tracks[tid]
        
        displayable = self._get_displayable_tracks()
        
        displayable = self._suppress_per_class(displayable)
        
        return displayable
    
    def _associate(self, detections: List[Detection], min_detection_confidence: float = 0.0) -> Tuple[List[int], List[int], List[int], List[int]]:
        """Associate detections with tracks using IoU + Hungarian algorithm."""
        if not self.tracks or not detections:
            return [], [], list(self.tracks.keys()), list(range(len(detections)))
        
        track_ids = list(self.tracks.keys())
        n_tracks = len(track_ids)
        n_dets = len(detections)
        
        # Build cost matrix
        cost_matrix = np.zeros((n_tracks, n_dets), dtype=np.float64)
        
        for i, track_id in enumerate(track_ids):
            track = self.tracks[track_id]
            track_bbox = track.get_bbox()
            effective_min_conf = track.get_effective_min_confidence(min_detection_confidence)
            
            for j, det in enumerate(detections):
                if track.label == det.label:
                    if det.confidence >= effective_min_conf:
                        iou_score = iou(track_bbox, det.get_bbox())
                        cost_matrix[i, j] = 1.0 - iou_score
                    else:
                        cost_matrix[i, j] = 1e6
                else:
                    cost_matrix[i, j] = 1e6
        
        # Use Hungarian algorithm for optimal assignment
        if HAS_SCIPY:
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
        else:
            row_indices, col_indices = self._greedy_match(cost_matrix)
        
        matched_tracks = []
        matched_detections = []
        unmatched_tracks = set(track_ids)
        unmatched_detections = set(range(n_dets))
        
        for row, col in zip(row_indices, col_indices):
            iou_score = 1.0 - cost_matrix[row, col]
            if iou_score >= self.config.iou_threshold:
                track_id = track_ids[row]
                matched_tracks.append(track_id)
                matched_detections.append(col)
                unmatched_tracks.discard(track_id)
                unmatched_detections.discard(col)
        
        # ── Second pass: center-distance fallback for unmatched ──────
        #   Handles the case where the NN outputs a different box size
        #   for the same insect, making IoU too low to match.
        if unmatched_tracks and unmatched_detections:
            remaining_tracks = list(unmatched_tracks)
            remaining_dets = list(unmatched_detections)
            dist_factor = self.config.center_distance_factor
            
            # Build center-distance pairs, sorted by distance
            candidates = []
            for tid in remaining_tracks:
                track = self.tracks[tid]
                tcx, tcy = track.get_center()
                tb = track.get_bbox()
                diag = np.sqrt((tb[2] - tb[0])**2 + (tb[3] - tb[1])**2)
                max_dist = diag * dist_factor
                
                for dj in remaining_dets:
                    det = detections[dj]
                    if track.label != det.label:
                        continue
                    eff_min_conf = track.get_effective_min_confidence(min_detection_confidence)
                    if det.confidence < eff_min_conf:
                        continue
                    dcx = (det.x1 + det.x2) / 2.0
                    dcy = (det.y1 + det.y2) / 2.0
                    dist = np.sqrt((tcx - dcx)**2 + (tcy - dcy)**2)
                    if dist < max_dist:
                        candidates.append((dist, tid, dj))
            
            candidates.sort()
            used_t: set = set()
            used_d: set = set()
            for _, tid, dj in candidates:
                if tid in used_t or dj in used_d:
                    continue
                matched_tracks.append(tid)
                matched_detections.append(dj)
                unmatched_tracks.discard(tid)
                unmatched_detections.discard(dj)
                used_t.add(tid)
                used_d.add(dj)
        
        return matched_tracks, matched_detections, list(unmatched_tracks), list(unmatched_detections)
    
    def _greedy_match(self, cost_matrix: np.ndarray) -> Tuple[List[int], List[int]]:
        """Fallback greedy matching without scipy."""
        n_tracks, n_dets = cost_matrix.shape
        row_indices = []
        col_indices = []
        used_rows = set()
        used_cols = set()
        
        # Find all valid pairs and sort by cost
        pairs = []
        for i in range(n_tracks):
            for j in range(n_dets):
                if cost_matrix[i, j] < 1e5:
                    pairs.append((cost_matrix[i, j], i, j))
        pairs.sort()
        
        for cost, row, col in pairs:
            if row not in used_rows and col not in used_cols:
                row_indices.append(row)
                col_indices.append(col)
                used_rows.add(row)
                used_cols.add(col)
        
        return row_indices, col_indices
    
    def _create_track(self, det: Detection) -> TrackedObject:
        """Create a new track from an unmatched detection."""
        track = TrackedObject(
            track_id=self.next_id,
            label=det.label,
            confidence=det.confidence,
            state=TrackState.TENTATIVE,
            # Pass motion detection config parameters
            motion_history_size=getattr(self.config, 'motion_history_size', 10),
            idle_threshold_px=getattr(self.config, 'idle_threshold_px', 5.0),
            moving_threshold_px=getattr(self.config, 'moving_threshold_px', 30.0),
            confidence_buffer_frames=getattr(self.config, 'confidence_buffer_frames', 20),
        )
        track.initialize(det.x1, det.y1, det.x2, det.y2, det.confidence)
        self.tracks[self.next_id] = track
        self.next_id += 1
        return track
    
    def _get_displayable_tracks(self) -> List[TrackedObject]:
        """Get tracks that should be shown."""
        displayable = []
        for track in self.tracks.values():
            # Only show confirmed or coasting tracks
            if track.state in (TrackState.CONFIRMED, TrackState.COASTING):
                # Check minimum confidence
                if track.smoothed_confidence >= self.config.min_confidence_display:
                    displayable.append(track)
        return displayable
    
    def _suppress_per_class(self, tracks: List[TrackedObject]) -> List[TrackedObject]:
        """Handle overlapping same-class tracks: merge, separate, or discard."""
        if not tracks:
            return []
        
        sorted_tracks = sorted(tracks, key=lambda t: t.smoothed_confidence, reverse=True)
        
        result: List[Tuple[TrackedObject, Tuple[float, float, float, float]]] = []
        
        for track in sorted_tracks:
            track_bbox = track.get_bbox()
            merged = False
            discarded = False
            
            for i, (other, other_bbox) in enumerate(result):
                if track.label != other.label:
                    continue
                
                pct_track, pct_other = overlap_percentage(track_bbox, other_bbox)
                max_overlap = max(pct_track, pct_other)
                
                if max_overlap >= self.config.same_class_discard_threshold:
                    discarded = True
                    break
                elif max_overlap >= self.config.same_class_merge_threshold:
                    new_bbox = merge_boxes(track_bbox, other_bbox)
                    result[i] = (other, new_bbox)
                    merged = True
                    break
            
            if not merged and not discarded:
                result.append((track, track_bbox))
        
        output = []
        for track, final_bbox in result:
            current_bbox = track.get_bbox()
            if final_bbox != current_bbox:
                track._merged_bbox = final_bbox
            output.append(track)
        
        return output
    
    def reset(self) -> None:
        """Reset the tracker state."""
        self.tracks.clear()
        self.next_id = 1
        self.frame_count = 0


def create_tracker_from_config(overlay_cfg: dict) -> MultiObjectTracker:
    """Create tracker from overlay config dict."""
    config = TrackerConfig(
        iou_threshold=float(overlay_cfg.get("tracker_iou_threshold", 0.3)),
        min_detection_confidence=float(overlay_cfg.get("tracker_min_detection_confidence", 0.15)),
        min_hits_to_confirm=int(overlay_cfg.get("tracker_min_hits", 3)),
        max_age_tentative=int(overlay_cfg.get("tracker_max_age_tentative", 3)),
        max_age_coasting=int(overlay_cfg.get("tracker_max_coasting_frames", 15)),
        min_confidence_display=float(overlay_cfg.get("tracker_min_confidence", 0.1)),
        velocity_decay=float(overlay_cfg.get("tracker_velocity_decay", 0.95)),
        confidence_decay=float(overlay_cfg.get("tracker_confidence_decay", 0.92)),
        same_class_merge_threshold=float(overlay_cfg.get("same_class_merge_threshold", 0.30)),
        same_class_discard_threshold=float(overlay_cfg.get("same_class_discard_threshold", 0.85)),
        center_distance_factor=float(overlay_cfg.get("tracker_center_distance_factor", 1.5)),
    )
    
    config.confidence_buffer_frames = int(overlay_cfg.get("tracker_confidence_buffer_frames", 20))
    config.motion_history_size = int(overlay_cfg.get("tracker_motion_history_size", 10))
    config.idle_threshold_px = float(overlay_cfg.get("tracker_idle_threshold_px", 5.0))
    config.moving_threshold_px = float(overlay_cfg.get("tracker_moving_threshold_px", 30.0))
    
    return MultiObjectTracker(config)
