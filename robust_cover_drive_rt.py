#!/usr/bin/env python3
"""
Robust Cover Drive Analysis - Fixes for MediaPipe Pose Issues

Key Improvements:
- Temporal smoothing for keypoints to handle jittery/jumping landmarks
- Confidence-based keypoint validation
- Alternative angle calculations when primary joints fail
- Occlusion handling and keypoint interpolation
- Better stance detection and front/back side determination
- Fallback metrics when knee tracking fails
- Visibility threshold adjustments for cricket-specific poses

Usage:
  python robust_cover_drive_rt.py \
      --source "https://youtube.com/shorts/vSX3IRxGnNY" \
      --stance auto \
      --confidence_threshold 0.6 \
      --smoothing_window 5 \
      --show 1
"""

import argparse
import json
import math
import os
import sys
import time
from collections import deque
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Deque

import cv2
import numpy as np
import pandas as pd

# Optional import: yt_dlp for YouTube download
try:
    from yt_dlp import YoutubeDL
except Exception:
    YoutubeDL = None

# MediaPipe Pose
try:
    import mediapipe as mp
except Exception as e:
    print("[ERROR] mediapipe not installed. pip install mediapipe", file=sys.stderr)
    raise

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

OUTPUT_DIR = os.path.join(os.getcwd(), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Enhanced colors and display
GREEN = (40, 200, 40)
RED = (40, 40, 220)
WHITE = (240, 240, 240)
YELLOW = (40, 220, 220)
ORANGE = (40, 165, 255)
BLUE = (255, 100, 40)
FONT = cv2.FONT_HERSHEY_SIMPLEX

# ----------------------------- Enhanced Keypoint Handling ----------------------------- #

@dataclass
class KeypointData:
    """Enhanced keypoint with confidence and temporal data"""
    x: float
    y: float
    confidence: float
    visibility: float
    timestamp: float
    valid: bool = True

class TemporalSmoother:
    """Temporal smoothing for keypoints to handle MediaPipe jitter"""
    
    def __init__(self, window_size: int = 5, confidence_threshold: float = 0.5):
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold
        self.history: Dict[str, Deque[KeypointData]] = {}
    
    def add_keypoint(self, name: str, kp: Optional[KeypointData]):
        if name not in self.history:
            self.history[name] = deque(maxlen=self.window_size)
        
        if kp is not None and kp.confidence >= self.confidence_threshold:
            self.history[name].append(kp)
        elif len(self.history[name]) > 0:
            # If current detection is poor, don't add but keep history
            pass
    
    def get_smoothed(self, name: str) -> Optional[KeypointData]:
        if name not in self.history or len(self.history[name]) == 0:
            return None
        
        recent = list(self.history[name])
        if len(recent) == 1:
            return recent[0]
        
        # Weighted average based on confidence and recency
        weights = []
        xs, ys = [], []
        
        for i, kp in enumerate(recent):
            # More recent points get higher weight
            recency_weight = (i + 1) / len(recent)
            confidence_weight = kp.confidence
            weight = recency_weight * confidence_weight
            
            weights.append(weight)
            xs.append(kp.x)
            ys.append(kp.y)
        
        if sum(weights) == 0:
            return recent[-1]  # Return most recent if all weights are 0
        
        # Weighted average
        total_weight = sum(weights)
        avg_x = sum(x * w for x, w in zip(xs, weights)) / total_weight
        avg_y = sum(y * w for y, w in zip(ys, weights)) / total_weight
        avg_conf = sum(kp.confidence for kp in recent) / len(recent)
        
        return KeypointData(
            x=avg_x, y=avg_y, 
            confidence=avg_conf,
            visibility=recent[-1].visibility,
            timestamp=recent[-1].timestamp,
            valid=True
        )

# ----------------------------- Robust Angle Calculations ----------------------------- #

def robust_angle_between(a: Optional[KeypointData], b: Optional[KeypointData], 
                        c: Optional[KeypointData], min_confidence: float = 0.4) -> Optional[float]:
    """Enhanced angle calculation with confidence checking"""
    if any(p is None for p in [a, b, c]):
        return None
    if any(p.confidence < min_confidence for p in [a, b, c]):
        return None
    
    try:
        p1 = np.array([a.x, a.y])
        p2 = np.array([b.x, b.y])
        p3 = np.array([c.x, c.y])
        
        # Check if points are too close (degenerate case)
        if (np.linalg.norm(p1 - p2) < 5 or np.linalg.norm(p3 - p2) < 5):
            return None
        
        ba = p1 - p2
        bc = p3 - p2
        
        nba = np.linalg.norm(ba)
        nbc = np.linalg.norm(bc)
        
        if nba == 0 or nbc == 0:
            return None
            
        cosang = np.dot(ba, bc) / (nba * nbc)
        cosang = np.clip(cosang, -1.0, 1.0)
        angle = float(np.degrees(np.arccos(cosang)))
        
        # Sanity check - anatomically impossible angles
        if angle > 180 or angle < 0:
            return None
            
        return angle
        
    except Exception:
        return None

def estimate_knee_from_hip_ankle(hip: Optional[KeypointData], 
                                ankle: Optional[KeypointData],
                                typical_thigh_calf_ratio: float = 0.55) -> Optional[KeypointData]:
    """Estimate knee position when MediaPipe knee detection fails"""
    if hip is None or ankle is None:
        return None
    
    # Typical human proportions: thigh is ~55% of leg length
    dx = ankle.x - hip.x
    dy = ankle.y - hip.y
    
    est_x = hip.x + dx * typical_thigh_calf_ratio
    est_y = hip.y + dy * typical_thigh_calf_ratio
    
    return KeypointData(
        x=est_x, y=est_y, 
        confidence=min(hip.confidence, ankle.confidence) * 0.7,  # Reduce confidence for estimate
        visibility=min(hip.visibility, ankle.visibility),
        timestamp=max(hip.timestamp, ankle.timestamp),
        valid=True
    )

# ----------------------------- Enhanced Pose Processing ----------------------------- #

class CricketPoseAnalyzer:
    def __init__(self, confidence_threshold: float = 0.5, smoothing_window: int = 5):
        self.smoother = TemporalSmoother(smoothing_window, confidence_threshold)
        self.confidence_threshold = confidence_threshold
        
        # Cricket-specific visibility thresholds (lower for bent poses)
        self.visibility_thresholds = {
            'head': 0.3,
            'shoulder': 0.4,
            'elbow': 0.3,  # Often partially occluded in cricket
            'wrist': 0.2,  # Can be behind body
            'hip': 0.5,
            'knee': 0.2,   # Problematic in bent poses - lowered threshold
            'ankle': 0.3,
            'foot': 0.2
        }
    
    def extract_keypoint(self, landmarks, idx: int, w: int, h: int, 
                        timestamp: float, point_type: str = 'default') -> Optional[KeypointData]:
        """Extract keypoint with cricket-specific confidence thresholds"""
        try:
            lm = landmarks[idx]
            
            # Use cricket-specific visibility threshold
            vis_threshold = self.visibility_thresholds.get(point_type, 0.3)
            
            if (hasattr(lm, 'visibility') and lm.visibility is not None and 
                lm.visibility < vis_threshold):
                return None
            
            confidence = getattr(lm, 'visibility', 1.0)
            
            return KeypointData(
                x=lm.x * w,
                y=lm.y * h,
                confidence=confidence,
                visibility=confidence,
                timestamp=timestamp,
                valid=True
            )
        except Exception:
            return None
    
    def process_frame(self, landmarks, w: int, h: int, timestamp: float, stance: str) -> Dict:
        """Process frame with robust keypoint extraction and smoothing"""
        LANDMARKS = mp_pose.PoseLandmark
        
        # Extract all keypoints
        keypoints = {}
        
        # Head
        keypoints['nose'] = self.extract_keypoint(landmarks, LANDMARKS.NOSE, w, h, timestamp, 'head')
        
        # Upper body
        keypoints['l_shoulder'] = self.extract_keypoint(landmarks, LANDMARKS.LEFT_SHOULDER, w, h, timestamp, 'shoulder')
        keypoints['r_shoulder'] = self.extract_keypoint(landmarks, LANDMARKS.RIGHT_SHOULDER, w, h, timestamp, 'shoulder')
        keypoints['l_elbow'] = self.extract_keypoint(landmarks, LANDMARKS.LEFT_ELBOW, w, h, timestamp, 'elbow')
        keypoints['r_elbow'] = self.extract_keypoint(landmarks, LANDMARKS.RIGHT_ELBOW, w, h, timestamp, 'elbow')
        keypoints['l_wrist'] = self.extract_keypoint(landmarks, LANDMARKS.LEFT_WRIST, w, h, timestamp, 'wrist')
        keypoints['r_wrist'] = self.extract_keypoint(landmarks, LANDMARKS.RIGHT_WRIST, w, h, timestamp, 'wrist')
        
        # Lower body
        keypoints['l_hip'] = self.extract_keypoint(landmarks, LANDMARKS.LEFT_HIP, w, h, timestamp, 'hip')
        keypoints['r_hip'] = self.extract_keypoint(landmarks, LANDMARKS.RIGHT_HIP, w, h, timestamp, 'hip')
        keypoints['l_knee'] = self.extract_keypoint(landmarks, LANDMARKS.LEFT_KNEE, w, h, timestamp, 'knee')
        keypoints['r_knee'] = self.extract_keypoint(landmarks, LANDMARKS.RIGHT_KNEE, w, h, timestamp, 'knee')
        keypoints['l_ankle'] = self.extract_keypoint(landmarks, LANDMARKS.LEFT_ANKLE, w, h, timestamp, 'ankle')
        keypoints['r_ankle'] = self.extract_keypoint(landmarks, LANDMARKS.RIGHT_ANKLE, w, h, timestamp, 'ankle')
        
        # Feet
        keypoints['l_heel'] = self.extract_keypoint(landmarks, LANDMARKS.LEFT_HEEL, w, h, timestamp, 'foot')
        keypoints['r_heel'] = self.extract_keypoint(landmarks, LANDMARKS.RIGHT_HEEL, w, h, timestamp, 'foot')
        keypoints['l_foot_index'] = self.extract_keypoint(landmarks, LANDMARKS.LEFT_FOOT_INDEX, w, h, timestamp, 'foot')
        keypoints['r_foot_index'] = self.extract_keypoint(landmarks, LANDMARKS.RIGHT_FOOT_INDEX, w, h, timestamp, 'foot')
        
        # Add to temporal smoother
        for name, kp in keypoints.items():
            self.smoother.add_keypoint(name, kp)
        
        # Get smoothed keypoints
        smoothed = {}
        for name in keypoints.keys():
            smoothed[name] = self.smoother.get_smoothed(name)
        
        # Handle failed knee detection with estimation
        if smoothed['l_knee'] is None and smoothed['l_hip'] is not None and smoothed['l_ankle'] is not None:
            smoothed['l_knee'] = estimate_knee_from_hip_ankle(smoothed['l_hip'], smoothed['l_ankle'])
            
        if smoothed['r_knee'] is None and smoothed['r_hip'] is not None and smoothed['r_ankle'] is not None:
            smoothed['r_knee'] = estimate_knee_from_hip_ankle(smoothed['r_hip'], smoothed['r_ankle'])
        
        return smoothed
    
    def determine_front_side(self, keypoints: Dict, stance: str) -> str:
        """Enhanced stance detection"""
        if stance in ['right', 'left']:
            return 'left' if stance == 'right' else 'right'
        
        # Auto detection with multiple cues
        l_hip = keypoints.get('l_hip')
        r_hip = keypoints.get('r_hip')
        l_shoulder = keypoints.get('l_shoulder')
        r_shoulder = keypoints.get('r_shoulder')
        
        votes = []
        
        # Hip position vote
        if l_hip and r_hip:
            votes.append('left' if l_hip.x < r_hip.x else 'right')
        
        # Shoulder position vote  
        if l_shoulder and r_shoulder:
            votes.append('left' if l_shoulder.x < r_shoulder.x else 'right')
        
        # Return majority vote or default
        if not votes:
            return 'left'  # Default
        
        left_votes = votes.count('left')
        return 'left' if left_votes > len(votes) / 2 else 'right'

# ----------------------------- Enhanced Metrics ----------------------------- #

@dataclass
class RobustFrameMetrics:
    frame_idx: int
    timestamp: float
    # Primary metrics
    elbow_angle: Optional[float]
    knee_angle: Optional[float]  # Added knee angle
    spine_lean_deg: Optional[float]
    head_knee_xproj_norm: Optional[float]
    foot_dir_deg: Optional[float]
    # Quality indicators
    pose_quality: float  # Overall pose detection quality
    front_side: str
    # Individual joint qualities
    elbow_quality: float
    knee_quality: float
    spine_quality: float
    # Enhanced cues
    cues: Dict[str, bool]
    warnings: List[str]

def compute_robust_metrics(keypoints: Dict, w: int, h: int, stance: str, analyzer: CricketPoseAnalyzer) -> RobustFrameMetrics:
    """Compute metrics with fallbacks and quality assessment"""
    
    front_side = analyzer.determine_front_side(keypoints, stance)
    warnings = []
    
    # Select front/back keypoints
    prefix = 'l_' if front_side == 'left' else 'r_'
    back_prefix = 'r_' if front_side == 'left' else 'l_'
    
    front_shoulder = keypoints.get(f'{prefix}shoulder')
    front_elbow = keypoints.get(f'{prefix}elbow')
    front_wrist = keypoints.get(f'{prefix}wrist')
    front_hip = keypoints.get(f'{prefix}hip')
    front_knee = keypoints.get(f'{prefix}knee')
    front_ankle = keypoints.get(f'{prefix}ankle')
    
    back_hip = keypoints.get(f'{back_prefix}hip')
    back_shoulder = keypoints.get(f'{back_prefix}shoulder')
    
    nose = keypoints.get('nose')
    
    # 1. Front elbow angle with quality assessment
    elbow_angle = robust_angle_between(front_shoulder, front_elbow, front_wrist)
    elbow_quality = 0.0
    if front_shoulder and front_elbow and front_wrist:
        elbow_quality = min(front_shoulder.confidence, front_elbow.confidence, front_wrist.confidence)
    
    if elbow_angle is None and elbow_quality > 0.3:
        warnings.append("Elbow angle calculation failed despite reasonable detection")
    
    # 2. Front knee angle (thigh-shin angle)
    knee_angle = robust_angle_between(front_hip, front_knee, front_ankle)
    knee_quality = 0.0
    if front_hip and front_knee and front_ankle:
        knee_quality = min(front_hip.confidence, front_knee.confidence, front_ankle.confidence)
    elif front_knee and front_knee.confidence < 0.5:
        warnings.append("Knee tracking unreliable - using estimated position")
    
    # 3. Spine lean
    spine_lean_deg = None
    spine_quality = 0.0
    
    if front_hip and back_hip and front_shoulder and back_shoulder:
        # Use hip midpoint to shoulder midpoint
        hip_mid_x = (front_hip.x + back_hip.x) / 2
        hip_mid_y = (front_hip.y + back_hip.y) / 2
        sho_mid_x = (front_shoulder.x + back_shoulder.x) / 2
        sho_mid_y = (front_shoulder.y + back_shoulder.y) / 2
        
        spine_vec = np.array([sho_mid_x - hip_mid_x, sho_mid_y - hip_mid_y])
        vertical = np.array([0.0, -1.0])  # Up in image coordinates
        
        spine_quality = min(front_hip.confidence, back_hip.confidence, 
                           front_shoulder.confidence, back_shoulder.confidence)
        
        if np.linalg.norm(spine_vec) > 10:  # Minimum spine length
            cosang = np.dot(spine_vec, vertical) / (np.linalg.norm(spine_vec) * np.linalg.norm(vertical))
            cosang = np.clip(cosang, -1.0, 1.0)
            spine_lean_deg = float(np.degrees(np.arccos(cosang)))
    
    # 4. Head over front knee
    head_knee_xproj_norm = None
    if nose and front_knee and front_hip and back_hip:
        hip_width = abs(back_hip.x - front_hip.x)
        if hip_width > 10:  # Reasonable hip width
            dx = abs(nose.x - front_knee.x)
            head_knee_xproj_norm = dx / hip_width
    
    # 5. Front foot direction
    foot_dir_deg = None
    front_heel = keypoints.get(f'{prefix}heel')
    front_toe = keypoints.get(f'{prefix}foot_index')
    
    if front_heel and front_toe:
        foot_vec = np.array([front_toe.x - front_heel.x, front_toe.y - front_heel.y])
        if np.linalg.norm(foot_vec) > 5:  # Minimum foot length
            xaxis = np.array([1.0, 0.0])
            cosang = np.dot(foot_vec, xaxis) / np.linalg.norm(foot_vec)
            cosang = np.clip(cosang, -1.0, 1.0)
            foot_dir_deg = float(np.degrees(np.arccos(cosang)))
    
    # Overall pose quality
    qualities = [q for q in [elbow_quality, knee_quality, spine_quality] if q > 0]
    pose_quality = sum(qualities) / len(qualities) if qualities else 0.0
    
    # Enhanced cues with quality consideration
    cues = {}
    if elbow_angle is not None and elbow_quality > 0.4:
        cues["elbow_ok"] = (100 <= elbow_angle <= 150)
    else:
        cues["elbow_ok"] = None  # Unknown
        
    if knee_angle is not None and knee_quality > 0.3:
        cues["knee_ok"] = (120 <= knee_angle <= 170)  # Reasonable knee bend
    else:
        cues["knee_ok"] = None
        
    if spine_lean_deg is not None and spine_quality > 0.4:
        cues["spine_ok"] = (5 <= spine_lean_deg <= 25)
    else:
        cues["spine_ok"] = None
        
    if head_knee_xproj_norm is not None:
        cues["head_ok"] = (head_knee_xproj_norm <= 0.25)
    else:
        cues["head_ok"] = None
        
    if foot_dir_deg is not None:
        cues["foot_ok"] = (foot_dir_deg <= 30)
    else:
        cues["foot_ok"] = None
    
    return RobustFrameMetrics(
        frame_idx=-1,  # Will be set by caller
        timestamp=-1,  # Will be set by caller
        elbow_angle=elbow_angle,
        knee_angle=knee_angle,
        spine_lean_deg=spine_lean_deg,
        head_knee_xproj_norm=head_knee_xproj_norm,
        foot_dir_deg=foot_dir_deg,
        pose_quality=pose_quality,
        front_side=front_side,
        elbow_quality=elbow_quality,
        knee_quality=knee_quality,
        spine_quality=spine_quality,
        cues=cues,
        warnings=warnings
    )

# ----------------------------- Enhanced Display ----------------------------- #

def draw_enhanced_cues(frame: np.ndarray, x: int, y: int, metrics: RobustFrameMetrics):
    """Draw enhanced cues with quality indicators"""
    line_h = 18
    y0 = y
    
    # Title with overall quality
    quality_color = GREEN if metrics.pose_quality > 0.6 else (ORANGE if metrics.pose_quality > 0.3 else RED)
    put_text(frame, f"Cricket Analysis (Quality: {metrics.pose_quality:.1f})", (x, y0), quality_color, 0.6, 2)
    y0 += line_h + 5
    
    # Front side indicator
    put_text(frame, f"Front side: {metrics.front_side}", (x, y0), WHITE, 0.5)
    y0 += line_h
    
    # Elbow angle with quality
    if metrics.elbow_angle is not None:
        cue_ok = metrics.cues.get("elbow_ok", False)
        color = GREEN if cue_ok else RED
        conf_indicator = f" ({metrics.elbow_quality:.1f})" if metrics.elbow_quality > 0 else ""
        put_text(frame, f"Elbow: {metrics.elbow_angle:.0f}°{conf_indicator}", (x, y0), color)
        y0 += line_h
        cue_text = "✅ Good elbow elevation" if cue_ok else "❌ Adjust elbow angle"
        put_text(frame, cue_text, (x, y0), color, 0.45)
        y0 += line_h
    else:
        put_text(frame, "Elbow: Not detected", (x, y0), RED, 0.5)
        y0 += line_h
    
    # Knee angle (new)
    if metrics.knee_angle is not None:
        cue_ok = metrics.cues.get("knee_ok")
        if cue_ok is not None:
            color = GREEN if cue_ok else ORANGE
            conf_indicator = f" ({metrics.knee_quality:.1f})" if metrics.knee_quality > 0 else ""
            put_text(frame, f"Knee: {metrics.knee_angle:.0f}°{conf_indicator}", (x, y0), color)
            y0 += line_h
            cue_text = "✅ Good knee bend" if cue_ok else "⚠️ Check knee bend"
            put_text(frame, cue_text, (x, y0), color, 0.45)
            y0 += line_h
    else:
        put_text(frame, "Knee: Estimated/Poor", (x, y0), ORANGE, 0.5)
        y0 += line_h
    
    # Spine lean
    if metrics.spine_lean_deg is not None:
        cue_ok = metrics.cues.get("spine_ok")
        if cue_ok is not None:
            color = GREEN if cue_ok else RED
            put_text(frame, f"Spine: {metrics.spine_lean_deg:.0f}°", (x, y0), color)
            y0 += line_h
            cue_text = "✅ Good posture" if cue_ok else "❌ Adjust lean"
            put_text(frame, cue_text, (x, y0), color, 0.45)
            y0 += line_h
    
    # Head position
    if metrics.head_knee_xproj_norm is not None:
        cue_ok = metrics.cues.get("head_ok")
        if cue_ok is not None:
            color = GREEN if cue_ok else RED
            put_text(frame, f"Head-Knee: {metrics.head_knee_xproj_norm:.2f}", (x, y0), color)
            y0 += line_h
            cue_text = "✅ Head positioned well" if cue_ok else "❌ Move head forward"
            put_text(frame, cue_text, (x, y0), color, 0.45)
            y0 += line_h
    
    # Warnings
    if metrics.warnings:
        y0 += 5
        put_text(frame, "Warnings:", (x, y0), ORANGE, 0.5)
        y0 += line_h
        for warning in metrics.warnings[:2]:  # Show max 2 warnings
            put_text(frame, f"⚠️ {warning[:30]}...", (x, y0), ORANGE, 0.4)
            y0 += line_h

def put_text(img, text, org, color=WHITE, scale=0.5, thickness=1):
    cv2.putText(img, text, org, FONT, scale, color, thickness, cv2.LINE_AA)

def draw_enhanced_pose(frame: np.ndarray, results, keypoints: Dict):
    """Draw pose with confidence-based styling"""
    if results.pose_landmarks:
        # Draw standard pose
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=YELLOW, thickness=2, circle_radius=2),
        )
    
    # Draw confidence indicators for key joints
    key_joints = ['l_knee', 'r_knee', 'l_elbow', 'r_elbow']
    for joint_name in key_joints:
        joint = keypoints.get(joint_name)
        if joint and joint.confidence > 0:
            color = GREEN if joint.confidence > 0.7 else (ORANGE if joint.confidence > 0.4 else RED)
            cv2.circle(frame, (int(joint.x), int(joint.y)), 8, color, 2)
            # Small confidence text
            put_text(frame, f"{joint.confidence:.1f}", 
                    (int(joint.x) + 10, int(joint.y) - 10), color, 0.3)

# ----------------------------- Main Processing with Enhancements ----------------------------- #

def process_video_robust(args: argparse.Namespace):
    """Enhanced video processing with robust pose analysis"""
    
    # Initialize components
    analyzer = CricketPoseAnalyzer(args.confidence_threshold, args.smoothing_window)
    
    # Video setup (same as original)
    src_path = open_video(args.source)
    cap = cv2.VideoCapture(src_path)
    if not cap.isOpened():
        raise RuntimeError("Failed to open video")

    in_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    in_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    in_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    target_w = args.width if args.width else in_w
    scale = target_w / float(in_w)
    target_h = int(round(in_h * scale))
    target_fps = args.target_fps if args.target_fps else in_fps

    fourcc = cv2.VideoWriter_fourcc(*('mp4v'))
    out_path = os.path.join(OUTPUT_DIR, 'robust_annotated.mp4')
    writer = cv2.VideoWriter(out_path, fourcc, target_fps, (target_w, target_h))

    # Enhanced pose detection settings for cricket
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,  # Higher complexity for better accuracy
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.3,  # Lower for cricket poses
        min_tracking_confidence=0.3    # Lower for cricket poses
    )

    metrics_rows: List[RobustFrameMetrics] = []
    frame_idx = 0
    t0 = time.time()
    last_time = t0

    print("[INFO] Processing with robust cricket pose analysis...")
    print(f"[INFO] Confidence threshold: {args.confidence_threshold}")
    print(f"[INFO] Smoothing window: {args.smoothing_window}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Resize
            if target_w != in_w:
                frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

            # Process pose
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)
            
            timestamp = time.time() - t0

            if results.pose_landmarks:
                # Extract and process keypoints
                keypoints = analyzer.process_frame(
                    results.pose_landmarks.landmark, target_w, target_h, timestamp, args.stance
                )
                
                # Compute robust metrics
                metrics = compute_robust_metrics(keypoints, target_w, target_h, args.stance, analyzer)
                metrics.frame_idx = frame_idx
                metrics.timestamp = timestamp
                
                # Draw enhanced pose and overlays
                draw_enhanced_pose(frame, results, keypoints)
                draw_enhanced_cues(frame, 10, 50, metrics)
                
                # Draw reference lines for alignment
                draw_alignment_guides(frame, keypoints, metrics.front_side)
                
            else:
                # No pose detected
                metrics = RobustFrameMetrics(
                    frame_idx=frame_idx,
                    timestamp=timestamp,
                    elbow_angle=None, knee_angle=None, spine_lean_deg=None,
                    head_knee_xproj_norm=None, foot_dir_deg=None,
                    pose_quality=0.0, front_side='unknown',
                    elbow_quality=0.0, knee_quality=0.0, spine_quality=0.0,
                    cues={}, warnings=["No pose detected"]
                )
                put_text(frame, "❌ No pose detected", (10, 50), RED, 0.6)

            # Title overlay
            put_text(frame, "Robust Cricket Cover Drive Analysis", (10, 25), YELLOW, 0.7, 2)

            # Write frame
            writer.write(frame)
            metrics_rows.append(metrics)

            # Display handling
            if args.show:
                cv2.imshow('Robust Cricket Analysis', frame)
                elapsed = time.time() - last_time
                min_dt = 1.0 / max(1.0, target_fps)
                if elapsed < min_dt:
                    time.sleep(min_dt - elapsed)
                last_time = time.time()
                if (cv2.waitKey(1) & 0xFF) == ord('q'):
                    break

            frame_idx += 1

    finally:
        cap.release()
        writer.release()
        if args.show:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        pose.close()

    # Save enhanced metrics
    save_robust_results(metrics_rows, out_path)
    print(f"[INFO] Robust analysis complete! Video: {out_path}")


def draw_alignment_guides(frame: np.ndarray, keypoints: Dict, front_side: str):
    """Draw alignment reference lines"""
    try:
        h, w = frame.shape[:2]
        
        # Vertical line through front knee for head alignment
        front_knee = keypoints.get(f'{"l" if front_side == "left" else "r"}_knee')
        if front_knee and front_knee.confidence > 0.4:
            x = int(front_knee.x)
            cv2.line(frame, (x, 0), (x, h), (128, 128, 255), 1, cv2.LINE_AA)
            put_text(frame, "Knee line", (x + 5, 30), (128, 128, 255), 0.4)
        
        # Hip alignment line
        l_hip = keypoints.get('l_hip')
        r_hip = keypoints.get('r_hip')
        if l_hip and r_hip and l_hip.confidence > 0.4 and r_hip.confidence > 0.4:
            cv2.line(frame, (int(l_hip.x), int(l_hip.y)), 
                    (int(r_hip.x), int(r_hip.y)), (255, 128, 128), 1, cv2.LINE_AA)
    
    except Exception:
        pass


def save_robust_results(metrics_rows: List[RobustFrameMetrics], video_path: str):
    """Save comprehensive results with quality metrics"""
    
    # Convert to DataFrame with all metrics
    df_data = []
    for m in metrics_rows:
        row = {
            'frame_idx': m.frame_idx,
            'timestamp': m.timestamp,
            'elbow_angle': m.elbow_angle,
            'knee_angle': m.knee_angle,
            'spine_lean_deg': m.spine_lean_deg,
            'head_knee_xproj_norm': m.head_knee_xproj_norm,
            'foot_dir_deg': m.foot_dir_deg,
            'pose_quality': m.pose_quality,
            'front_side': m.front_side,
            'elbow_quality': m.elbow_quality,
            'knee_quality': m.knee_quality,
            'spine_quality': m.spine_quality,
        }
        
        # Add cue results
        for cue_name, cue_result in m.cues.items():
            row[f'cue_{cue_name}'] = cue_result
        
        # Add warning count
        row['warning_count'] = len(m.warnings)
        
        df_data.append(row)
    
    df = pd.DataFrame(df_data)
    
    # Save detailed metrics
    csv_path = os.path.join(OUTPUT_DIR, 'robust_metrics.csv')
    df.to_csv(csv_path, index=False)
    print(f"[INFO] Detailed metrics saved: {csv_path}")
    
    # Generate enhanced summary
    summary = generate_robust_summary(df, metrics_rows)
    eval_path = os.path.join(OUTPUT_DIR, 'robust_evaluation.json')
    with open(eval_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"[INFO] Enhanced evaluation saved: {eval_path}")
    
    # Generate quality report
    quality_report = generate_quality_report(df, metrics_rows)
    quality_path = os.path.join(OUTPUT_DIR, 'quality_report.json')
    with open(quality_path, 'w') as f:
        json.dump(quality_report, f, indent=2)
    print(f"[INFO] Quality report saved: {quality_path}")


def generate_robust_summary(df: pd.DataFrame, metrics_rows: List[RobustFrameMetrics]) -> Dict:
    """Generate comprehensive technique summary"""
    
    # Filter high-quality frames for analysis
    high_quality = df[df['pose_quality'] > 0.5]
    
    if len(high_quality) == 0:
        return {
            "error": "Insufficient high-quality pose data for analysis",
            "total_frames": len(df),
            "high_quality_frames": 0
        }
    
    def safe_median(series):
        clean = series.dropna()
        return float(clean.median()) if len(clean) > 0 else None
    
    # Core metrics from high-quality frames
    metrics = {
        'elbow_angle_median': safe_median(high_quality['elbow_angle']),
        'knee_angle_median': safe_median(high_quality['knee_angle']),
        'spine_lean_median': safe_median(high_quality['spine_lean_deg']),
        'head_knee_alignment': safe_median(high_quality['head_knee_xproj_norm']),
        'foot_direction_median': safe_median(high_quality['foot_dir_deg'])
    }
    
    # Scoring with quality weighting
    def weighted_score(value, good_min, good_max, hard_min, hard_max):
        if value is None:
            return 0.0
        if good_min <= value <= good_max:
            return 1.0
        if value < good_min:
            return max(0.0, (value - hard_min) / (good_min - hard_min))
        else:
            return max(0.0, (hard_max - value) / (hard_max - good_max))
    
    scores = {
        'elbow_control': weighted_score(metrics['elbow_angle_median'], 100, 150, 70, 180),
        'knee_stability': weighted_score(metrics['knee_angle_median'], 120, 170, 90, 180),
        'spine_posture': weighted_score(metrics['spine_lean_median'], 8, 22, 0, 40),
        'head_position': 1.0 - weighted_score(metrics['head_knee_alignment'], 0.0, 0.2, 0.0, 0.6),
        'foot_alignment': weighted_score(metrics['foot_direction_median'], 0, 25, 0, 60)
    }
    
    # Ensure scores are in [0,1]
    for key in scores:
        scores[key] = max(0.0, min(1.0, scores[key] or 0.0))
    
    # Convert to 1-10 scale
    def to_10_scale(score):
        return int(round(1 + 9 * score))
    
    technique_scores = {
        'Swing_Control': to_10_scale(scores['elbow_control']),
        'Footwork': to_10_scale((scores['foot_alignment'] + scores['knee_stability']) / 2),
        'Head_Position': to_10_scale(scores['head_position']),
        'Balance': to_10_scale(scores['spine_posture']),
        'Overall_Stability': to_10_scale(scores['knee_stability'])
    }
    
    # Generate specific advice
    advice = []
    if metrics['elbow_angle_median'] and (metrics['elbow_angle_median'] < 90 or metrics['elbow_angle_median'] > 160):
        advice.append("Work on maintaining front elbow elevation around 110-140° through the shot")
    
    if metrics['knee_angle_median'] and metrics['knee_angle_median'] < 120:
        advice.append("Consider less aggressive knee bend - maintain balance and power transfer")
    
    if metrics['head_knee_alignment'] and metrics['head_knee_alignment'] > 0.3:
        advice.append("Focus on getting your head position over or closer to the front knee at contact")
    
    if metrics['spine_lean_median'] and (metrics['spine_lean_median'] < 5 or metrics['spine_lean_median'] > 25):
        advice.append("Adjust your forward lean - aim for 10-15° forward body angle")
    
    return {
        'technique_scores': technique_scores,
        'raw_metrics': metrics,
        'data_quality': {
            'total_frames': len(df),
            'high_quality_frames': len(high_quality),
            'average_pose_quality': float(df['pose_quality'].mean()),
            'frames_with_elbow_data': len(df.dropna(subset=['elbow_angle'])),
            'frames_with_knee_data': len(df.dropna(subset=['knee_angle']))
        },
        'personalized_advice': advice,
        'front_side_consistency': df['front_side'].mode().iloc[0] if len(df['front_side'].mode()) > 0 else 'unknown'
    }


def generate_quality_report(df: pd.DataFrame, metrics_rows: List[RobustFrameMetrics]) -> Dict:
    """Generate pose detection quality report"""
    
    total_frames = len(df)
    if total_frames == 0:
        return {"error": "No frames processed"}
    
    # Quality statistics
    avg_quality = float(df['pose_quality'].mean())
    quality_distribution = {
        'excellent': len(df[df['pose_quality'] > 0.8]) / total_frames,
        'good': len(df[(df['pose_quality'] > 0.6) & (df['pose_quality'] <= 0.8)]) / total_frames,
        'fair': len(df[(df['pose_quality'] > 0.4) & (df['pose_quality'] <= 0.6)]) / total_frames,
        'poor': len(df[df['pose_quality'] <= 0.4]) / total_frames
    }
    
    # Joint-specific quality
    joint_quality = {
        'elbow': float(df['elbow_quality'].mean()) if 'elbow_quality' in df else 0.0,
        'knee': float(df['knee_quality'].mean()) if 'knee_quality' in df else 0.0,
        'spine': float(df['spine_quality'].mean()) if 'spine_quality' in df else 0.0
    }
    
    # Problem detection
    common_issues = []
    if joint_quality['knee'] < 0.4:
        common_issues.append("Knee tracking problematic - likely due to occlusion in cricket stance")
    if joint_quality['elbow'] < 0.5:
        common_issues.append("Elbow detection issues - may need better camera angle")
    if avg_quality < 0.5:
        common_issues.append("Overall pose quality low - consider better lighting or camera position")
    
    # Warning analysis
    all_warnings = []
    for m in metrics_rows:
        all_warnings.extend(m.warnings)
    
    warning_frequency = {}
    for warning in all_warnings:
        warning_frequency[warning] = warning_frequency.get(warning, 0) + 1
    
    return {
        'overall_quality': avg_quality,
        'quality_distribution': quality_distribution,
        'joint_quality': joint_quality,
        'data_completeness': {
            'frames_with_pose': len(df[df['pose_quality'] > 0]) / total_frames,
            'elbow_data_coverage': len(df.dropna(subset=['elbow_angle'])) / total_frames,
            'knee_data_coverage': len(df.dropna(subset=['knee_angle'])) / total_frames,
            'spine_data_coverage': len(df.dropna(subset=['spine_lean_deg'])) / total_frames
        },
        'common_issues': common_issues,
        'most_frequent_warnings': sorted(warning_frequency.items(), key=lambda x: x[1], reverse=True)[:5],
        'recommendations': [
            "For better knee tracking: ensure side-on camera angle with clear view of legs",
            "For better elbow tracking: avoid clothing that obscures arm movement",
            "For overall quality: ensure good lighting and minimize background clutter"
        ]
    }


# ----------------------------- Video I/O (same as original) ----------------------------- #

def download_video(url: str) -> str:
    if YoutubeDL is None:
        raise RuntimeError("yt-dlp not available. Install with: pip install yt-dlp")
    tmp_dir = os.path.join(OUTPUT_DIR, "downloads")
    os.makedirs(tmp_dir, exist_ok=True)
    outtmpl = os.path.join(tmp_dir, "%(id)s.%(ext)s")
    ydl_opts = {
        'outtmpl': outtmpl,
        'format': 'mp4/bestvideo+bestaudio/best',
        'merge_output_format': 'mp4',
        'quiet': True,
        'noprogress': True,
    }
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)
    if not os.path.exists(filename):
        base, _ = os.path.splitext(filename)
        alt = base + ".mp4"
        if os.path.exists(alt):
            filename = alt
    print(f"[INFO] Downloaded: {filename}")
    return filename


def open_video(source: str) -> str:
    if source.startswith('http'):
        return download_video(source)
    if not os.path.exists(source):
        raise FileNotFoundError(f"Video not found: {source}")
    return source


# ----------------------------- Enhanced CLI ----------------------------- #

def parse_args():
    p = argparse.ArgumentParser(description="Robust Cricket Cover Drive Analyzer")
    p.add_argument('--source', type=str, required=True, 
                   help='YouTube URL or local video path')
    p.add_argument('--stance', type=str, default='auto', choices=['right', 'left', 'auto'],
                   help='Batter stance (auto-detection recommended)')
    p.add_argument('--confidence_threshold', type=float, default=0.5,
                   help='Minimum confidence for keypoint acceptance (0.3-0.8)')
    p.add_argument('--smoothing_window', type=int, default=5,
                   help='Temporal smoothing window size (3-10)')
    p.add_argument('--target_fps', type=float, default=30.0,
                   help='Output video FPS')
    p.add_argument('--width', type=int, default=720,
                   help='Output width (height maintains aspect ratio)')
    p.add_argument('--show', type=int, default=0,
                   help='Display live analysis window (1=yes, 0=no)')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    print(f"[INFO] Starting robust cricket pose analysis...")
    print(f"[INFO] Source: {args.source}")
    print(f"[INFO] Stance: {args.stance}")
    print(f"[INFO] Confidence threshold: {args.confidence_threshold}")
    print(f"[INFO] Smoothing window: {args.smoothing_window}")
    
    try:
        process_video_robust(args)
        print("\n[SUCCESS] Analysis complete! Check output/ directory for results.")
        print("Files generated:")
        print("  - robust_annotated.mp4: Video with enhanced overlays")
        print("  - robust_metrics.csv: Frame-by-frame data with quality scores")
        print("  - robust_evaluation.json: Technique assessment")
        print("  - quality_report.json: Pose detection quality analysis")
        
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)