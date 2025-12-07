"""
KeyPoints Extractor for sign language recognition.
Extracts comprehensive hand features including 3D coordinates, velocities, angles, and shape descriptors.
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Tuple
from collections import deque
from scipy.spatial.distance import euclidean

from config import MEDIAPIPE_CONFIG


class KeyPointsExtractor:
    """
    Extract comprehensive hand features from video frames.
    
    Features extracted (240 total):
    - 3D normalized keypoints (63): x, y, z for 21 landmarks
    - Geometric features (35): angles, distances, lengths
    - Shape features (15): palm area, aspect ratio, spread, etc.
    - Temporal features (126): velocity and acceleration
    - Handedness (1): left/right hand indicator
    """
    
    # Hand landmark connections for angle calculation
    FINGER_CONNECTIONS = {
        'thumb': [(0, 1), (1, 2), (2, 3), (3, 4)],
        'index': [(0, 5), (5, 6), (6, 7), (7, 8)],
        'middle': [(0, 9), (9, 10), (10, 11), (11, 12)],
        'ring': [(0, 13), (13, 14), (14, 15), (15, 16)],
        'pinky': [(0, 17), (17, 18), (18, 19), (19, 20)]
    }
    
    def __init__(self, history_size: int = 5):
        """
        Initialize the MediaPipe hands module.
        
        Args:
            history_size: Number of frames to keep for temporal features
        """
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=MEDIAPIPE_CONFIG["max_num_hands"],
            min_detection_confidence=MEDIAPIPE_CONFIG["min_detection_confidence"],
            min_tracking_confidence=MEDIAPIPE_CONFIG["min_tracking_confidence"]
        )
        
        # History for temporal features
        self.history_size = history_size
        self.keypoints_history = deque(maxlen=history_size)
        
    def extract_keypoints(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract comprehensive hand features from a video frame.
        
        Args:
            frame: Video frame (BGR format)
            
        Returns:
            Tuple of (processed_frame, features)
            - processed_frame: Frame with keypoints drawn
            - features: 240-dim feature vector
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.hands.process(frame_rgb)
        
        # Initialize feature vector
        features = np.zeros(240)
        
        # Extract features if hands are detected
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            handedness = results.multi_handedness[0].classification[0].label
            
            # Draw landmarks
            self.mp_drawing.draw_landmarks(
                frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
            
            # Extract features
            raw_keypoints = self._extract_raw_keypoints(hand_landmarks)
            spatial_features = self._extract_spatial_features(raw_keypoints)
            geometric_features = self._extract_geometric_features(raw_keypoints)
            shape_features = self._extract_shape_features(raw_keypoints)
            
            # Store in history
            self.keypoints_history.append(raw_keypoints)
            temporal_features = self._extract_temporal_features()
            
            # Handedness (0=left, 1=right)
            handedness_feature = 1.0 if handedness == "Right" else 0.0
            
            # Combine all features
            features = np.concatenate([
                spatial_features,      # 63
                geometric_features,    # 35
                shape_features,        # 15
                temporal_features,     # 126
                [handedness_feature]   # 1
            ])
        else:
            # No hand detected
            self.keypoints_history.append(np.zeros(63))
        
        return frame, features
    
    def _extract_raw_keypoints(self, hand_landmarks) -> np.ndarray:
        """Extract raw 3D keypoints (21 landmarks × 3 coords = 63)."""
        keypoints = np.zeros(63)
        for i, landmark in enumerate(hand_landmarks.landmark):
            keypoints[i*3] = landmark.x
            keypoints[i*3 + 1] = landmark.y
            keypoints[i*3 + 2] = landmark.z
        return keypoints
    
    def _extract_spatial_features(self, raw_keypoints: np.ndarray) -> np.ndarray:
        """
        Extract normalized spatial features (63).
        
        Normalizations:
        - Translation: Center at wrist
        - Rotation: Align palm plane
        - Scale: Normalize by palm size
        """
        if np.all(raw_keypoints == 0):
            return raw_keypoints
        
        points = raw_keypoints.reshape(21, 3)
        
        # 1. Translation: center at wrist
        wrist = points[0]
        centered = points - wrist
        
        # 2. Scale: normalize by palm size
        middle_mcp = centered[9]
        palm_size = np.linalg.norm(middle_mcp)
        scaled = centered / palm_size if palm_size > 1e-6 else centered
        
        # 3. Rotation: align palm plane
        index_mcp, pinky_mcp = scaled[5], scaled[17]
        v1, v2 = index_mcp - scaled[0], pinky_mcp - scaled[0]
        normal = np.cross(v1, v2)
        normal_mag = np.linalg.norm(normal)
        
        if normal_mag > 1e-6:
            normal = normal / normal_mag
            target = np.array([0, 0, 1])
            rotation_axis = np.cross(normal, target)
            axis_mag = np.linalg.norm(rotation_axis)
            
            if axis_mag > 1e-6:
                rotation_axis = rotation_axis / axis_mag
                angle = np.arccos(np.clip(np.dot(normal, target), -1.0, 1.0))
                rotated = self._rotate_points(scaled, rotation_axis, angle)
            else:
                rotated = scaled
        else:
            rotated = scaled
        
        return rotated.reshape(63)
    
    def _rotate_points(self, points: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
        """Rotate points using Rodrigues' formula."""
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rotated = (points * cos_a + 
                  np.cross(axis, points) * sin_a + 
                  axis * np.dot(points, axis)[:, np.newaxis] * (1 - cos_a))
        return rotated
    
    def _extract_geometric_features(self, raw_keypoints: np.ndarray) -> np.ndarray:
        """
        Extract geometric features (35).
        - 20 angles between finger joints
        - 10 distances from palm to fingertips
        - 5 finger lengths
        """
        if np.all(raw_keypoints == 0):
            return np.zeros(35)
        
        points = raw_keypoints.reshape(21, 3)
        features = []
        
        # Angles between joints
        for finger, connections in self.FINGER_CONNECTIONS.items():
            for i in range(len(connections) - 1):
                p1 = points[connections[i][0]]
                p2 = points[connections[i][1]]
                p3 = points[connections[i+1][1]]
                angle = self._calculate_angle(p1, p2, p3)
                features.append(angle)
        
        # Distances from palm center to fingertips
        palm_center = np.mean(points[[0, 5, 9, 13, 17]], axis=0)
        fingertips = [4, 8, 12, 16, 20]
        for tip in fingertips:
            features.append(euclidean(palm_center, points[tip]))
        
        # Finger lengths
        mcps = [1, 5, 9, 13, 17]
        for mcp, tip in zip(mcps, fingertips):
            features.append(euclidean(points[mcp], points[tip]))
        
        # Inter-finger distances
        features.extend([
            euclidean(points[4], points[8]),   # Thumb-Index
            euclidean(points[8], points[12]),  # Index-Middle
            euclidean(points[12], points[16]), # Middle-Ring
            euclidean(points[16], points[20]), # Ring-Pinky
            euclidean(points[4], points[20])   # Thumb-Pinky
        ])
        
        return np.array(features)
    
    def _calculate_angle(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """Calculate angle between three points (p1-p2-p3)."""
        v1, v2 = p1 - p2, p3 - p2
        v1_norm, v2_norm = np.linalg.norm(v1), np.linalg.norm(v2)
        
        if v1_norm < 1e-6 or v2_norm < 1e-6:
            return 0.0
        
        cos_angle = np.clip(np.dot(v1/v1_norm, v2/v2_norm), -1.0, 1.0)
        return np.arccos(cos_angle)
    
    def _extract_shape_features(self, raw_keypoints: np.ndarray) -> np.ndarray:
        """
        Extract hand shape features (15).
        - Palm area
        - Palm aspect ratio
        - Finger spread (4 angles)
        - Hand openness
        - Bounding box features (4)
        - Compactness
        """
        if np.all(raw_keypoints == 0):
            return np.zeros(15)
        
        points = raw_keypoints.reshape(21, 3)
        features = []
        
        # Palm area (convex hull)
        palm_indices = [0, 1, 5, 9, 13, 17]
        palm_2d = points[palm_indices, :2]
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(palm_2d)
            palm_area = hull.volume
        except:
            palm_area = 0.0
        features.append(palm_area)
        
        # Palm aspect ratio
        palm_width = euclidean(points[5, :2], points[17, :2])
        palm_height = euclidean(points[0, :2], points[9, :2])
        features.append(palm_width / palm_height if palm_height > 1e-6 else 0.0)
        
        # Finger spread angles
        fingertips = [4, 8, 12, 16, 20]
        for i in range(len(fingertips) - 1):
            angle = self._calculate_angle(points[fingertips[i]], points[0], points[fingertips[i+1]])
            features.append(angle)
        
        # Hand openness
        palm_center = np.mean(points[[0, 5, 9, 13, 17]], axis=0)
        avg_distance = np.mean([euclidean(palm_center, points[t]) for t in fingertips])
        features.append(avg_distance)
        
        # Bounding box
        bbox_w = np.max(points[:, 0]) - np.min(points[:, 0])
        bbox_h = np.max(points[:, 1]) - np.min(points[:, 1])
        bbox_d = np.max(points[:, 2]) - np.min(points[:, 2])
        bbox_aspect = bbox_w / bbox_h if bbox_h > 1e-6 else 0.0
        features.extend([bbox_w, bbox_h, bbox_d, bbox_aspect])
        
        # Compactness
        perimeter = sum([euclidean(points[fingertips[i], :2], points[fingertips[(i+1)%5], :2]) 
                        for i in range(5)])
        compactness = (4 * np.pi * palm_area) / (perimeter ** 2) if perimeter > 1e-6 else 0.0
        features.append(compactness)
        
        return np.array(features)
    
    def _extract_temporal_features(self) -> np.ndarray:
        """
        Extract temporal features (126).
        - Velocity: 63 (first derivative)
        - Acceleration: 63 (second derivative)
        """
        if len(self.keypoints_history) < 2:
            return np.zeros(126)
        
        history = np.array(list(self.keypoints_history))
        
        # Velocity
        velocity = history[-1] - history[-2] if len(history) >= 2 else np.zeros(63)
        
        # Acceleration
        acceleration = (history[-1] - 2*history[-2] + history[-3]) if len(history) >= 3 else np.zeros(63)
        
        return np.concatenate([velocity, acceleration])
    
    def reset_history(self):
        """Reset the keypoint history."""
        self.keypoints_history.clear()
    
    def release(self):
        """Release resources."""
        self.hands.close()
        self.keypoints_history.clear()