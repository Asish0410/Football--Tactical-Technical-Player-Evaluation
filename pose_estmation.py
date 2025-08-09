import mediapipe as mp
import numpy as np

class PoseAnalyzer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.7
        )
        
    def analyze_frame(self, frame):
        results = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        if not results.pose_landmarks:
            return None
            
        landmarks = results.pose_landmarks.landmark
        keypoints = {}
        
        # Convert landmarks to more usable format
        for i, landmark in enumerate(landmarks):
            keypoints[self.mp_pose.PoseLandmark(i).name] = {
                'x': landmark.x,
                'y': landmark.y,
                'z': landmark.z,
                'visibility': landmark.visibility
            }
        
        return {
            'keypoints': keypoints,
            'analysis': self._analyze_pose(keypoints)
        }
    
    def _analyze_pose(self, keypoints):
        analysis = {}
        
        # Body lean analysis
        shoulder_avg = self._avg_point(keypoints['LEFT_SHOULDER'], 
                                      keypoints['RIGHT_SHOULDER'])
        hip_avg = self._avg_point(keypoints['LEFT_HIP'], 
                                 keypoints['RIGHT_HIP'])
        analysis['body_lean'] = shoulder_avg['x'] - hip_avg['x']
        
        # Hip-shoulder separation (for torque analysis)
        left_hip_shoulder = self._distance(
            keypoints['LEFT_HIP'], keypoints['LEFT_SHOULDER'])
        right_hip_shoulder = self._distance(
            keypoints['RIGHT_HIP'], keypoints['RIGHT_SHOULDER'])
        analysis['hip_torque'] = abs(left_hip_shoulder - right_hip_shoulder)
        
        return analysis
    
    def _avg_point(self, p1, p2):
        return {
            'x': (p1['x'] + p2['x']) / 2,
            'y': (p1['y'] + p2['y']) / 2,
            'z': (p1['z'] + p2['z']) / 2
        }
    
    def _distance(self, p1, p2):
        return np.sqrt(
            (p1['x']-p2['x'])**2 + 
            (p1['y']-p2['y'])**2 + 
            (p1['z']-p2['z'])**2
        )