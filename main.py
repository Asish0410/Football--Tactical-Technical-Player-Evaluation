import cv2
import numpy as np
import yaml
import os
from tqdm import tqdm
from detection import PlayerDetector
from tracking import Tracker
from pose_estmation import PoseAnalyzer
from tactical import TacticalAnalyzer
from visualization import Visualizer

class FootballAnalyzer:
    def __init__(self, config_path='config.yaml'):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
            
        self.detector = PlayerDetector()
        self.tracker = Tracker()
        self.pose_analyzer = PoseAnalyzer()
        self.tactical_analyzer = TacticalAnalyzer()
        self.visualizer = Visualizer()
        
    def analyze_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        results = []
        
        for frame_idx in tqdm(range(total_frames)):
            ret, frame = cap.read()
            if not ret:
                break
                
            # Player detection and tracking
            detections = self.detector.detect(frame)
            tracks = self.tracker.update(detections)
            
            # Find ball (simplified - would use specialized detector in practice)
            ball = self._find_ball(frame, tracks)
            
            # Analyze each player
            frame_results = []
            for track_id, player in tracks.items():
                # Crop player from frame
                x1, y1, x2, y2 = player['bbox']
                player_img = frame[y1:y2, x1:x2]
                
                # Pose estimation
                pose_analysis = self.pose_analyzer.analyze_frame(player_img)
                
                frame_results.append({
                    'track_id': track_id,
                    'bbox': player['bbox'],
                    'pose': pose_analysis
                })
            
            # Tactical analysis
            tactical_analysis = self.tactical_analyzer.analyze_positions(
                [p for p in tracks.values()], ball)
            
            # Generate visualizations for key frames
            if frame_idx % self.config['visualization_interval'] == 0:
                self._generate_visualizations(
                    frame, frame_idx, frame_results, tactical_analysis)
            
            results.append({
                'frame': frame_idx,
                'time': frame_idx / fps,
                'players': frame_results,
                'tactical': tactical_analysis,
                'ball': ball
            })
            
        cap.release()
        return results
    
    def _find_ball(self, frame, players):
        # Simplified ball detection - would use specialized CNN in production
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                                 param1=50, param2=30, minRadius=5, maxRadius=30)
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for circle in circles[0, :]:
                x, y, r = circle
                # Check if circle is not inside any player bbox
                is_ball = True
                for player in players.values():
                    px1, py1, px2, py2 = player['bbox']
                    if px1 <= x <= px2 and py1 <= y <= py2:
                        is_ball = False
                        break
                
                if is_ball:
                    return {
                        'bbox': [x-r, y-r, x+r, y+r],
                        'confidence': 0.8  # Placeholder
                    }
        
        return {'bbox': [0, 0, 0, 0], 'confidence': 0}
    
    def _generate_visualizations(self, frame, frame_idx, players, tactical):
        # Create output directory if needed
        os.makedirs('output/visualizations', exist_ok=True)
        
        # Generate 3D pose visualization for each player
        for player in players:
            if player['pose']:
                fig = self.visualizer.create_3d_pose(
                    player['pose']['keypoints'], frame_idx)
                fig.write_html(
                    f"output/visualizations/pose_{frame_idx}_{player['track_id']}.html")
        
        # Generate tactical view
        fig = self.visualizer.create_tactical_view(
            [p for p in players], 
            self._find_ball(frame, {i: p for i, p in enumerate(players)}), 
            frame_idx
        )
        fig.write_html(f"output/visualizations/tactical_{frame_idx}.html")
        
        # Save annotated frame
        annotated = frame.copy()
        for player in players:
            x1, y1, x2, y2 = map(int, player['bbox'])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, str(player['track_id']), (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        ball = self._find_ball(frame, {i: p for i, p in enumerate(players)})
        if ball['confidence'] > 0.5:
            x1, y1, x2, y2 = map(int, ball['bbox'])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(annotated, "Ball", (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
        cv2.imwrite(f"output/visualizations/frame_{frame_idx}.jpg", annotated)

if __name__ == "__main__":
    analyzer = FootballAnalyzer()
    video_path = "data/Video-2.mp4"  # Replace with your video
    results = analyzer.analyze_video(video_path)
    
    # Save results
    import json
    def convert(obj):
        import numpy as np
        if isinstance(obj, (np.integer, np.int32, np.int64, np.uint16)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open("output/results.json", "w") as f:
        json.dump(results, f, default=convert)