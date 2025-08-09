from collections import defaultdict
import numpy as np

class Tracker:
    def __init__(self):
        self.tracked_objects = {}
        self.next_id = 0
        self.max_distance = 50  # pixels
        
    def update(self, detections):
        # Simple tracking based on bbox proximity
        updated_tracks = {}
        used_detections = set()
        
        # Match existing tracks
        for track_id, track in self.tracked_objects.items():
            best_match = None
            min_dist = float('inf')
            
            for i, det in enumerate(detections):
                if i in used_detections:
                    continue
                    
                dist = self._bbox_distance(track['bbox'], det['bbox'])
                if dist < self.max_distance and dist < min_dist:
                    min_dist = dist
                    best_match = i
            
            if best_match is not None:
                used_detections.add(best_match)
                updated_tracks[track_id] = {
                    'bbox': detections[best_match]['bbox'],
                    'frames': track['frames'] + 1
                }
        
        # Add new detections
        for i, det in enumerate(detections):
            if i not in used_detections:
                updated_tracks[self.next_id] = {
                    'bbox': det['bbox'],
                    'frames': 1
                }
                self.next_id += 1
                
        self.tracked_objects = updated_tracks
        return updated_tracks
    
    def _bbox_distance(self, box1, box2):
        # Calculate center point distance
        c1 = np.array([(box1[0]+box1[2])/2, (box1[1]+box1[3])/2])
        c2 = np.array([(box2[0]+box2[2])/2, (box2[1]+box2[3])/2])
        return np.linalg.norm(c1 - c2)