import numpy as np
from scipy.spatial import Voronoi

class TacticalAnalyzer:
    def __init__(self, pitch_dimensions=(105, 68)):
        self.pitch_width, self.pitch_length = pitch_dimensions
        
    def analyze_positions(self, players, ball):
        analysis = {}
        
        # Convert to pitch coordinates (normalized 0-1)
        norm_players = self._normalize_positions(players)
        norm_ball = self._normalize_position(ball)
        
        # Space control analysis
        analysis['space_control'] = self._calculate_space_control(norm_players)
        
        # Passing lanes
        analysis['passing_options'] = self._find_passing_lanes(
            norm_players, norm_ball)
            
        return analysis
    
    def _normalize_positions(self, players):
        # Accepts list of dicts with 'bbox' or direct bbox lists
        normed = []
        for p in players:
            bbox = p['bbox'] if isinstance(p, dict) and 'bbox' in p else p
            normed.append(self._normalize_position(bbox))
        return normed
    
    def _normalize_position(self, bbox):
        # Accepts dict with 'bbox' or direct bbox list/tuple
        if isinstance(bbox, dict) and 'bbox' in bbox:
            bbox = bbox['bbox']
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            return {'x': 0, 'y': 0}
        x_center = (bbox[0] + bbox[2]) / 2
        y_center = (bbox[1] + bbox[3]) / 2
        return {
            'x': x_center / self.pitch_length,
            'y': y_center / self.pitch_width
        }
    
    def _calculate_space_control(self, players):
        points = np.array([[p['x'], p['y']] for p in players])
        if points.shape[0] < 3:
            return []  # Not enough points for Voronoi
        vor = Voronoi(points)
        # Calculate area for each player's Voronoi cell
        areas = []
        for i, region in enumerate(vor.regions):
            if not region or -1 in region:
                continue
            poly = [vor.vertices[j] for j in region]
            areas.append(self._polygon_area(poly))
        return areas
    
    def _find_passing_lanes(self, players, ball):
        passing_options = []
        
        for i, receiver in enumerate(players):
            # Simple check - no defenders between ball and receiver
            line_clear = True
            for j, defender in enumerate(players):
                if i == j:
                    continue
                if self._point_to_line_distance(
                    defender, ball, receiver) < 0.1:  # Threshold
                    line_clear = False
                    break
                    
            if line_clear:
                passing_options.append({
                    'receiver': i,
                    'distance': self._distance(ball, receiver)
                })
                
        return passing_options
    
    def _polygon_area(self, vertices):
        x = [v[0] for v in vertices]
        y = [v[1] for v in vertices]
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    
    def _distance(self, p1, p2):
        return np.sqrt((p1['x']-p2['x'])**2 + (p1['y']-p2['y'])**2)
    
    def _point_to_line_distance(self, point, line_p1, line_p2):
        # Distance from point to line segment
        x, y = point['x'], point['y']
        x1, y1 = line_p1['x'], line_p1['y']
        x2, y2 = line_p2['x'], line_p2['y']
        
        numerator = abs((y2-y1)*x - (x2-x1)*y + x2*y1 - y2*x1)
        denominator = np.sqrt((y2-y1)**2 + (x2-x1)**2)
        return numerator / denominator