import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

class Visualizer:
    def __init__(self):
        self.fig = None
        
    def create_3d_pose(self, keypoints, frame_idx):
        if not keypoints:
            return None
            
        # Create 3D scatter plot
        fig = go.Figure()
        
        # Add skeleton connections
        connections = [
            ('LEFT_HIP', 'LEFT_KNEE'), ('LEFT_KNEE', 'LEFT_ANKLE'),
            ('RIGHT_HIP', 'RIGHT_KNEE'), ('RIGHT_KNEE', 'RIGHT_ANKLE'),
            ('LEFT_SHOULDER', 'LEFT_ELBOW'), ('LEFT_ELBOW', 'LEFT_WRIST'),
            ('RIGHT_SHOULDER', 'RIGHT_ELBOW'), ('RIGHT_ELBOW', 'RIGHT_WRIST'),
            ('LEFT_SHOULDER', 'RIGHT_SHOULDER'),
            ('LEFT_HIP', 'RIGHT_HIP'), ('LEFT_SHOULDER', 'LEFT_HIP'),
            ('RIGHT_SHOULDER', 'RIGHT_HIP')
        ]
        
        for start, end in connections:
            fig.add_trace(go.Scatter3d(
                x=[keypoints[start]['x'], keypoints[end]['x']],
                y=[keypoints[start]['y'], keypoints[end]['y']],
                z=[keypoints[start]['z'], keypoints[end]['z']],
                mode='lines',
                line=dict(width=4, color='blue')
            ))
        
        # Add keypoints
        x = [kp['x'] for kp in keypoints.values()]
        y = [kp['y'] for kp in keypoints.values()]
        z = [kp['z'] for kp in keypoints.values()]
        
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(size=5, color='red')
        ))
        
        fig.update_layout(
            title=f'Player Pose - Frame {frame_idx}',
            scene=dict(
                xaxis=dict(title='X', range=[-1, 1]),
                yaxis=dict(title='Y', range=[-1, 1]),
                zaxis=dict(title='Z', range=[-1, 1]),
                aspectmode='cube'
            ),
            showlegend=False
        )
        
        return fig
    
    def create_tactical_view(self, players, ball, frame_idx):
        fig = go.Figure()
        
        # Draw pitch outline
        fig.add_shape(type="rect",
            x0=0, y0=0, x1=105, y1=68,
            line=dict(color="Green"),
            fillcolor="LightGreen"
        )
        
        # Add players
        for i, player in enumerate(players):
            x = (player['bbox'][0] + player['bbox'][2]) / 2
            y = (player['bbox'][1] + player['bbox'][3]) / 2
            
            fig.add_trace(go.Scatter(
                x=[x], y=[y],
                mode='markers',
                marker=dict(size=10, color='blue'),
                text=f"Player {i}",
                hoverinfo='text'
            ))
        
        # Add ball
        x_ball = (ball['bbox'][0] + ball['bbox'][2]) / 2
        y_ball = (ball['bbox'][1] + ball['bbox'][3]) / 2
        fig.add_trace(go.Scatter(
            x=[x_ball], y=[y_ball],
            mode='markers',
            marker=dict(size=8, color='white'),
            text="Ball",
            hoverinfo='text'
        ))
        
        fig.update_layout(
            title=f'Tactical View - Frame {frame_idx}',
            xaxis=dict(range=[0, 105], scaleanchor="y", scaleratio=1),
            yaxis=dict(range=[0, 68]),
            showlegend=False
        )
        
        return fig