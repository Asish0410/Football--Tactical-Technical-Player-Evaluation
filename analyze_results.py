import json
import pandas as pd
import matplotlib.pyplot as plt

# Load results
with open('output/results.json', 'r') as f:
    results = json.load(f)

# Extract statistics
frames = []
for frame in results:
    frame_idx = frame.get('frame')
    time = frame.get('time')
    players = frame.get('players', [])
    ball = frame.get('ball', {})
    tactical = frame.get('tactical', {})
    player_count = len(players)
    ball_x = ball.get('bbox', [None, None, None, None])[0]
    ball_y = ball.get('bbox', [None, None, None, None])[1]
    space_control = tactical.get('space_control', [])
    passing_options = tactical.get('passing_options', [])
    frames.append({
        'frame': frame_idx,
        'time': time,
        'player_count': player_count,
        'ball_x': ball_x,
        'ball_y': ball_y,
        'space_control': space_control,
        'passing_options': len(passing_options)
    })

df = pd.DataFrame(frames)

# Save to CSV and Excel
csv_path = 'output/results_summary.csv'
excel_path = 'output/results_summary.xlsx'
df.to_csv(csv_path, index=False)
df.to_excel(excel_path, index=False)

# Plot player count per frame
df.plot(x='frame', y='player_count', kind='line', title='Player Count per Frame')
plt.xlabel('Frame')
plt.ylabel('Number of Players')
plt.tight_layout()
plt.savefig('output/player_count_per_frame.png')
plt.close()

# Plot ball trajectory
plt.plot(df['ball_x'], df['ball_y'], marker='o')
plt.title('Ball Trajectory')
plt.xlabel('Ball X')
plt.ylabel('Ball Y')
plt.tight_layout()
plt.savefig('output/ball_trajectory.png')
plt.close()

# Plot tactical metrics: number of passing options per frame
df.plot(x='frame', y='passing_options', kind='line', title='Passing Options per Frame')
plt.xlabel('Frame')
plt.ylabel('Number of Passing Options')
plt.tight_layout()
plt.savefig('output/passing_options_per_frame.png')
plt.close()

print('Summary CSV, Excel, and plots saved in output/.')
