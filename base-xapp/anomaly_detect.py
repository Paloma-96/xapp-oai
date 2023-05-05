import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pyod.models.knn import KNN
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib
import sqlite3

matplotlib.use('TkAgg')

# Read data

# Connect to the SQLite database
conn = sqlite3.connect('/home/ipalama/work/OAI-colosseum-ric-integration-paloma/xapp-oai/base-xapp/toa_measurements.db')
cur = conn.cursor()
cur.execute("SELECT * FROM measurements")
rows = cur.fetchall()
conn.close()

# Convert fetched data to a suitable format (e.g., Pandas DataFrame)
df_raw = pd.DataFrame(rows, columns=['gnb_id', 'rnti', 'toa_val', 'snr', 'timestamp'])

df = df_raw.dropna()

unique_rntis = df['rnti'].unique()
timestamps = df['timestamp'].unique()

new_rows = []

for rnti in unique_rntis:
    for timestamp in timestamps:
        existing_row = df.loc[(df['rnti'] == rnti) & (df['timestamp'] == timestamp)].iloc[0]
        
        for gnb_id in [1, 2]:
            new_row = {
                'gnb_id': gnb_id,
                'rnti': rnti,
                'toa_val': existing_row['toa_val'] + np.random.normal(0, 5),  # Add small random noise to the existing toa_val
                'snr': existing_row['snr'] + np.random.normal(0, 5),  # Add small random noise to the existing snr
                'timestamp': timestamp
            }
            new_rows.append(new_row)

new_rows_df = pd.DataFrame(new_rows)

df = pd.concat([df, new_rows_df], ignore_index=True)  # Add new rows to the df DataFrame

print(df)

df[['gnb_id', 'rnti']] = df[['gnb_id', 'rnti']].astype(int)
df[['toa_val', 'snr']] = df[['toa_val', 'snr']].astype(float)
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Anomaly detection
detector = KNN()
detector.fit(df[['toa_val', 'snr']])
outliers = detector.predict(df[['toa_val', 'snr']])

# User positioning
time_window = timedelta(seconds=1)

def triangulate(gnb_positions, toa_vals):
    def objective(x, gnb_positions, toa_vals):
        return sum([(np.linalg.norm(pos - x) - toa) ** 2 for pos, toa in zip(gnb_positions, toa_vals)])

    initial_guess = np.mean(gnb_positions, axis=0)
    result = minimize(objective, initial_guess, args=(gnb_positions, toa_vals))
    return result.x

gnb_positions = {
    0: np.array([0, 0]),
    1: np.array([100, 0]),
    2: np.array([50, 100])
}

# Plotting
fig, ax = plt.subplots()

# Plot gNB positions
for gnb_id, pos in gnb_positions.items():
    plt.scatter(pos[0], pos[1], label=f'gNB {gnb_id}', marker='^', s=150)

# Plot UE positions
for timestamp, group in df.groupby('timestamp'):
    if len(group) >= 3:
        toa_vals = group['toa_val'].values
        gnb_ids = group['gnb_id'].values
        gnb_coords = [gnb_positions[id] for id in gnb_ids]
        user_position = triangulate(gnb_coords, toa_vals)
        print(f"User position at {timestamp}: {user_position}")
        plt.scatter(user_position[0], user_position[1], label='UE', marker='o', s=150)


plt.xlabel('X')
plt.ylabel('Y')
# set axes range
plt.xlim([-10, 110])
plt.ylim([-10, 110])
plt.legend()
plt.grid()
plt.show()