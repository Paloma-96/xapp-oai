import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pyod.models.knn import KNN
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib
import sqlite3
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


matplotlib.use('TkAgg')

Spoof_magnitude = 30

def DBConnect():
    conn = sqlite3.connect('/home/ipalama/work/OAI-colosseum-ric-integration-paloma/xapp-oai/base-xapp/toa_measurements.db')
    return conn

def populateWithGaussValues(df):
    # Get the unique rntis
    unique_rntis = df['rnti'].unique()
    # Get the unique timestamps
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
    return df

def populateWithSpoofedValues(df):
    # Get the unique rntis
    unique_rntis = df['rnti'].unique()
    # Get the unique timestamps
    timestamps = df['timestamp'].unique()
    new_rows = []
    spoofed_gnb_id = 0
    for rnti in unique_rntis:
        for timestamp in timestamps:
            existing_row = df.loc[(df['rnti'] == rnti) & (df['timestamp'] == timestamp)].iloc[0]

            for gnb_id in [0, 1, 2]:
                # set the spoofed values for the toa_val and snr when timestamp is after the half of the total time
                #
                if timestamp > timestamps[len(timestamps) // 2] and gnb_id == spoofed_gnb_id:
                    new_row = {
                        'gnb_id': gnb_id,
                        'rnti': rnti,
                        'toa_val': existing_row['toa_val'] + np.random.normal(Spoof_magnitude, 1),  # Add small random noise to the existing toa_val
                        'snr': existing_row['snr'] + np.random.normal(3, 1),  # Add small random noise to the existing snr
                        'timestamp': timestamp
                    }
                    new_rows.append(new_row)
                else:
                    new_row = {
                        'gnb_id': gnb_id,
                        'rnti': rnti,
                        'toa_val': existing_row['toa_val'] + np.random.normal(0, 1),  # Add small random noise to the existing toa_val
                        'snr': existing_row['snr'] + np.random.normal(0, 1),  # Add small random noise to the existing snr
                        'timestamp': timestamp
                    }
                    new_rows.append(new_row)

    new_rows_df = pd.DataFrame(new_rows)
    #df = pd.concat([df, new_rows_df], ignore_index=True)  # Add new rows to the df DataFrame
    return new_rows_df

def triangulate(gnb_positions, toa_vals):
    def objective(x, gnb_positions, toa_vals):
        return sum([(np.linalg.norm(pos - x) - toa) ** 2 for pos, toa in zip(gnb_positions, toa_vals)])

    initial_guess = np.mean(gnb_positions, axis=0)
    result = minimize(objective, initial_guess, args=(gnb_positions, toa_vals))
    return result.x

def generate_alert(real_value, predicted_value, threshold):
    if abs(real_value - predicted_value) > threshold:
        return f"Alert: Real value ({real_value}) differs from predicted value ({predicted_value}) by more than {threshold}"
    return None

# Connect to the SQLite database
conn = DBConnect()
cur = conn.cursor()
cur.execute("SELECT * FROM measurements")
rows = cur.fetchall()
conn.close()

# Convert fetched data to a suitable format (e.g., Pandas DataFrame)
df_raw = pd.DataFrame(rows, columns=['gnb_id', 'rnti', 'toa_val', 'snr', 'timestamp'])

df = df_raw.dropna()

#Apply effect
#df = populateWithGaussValues(df)
df = populateWithSpoofedValues(df)

print(df)

df[['gnb_id', 'rnti']] = df[['gnb_id', 'rnti']].astype(int)
df[['toa_val', 'snr']] = df[['toa_val', 'snr']].astype(float)
df['timestamp'] = pd.to_datetime(df['timestamp'])

# apply datetime.strptime to convert the timestamp column to datetime
df['timestamp'] = df['timestamp'].apply(lambda x: datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S.%f'))

# User positioning
time_window = 5  # seconds

window_size = 5
threshold = 5

def xgboost_model(train_data, next_row):
    X_train = train_data[['gnb_id', 'snr']].values
    y_train = train_data['toa_val'].values
    X_test = next_row[['gnb_id', 'snr']].values.reshape(1, -1)

    model = xgb.XGBRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)[0]

    return y_pred

def lstm_model(train_data, next_row):
    X_train = train_data[['gnb_id', 'snr']].values.reshape(-1, 1, 2).astype('float32')
    y_train = train_data['toa_val'].values.astype('float32')
    X_test = next_row[['gnb_id', 'snr']].values.reshape(1, 1, 2).astype('float32')

    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(1, 2)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=100, verbose=0)

    y_pred = model.predict(X_test)[0][0]

    return y_pred

def linear_regression_model(train_data, next_row):
    X_train = train_data[['gnb_id', 'snr']].values
    y_train = train_data['toa_val'].values
    X_test = next_row[['gnb_id', 'snr']].values.reshape(1, -1)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)[0]

    return y_pred

alert_indexes = []

for i in range(window_size, len(df)):
    train_data = df.iloc[i - window_size:i]
    train_data = train_data[~train_data.index.isin(alert_indexes)]  # Exclude rows that generated alerts
    
    if len(train_data) == 0:
        continue

    next_row = df.iloc[i]
    # Choose the ML algorithm: xgboost_model or lstm_model
    y_pred = lstm_model(train_data, next_row)
    
    y_test = next_row['toa_val']

    alert = generate_alert(y_test, y_pred, threshold)
    if alert:
        print(alert)
        alert_indexes.append(i)  # Add the index of the row that generated an alert


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

timestamps_done = []
# check the toa_val and snr values in a 5 seconds time window, if you find a difference of both values > 3, then print an alert
for gnb_id, group in df.groupby('gnb_id'):
    anomaly_index = -1
    #print(gnb_id)
    if len(group) >= 0:
        for i in range(len(group)):

            # check if it's the last one
            if i == 0:
                continue
            
            if anomaly_index != -1:
                # check if the difference between the timestamps is less than 5 seconds
                diff = (group.iloc[i]['timestamp'] - group.iloc[anomaly_index]['timestamp']).seconds
                if diff <= time_window:
                    toa_val_diff = abs(group.iloc[anomaly_index - 1]['toa_val'] - group.iloc[i]['toa_val'])
                    snr_diff = abs(group.iloc[anomaly_index - 1]['snr'] - group.iloc[i]['snr'])

            else:
                diff = (group.iloc[i]['timestamp'] - group.iloc[i-1]['timestamp']).seconds
                if diff <= time_window:
                    toa_val_diff = abs(group.iloc[i]['toa_val'] - group.iloc[i - 1]['toa_val'])
                    snr_diff = abs(group.iloc[i]['snr'] - group.iloc[i - 1]['snr']) 

            if group.iloc[i]['timestamp'] not in timestamps_done:
                timestamps_done.append(group.iloc[i]['timestamp'])

                toa_vals = df.loc[df['timestamp'] == group.iloc[i]['timestamp']]['toa_val'].values
                gnb_ids = df.loc[df['timestamp'] == group.iloc[i]['timestamp']]['gnb_id'].values
                gnb_coords = [gnb_positions[id] for id in gnb_ids]
                user_position = triangulate(gnb_coords, toa_vals)
                print(f"User position at {gnb_id}: {user_position}")

                if toa_val_diff > 5:
                    if anomaly_index == -1:
                        anomaly_index = i
                    print(f"ALERT: gNB {gnb_id}: anomaly in the toa_val and snr values at {group.iloc[i]['timestamp']}")
                    plt.scatter(user_position[0], user_position[1], label='UE Alert', marker='X', s=150)
                else:
                    print(f"INFO: gNB {gnb_id}: normal toa_val at {group.iloc[i]['timestamp']}")
                    plt.scatter(user_position[0], user_position[1], label='UE', marker='o', s=150)
             
plt.xlabel('X')
plt.ylabel('Y')
plt.xlim([-10, 110])
plt.ylim([-10, 110])
plt.legend()
plt.grid()
plt.show()
