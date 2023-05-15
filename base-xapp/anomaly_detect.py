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
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D
from collections import defaultdict


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
    # mettere qui classificatore che prende in input toa storici, toa di ora e toa predetto ora e restituisce anomaly alert o no
    # NB il regressore deve essere addestrato con dati di toa e snr di vari esperiemnti senza attacco
    # NB il classificatore deve essere addestrato con dati di toa e snr di vari esperiemnti con e senza attacco
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

#print(df)

df[['gnb_id', 'rnti']] = df[['gnb_id', 'rnti']].astype(int)
df[['toa_val', 'snr']] = df[['toa_val', 'snr']].astype(float)
df['timestamp'] = pd.to_datetime(df['timestamp'])

# apply datetime.strptime to convert the timestamp column to datetime
df['timestamp'] = df['timestamp'].apply(lambda x: datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S.%f'))

# User positioning
time_window = 5  # seconds

window_size = 5
threshold = 5

gnb_positions = {
    0: np.array([0, 0]),
    1: np.array([100, 0]),
    2: np.array([50, 100])
}

# Plotting
#fig, ax = plt.subplots()

# Plot gNB positions
#for gnb_id, pos in gnb_positions.items():
#    plt.scatter(pos[0], pos[1], label=f'gNB {gnb_id}', marker='^', s=150)

# Models inventory

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

def deepAnt_model(train_data, next_row):
    X_train = train_data[['gnb_id', 'snr']].values.reshape(-1, 1, 2).astype('float32')
    y_train = train_data['toa_val'].values.astype('float32')
    X_test = next_row[['gnb_id', 'snr']].values.reshape(1, 1, 2).astype('float32')

    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=1, activation='relu', input_shape=(1, 2)))
    model.add(Conv1D(filters=32, kernel_size=1, activation='relu'))
    model.add(LSTM(50, activation='relu'))
    model.add(Dense(40, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=100, verbose=0)

    y_pred = model.predict(X_test)[0][0]

    return y_pred

def arima_model(train_data, next_row, order=(1, 1, 1)):
    y_train = train_data['toa_val'].values
    y_test_index = next_row.index.values[0]

    model = ARIMA(y_train, order=order)
    model_fit = model.fit()

    y_pred = model_fit.forecast(steps=1)[0]
    y_pred_series = pd.Series(y_pred, index=[y_test_index])

    return y_pred_series

alert_indexes = []
timestamps_done = []
models = [xgboost_model, linear_regression_model, lstm_model, deepAnt_model, arima_model]
performance_metrics_dict = {}

for i in range(window_size, len(df)):
    train_data = df.iloc[i - window_size:i]
    train_data = train_data[~train_data.index.isin(alert_indexes)]  # Exclude rows that generated alerts
    
    if len(train_data) == 0:
        continue

    next_row = df.iloc[i]
    
    # Iterate through each model
    for model in models:
        model_pred = model(train_data, next_row)        
        # Store the performance metrics
        if model.__name__ not in performance_metrics_dict:
            performance_metrics_dict[model.__name__] = []

        performance_metrics_dict[model.__name__].append([model_pred, next_row['toa_val'], next_row['timestamp'], next_row['gnb_id']])
    
    y_pred = linear_regression_model(train_data, next_row)
    
    y_test = next_row['toa_val']

    #print(f"Predicted: {y_pred.item()}, Actual: {y_test}, threshold: {threshold}")

    alert = generate_alert(y_test, y_pred.item(), threshold)
    if df.iloc[i]['timestamp'] not in timestamps_done:
        timestamps_done.append(df.iloc[i]['timestamp'])
        toa_vals = df.loc[df['timestamp'] == df.iloc[i]['timestamp']]['toa_val'].values
        gnb_ids = df.loc[df['timestamp'] == df.iloc[i]['timestamp']]['gnb_id'].values
        gnb_coords = [gnb_positions[id] for id in gnb_ids]
        user_position = triangulate(gnb_coords, toa_vals)
        #print(f"User position at: {user_position}")

        if alert:
            print(alert)
            #plt.scatter(user_position[0], user_position[1], label='UE Alert', marker='X', s=150)
            alert_indexes.append(i)  # Add the index of the row that generated an alert
        else:
            #plt.scatter(user_position[0], user_position[1], label='UE Alert', marker='o', s=150)
            print("No alert")



#print(performance_metrics_dict)

# Extract data for each gnb_id
gnb_data = defaultdict(lambda: defaultdict(lambda: {"timestamps": [], "estimated": [], "real": []}))
for model_name, model_data in performance_metrics_dict.items():
    #print(f"Model data: {model_data}")
    for est, real, timestamp, gnb_id in model_data:
        #print(f"GNB ID: {gnb_id}, Timestamp: {timestamp}, Estimated: {est}, Real: {real}")
        gnb_data[gnb_id][model_name]["timestamps"].append(timestamp)
        gnb_data[gnb_id][model_name]["estimated"].append(est)
        gnb_data[gnb_id][model_name]["real"].append(real)

# Create the plot
for gnb_id, model_data in gnb_data.items():
    plt.figure(figsize=(10, 5))
    real_value = -999
    for model_name, gnb_values in model_data.items():
        plt.scatter(gnb_values["timestamps"], gnb_values["estimated"], label=f"{model_name} Estimated", alpha=0.7)
        if real_value == -999:
            real_value = gnb_values["real"]
            plt.scatter(gnb_values["timestamps"], gnb_values["real"], label= "Real value", alpha=0.7)
    plt.xlabel("Timestamp")
    plt.ylabel("Value")
    plt.title(f"GNB ID {gnb_id}: Estimated vs Real Values")
    plt.legend()
    plt.show()

'''
plt.xlabel('X')
plt.ylabel('Y')
plt.xlim([-10, 110])
plt.ylim([-10, 110])
plt.legend()
plt.grid()
plt.show()
'''