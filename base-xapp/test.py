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
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, Input, concatenate
from sklearn.metrics import accuracy_score, classification_report
from collections import defaultdict
import random
import tensorflow as tf
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.utils import to_categorical

def DBConnect():
    conn = sqlite3.connect('/home/ipalama/work/OAI-colosseum-ric-integration-paloma/xapp-oai/base-xapp/toa_measurements_old.db')
    return conn

def helper_measure_toa(position, gnb_positions, emission_speed, sample_rate):
    toa_measurements = []
    for gnb_pos in gnb_positions:
        distance = np.linalg.norm(position - gnb_pos)
        toa = (distance / emission_speed) * sample_rate
        toa_measurements.append(toa)
    return toa_measurements

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
                    'toa_val': existing_row['toa_val'] + np.random.normal(0, 2),  # Add small random noise to the existing toa_val
                    'snr': existing_row['snr'] + np.random.normal(0, 1),  # Add small random noise to the existing snr
                    'timestamp': timestamp,
                }
                new_rows.append(new_row)

    new_rows_df = pd.DataFrame(new_rows)
    df = pd.concat([df, new_rows_df], ignore_index=True)  # Add new rows to the df DataFrame
    return df

def estimate_toa_random_path(initial_user_pos, gnb_positions, walking_speed, running_speed, emission_speed, duration, sample_rate):
    
    # Create a new dataset to store the TOA measurements
    toa_dataset_walking = pd.DataFrame(columns=['gnb_id', 'rnti', 'toa_val', 'snr', 'timestamp'])
    toa_dataset_running = pd.DataFrame(columns=['gnb_id', 'rnti', 'toa_val', 'snr', 'timestamp'])

    user_positions_walking = []
    user_positions_running = []
    tmp_pos_running = initial_user_pos
    tmp_pos_walking = initial_user_pos
    for t in range(duration):
        # Perform a random walk step
        new_position_walking = random_move(tmp_pos_walking, walking_speed)
        tmp_pos_walking = new_position_walking

        # Perform a random run step
        new_position_running = random_move(tmp_pos_running, running_speed)
        tmp_pos_running = new_position_running

        # Calculate TOA measurements for the new position
        toa_measurements_walking = helper_measure_toa(new_position_walking, gnb_positions, emission_speed, sample_rate)
        toa_measurements_running = helper_measure_toa(new_position_running, gnb_positions, emission_speed, sample_rate)

        # Append the measurements to the dataset
        for gnb_id in [0, 1, 2]:
            new_timestamp = datetime.strptime("2023-04-28T18:15:41.797775", "%Y-%m-%dT%H:%M:%S.%f") + timedelta(seconds=t+1)
            distance_walking = calculate_distance(new_position_walking, gnb_positions[gnb_id])
            distance_running = calculate_distance(new_position_running, gnb_positions[gnb_id])
            snr_walking = estimate_snr(distance_walking, 10.828835, 1, 2) + np.random.normal(0, 1)
            snr_running = estimate_snr(distance_running, 10.828835, 1, 2) + np.random.normal(0, 1)
            #print(gnb_id)
            toa_dataset_walking.loc[len(toa_dataset_walking)] = [gnb_id, 41206, toa_measurements_walking[gnb_id], snr_walking, new_timestamp]
            toa_dataset_running.loc[len(toa_dataset_running)] = [gnb_id, 41206, toa_measurements_running[gnb_id], snr_running, new_timestamp]        
        user_positions_walking.append(new_position_walking)
        user_positions_running.append(new_position_running)

    return toa_dataset_walking, toa_dataset_running, user_positions_walking, user_positions_running

def random_move(position, speed):
    angle = random.randint(0, 30)
    distance = random.randint(1, 2)
    dx = distance * np.cos(np.radians(angle))
    dy = distance * np.sin(np.radians(angle))
    new_position = position + np.array([dx, dy]) * speed
    #print(f"position: {position}")
    #print(f"new_position: {new_position}")

    return new_position

def spoof_dataframe(df, percentage, Spoof_magnitude, label):
    # Calculate the starting index for the last specified percentage of rows
    start_index = int(len(df) * (1 - percentage/100))
    
    # Create a new DataFrame to store the modified rows
    modified_df = df.copy()
    spoofed_gnb_id = [0]
    if label == 1:
        modified_df['spoofed'] = 0
    for index, row in df.iloc[start_index:].iterrows():
        if row['gnb_id'] in spoofed_gnb_id:
            #gnb_id = row['gnb_id']
            #rnti = row['rnti']
            #timestamp = row['timestamp']

            # Modify the toa_val and snr columns using the given formula
            modified_toa_val = row['toa_val'] + np.random.normal(Spoof_magnitude, 1)
            modified_snr = row['snr'] + np.random.normal(3, 1)

            # Update the modified DataFrame with the new values
            modified_df.at[index, 'toa_val'] = modified_toa_val
            modified_df.at[index, 'snr'] = modified_snr
            
            if label == 1:
                #add column spoofed to the dataframe
                modified_df.at[index, 'spoofed'] = 1

    return modified_df

#Models
def xgboost_model(train_data, next_row):
    X_train = train_data[['gnb_id', 'snr','timestamp']].values
    y_train = train_data['toa_val'].values
    X_test = next_row[['gnb_id', 'snr', 'timestamp']].values.reshape(1, -1)

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
    X_train = train_data[['gnb_id', 'snr','timestamp']].values
    y_train = train_data['toa_val'].values
    X_test = next_row[['gnb_id', 'snr', 'timestamp']].values.reshape(1, -1)

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

def generate_alert(real_value, predicted_value, threshold):
    # mettere qui classificatore che prende in input toa storici, toa di ora e toa predetto ora e restituisce anomaly alert o no
    # NB il regressore deve essere addestrato con dati di toa e snr di vari esperiemnti senza attacco
    # NB il classificatore deve essere addestrato con dati di toa e snr di vari esperiemnti con e senza attacco
    if abs(real_value - predicted_value) > threshold:
        print(f"Alert: Real value ({real_value}) differs from predicted value ({predicted_value}) by more than {threshold}")
        return 1
    else:
        return 0

def triangulate(gnb_positions, toa_vals):
    def objective(x, gnb_positions, toa_vals):
        return sum([(np.linalg.norm(pos - x) - toa) ** 2 for pos, toa in zip(gnb_positions, toa_vals)])

    initial_guess = np.mean(gnb_positions, axis=0)
    result = minimize(objective, initial_guess, args=(gnb_positions, toa_vals))
    return result.x

def calculate_distance(user_position, gnb_position):
    return np.sqrt((user_position[0] - gnb_position[0])**2 + (user_position[1] - gnb_position[1])**2)

def estimate_snr(distance, reference_snr, reference_distance, path_loss_exponent):
    return reference_snr - 10 * path_loss_exponent * np.log10(distance / reference_distance)

def new_exp(dataset):
    new_exp_dataset = dataset.copy()
    #apply random noise to columns snr and toa_val emulating a random experiment
    new_exp_dataset["snr"] = new_exp_dataset["snr"].apply(lambda x: x + np.random.normal(0, 1))
    new_exp_dataset["toa_val"] = new_exp_dataset["toa_val"].apply(lambda x: x + np.random.normal(0, 1))
    return new_exp_dataset

def same_spoof_index(df0, df1, df2):
    df0 = df0.copy(deep=True)
    df1 = df1.copy(deep=True)
    df2 = df2.copy(deep=True)
    df0 = df0.reset_index()  # make sure indexes pair with number of rows
    df1 = df1.reset_index()
    df2 = df2.reset_index()
    #propagate the spoofed value to the other dataframes
    for index, row in df0.iterrows():
        if row['spoofed'] == 1:
            df1.at[index, 'spoofed'] = 1
            df2.at[index, 'spoofed'] = 1
    for index, row in df1.iterrows():
        if row['spoofed'] == 1:
            df0.at[index, 'spoofed'] = 1
            df2.at[index, 'spoofed'] = 1
    for index, row in df2.iterrows():
        if row['spoofed'] == 1:
            df0.at[index, 'spoofed'] = 1
            df1.at[index, 'spoofed'] = 1
    return df0, df1, df2

# Define a function to calculate the MAD for a given window
def mad(window):
    median = window.median()
    deviations = abs(window - median)
    return deviations.median()

#create a function that take df and return df with rolling mean, standard deviation and MAD
def rolling_mean_std_mad(df, window_size):
    df['rolling_mean_snr'] = df['snr'].rolling(window=window_size, min_periods=1).mean()

    df['rolling_std_snr'] = df['snr'].rolling(window=window_size, min_periods=1).std()
    df['rolling_std_snr'] = df['rolling_std_snr'].fillna(-1)

    df['rolling_mad_snr'] = df['snr'].rolling(window=window_size, min_periods=1).apply(mad)
    df['rolling_mad_snr'] = df['rolling_mad_snr'].fillna(-1)

    df['rolling_mean_toa'] = df['toa_val'].rolling(window=window_size, min_periods=1).mean()

    df['rolling_std_toa'] = df['toa_val'].rolling(window=window_size, min_periods=1).std()
    df['rolling_std_toa'] = df['rolling_std_toa'].fillna(-1)
    
    df['rolling_mad_toa'] = df['toa_val'].rolling(window=window_size, min_periods=1).apply(mad)
    df['rolling_mad_toa'] = df['rolling_mad_toa'].fillna(-1)
    return df

# Processing phases
def phase_1(window_size, train_df, test_df):

    #print(f"train_df: {train_df}")
    #print(f"test_df: {test_df}")

    rolling_df = train_df.rolling(window=window_size, min_periods=window_size)
    train_df.reset_index(drop=True, inplace=True)

    # loop through each window
    for window_data in rolling_df:
        if len(window_data) < window_size:
            continue
        window_data.reset_index(drop=True, inplace=True)
        # get index of the element in train_df which is equal to the last element in window
        i = train_df[train_df['toa_val'] == window_data.iloc[-1]['toa_val']].index.values[0]

        next_row = test_df.iloc[i]
        y_pred = xgboost_model(window_data, next_row)
        #y_test = next_row['toa_val']

        #  transform train data from n x m to 1 x (n * m) where n is the number of rows and m is the number of columns, use column name with incremental index as column name (e.g. toa_val_0, toa_val_1, ...)
        window_data = window_data.stack().to_frame().T
        # for each column with same name, add incremental index to column name (e.g. toa_val_0, toa_val_1, ...)
        window_data.columns = ['_'.join((str(col[1]), str(col[0]))) for col in window_data.columns]

        #add column "y_pred" with value y_pred
        window_data['y_pred'] = y_pred

        if test_df.iloc[i]['spoofed'] == 1:
            window_data['spoofed'] = 1
        else:
            window_data['spoofed'] = 0
        
        #save all the window_data in a pd.DataFrame called phase1_out_df
        if 'phase1_out_df' not in locals():
            phase1_out_df = window_data
        else:
            phase1_out_df = phase1_out_df._append(window_data)

    return phase1_out_df

# Connect to the SQLite database
conn = DBConnect()
cur = conn.cursor()
cur.execute("SELECT * FROM measurements")
rows = cur.fetchall()
conn.close()

# Convert fetched data to a suitable format (e.g., Pandas DataFrame)
df_raw = pd.DataFrame(rows, columns=['gnb_id', 'rnti', 'toa_val', 'snr', 'timestamp'])

df_raw = df_raw.dropna()  # Remove rows with NaN values
#add column "spoofed" with value 0
df = df_raw.copy()

df = populateWithGaussValues(df) 

df[['gnb_id', 'rnti']] = df[['gnb_id', 'rnti']].astype(int)
df[['toa_val', 'snr']] = df[['toa_val', 'snr']].astype(float)
df['timestamp'] = pd.to_datetime(df['timestamp'])

# apply datetime.strptime to convert the timestamp column to datetime
df['timestamp'] = df['timestamp'].apply(lambda x: datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S.%f'))

user_data = np.array([50, 50]) # Initial user position

gnb_positions = {
    0: np.array([0, 0]),
    1: np.array([100, 0]),
    2: np.array([50, 100])
}

walking_speed = 2 * 1000 / 3600  # 3 km/h in meters per second
running_speed = 8 * 1000 / 3600  # 8 km/h in meters per second
emission_speed = 299792458  # Speed of light in meters per second
duration = 1000  # seconds
sample_rate = 61.44 * 1e6  # 61.44 MSps in samples per second
window_size = 10
threshold = 5

toa_dataset_walking, toa_dataset_running, user_positions_walking, user_positions_running = estimate_toa_random_path(user_data, gnb_positions, walking_speed, running_speed, emission_speed, duration, sample_rate)

# Convert the list of user positions to a NumPy array
user_positions_walking = np.array(user_positions_walking)
user_positions_running = np.array(user_positions_running)

data_running = df._append(toa_dataset_running, ignore_index=True)
data_running['timestamp'] = data_running.groupby('gnb_id')['timestamp'].diff()
data_running['timestamp'] = data_running['timestamp'].apply(lambda x: x.total_seconds())
data_running.loc[df.groupby('gnb_id')['timestamp'].idxmin(), 'timestamp'] = 0
data_running['timestamp'] = data_running['timestamp'].apply(lambda x: round(x))

data_walking = df._append(toa_dataset_walking, ignore_index=True)
data_walking['timestamp'] = data_walking.groupby('gnb_id')['timestamp'].diff()
data_walking['timestamp'] = data_walking['timestamp'].apply(lambda x: x.total_seconds())
data_walking.loc[df.groupby('gnb_id')['timestamp'].idxmin(), 'timestamp'] = 0
data_walking['timestamp'] = data_walking['timestamp'].apply(lambda x: round(x))

percentage = 30
Spoof_magnitude = 10



train_data_walking = new_exp(data_walking)
train_data_running = new_exp(data_running)

spoofed_train_data_walking = spoof_dataframe(train_data_walking, percentage, Spoof_magnitude, 1)
spoofed_train_data_running = spoof_dataframe(train_data_running, percentage, Spoof_magnitude, 1)


#sns.pairplot(spoofed_train_data_walking)
#plt.show()

#print(spoofed_train_data_walking.describe())

#create a function that move column 'spoofed' to the last column
def move_columns(df, col1, col2):
    col1 = df.pop(col1)
    col2 = df.pop(col2)
    df.insert(len(df.columns)-1, col1.name, col1)
    df.insert(len(df.columns), col2.name, col2)
    return df

# how call the function passing as pos the -1 position


########################################## Phase 1 ##########################################
phase1_out_df_0 = pd.DataFrame()
phase1_out_df_1 = pd.DataFrame()
phase1_out_df_2 = pd.DataFrame()
#group by gnb_id
for gnb_id, group in train_data_walking.groupby('gnb_id'):
    group = rolling_mean_std_mad(group, window_size)
    for gnb_id_spoofed, group_spoofed in spoofed_train_data_walking.groupby('gnb_id'):
        group_spoofed = rolling_mean_std_mad(group_spoofed, window_size)
        if gnb_id == gnb_id_spoofed:
            if gnb_id == 0:
                phase1_out_df_0 = phase_1(window_size, group, group_spoofed)    
            elif gnb_id == 1:
                phase1_out_df_1 = phase_1(window_size, group, group_spoofed)
            elif gnb_id == 2:
                phase1_out_df_2 = phase_1(window_size, group, group_spoofed)

# iterate over the rows of phase1_out_df_0, phase1_out_df_1, phase1_out_df_2, and if a row has spoofed = 1, then set spoofed = 1 for the same row in phase1_out_df_0, phase1_out_df_1, phase1_out_df_2

phase1_out_df_0,phase1_out_df_1,phase1_out_df_2 =  same_spoof_index(phase1_out_df_0, phase1_out_df_1, phase1_out_df_2)

phase1_out_df_0 = move_columns(phase1_out_df_0, 'y_pred', 'spoofed')
phase1_out_df_1 = move_columns(phase1_out_df_1, 'y_pred', 'spoofed')
phase1_out_df_2 = move_columns(phase1_out_df_2, 'y_pred', 'spoofed')

phase1_out_df_0.to_csv('phase1_out_df_0.csv', index=False)
phase1_out_df_1.to_csv('phase1_out_df_1.csv', index=False)
phase1_out_df_2.to_csv('phase1_out_df_2.csv', index=False)

########################################## Phase 2 ##########################################
#for each entry of phase1_out_df_0, phase1_out_df_1, phase1_out_df_2
#group by gnb_id
def multi_headed_mlp_model(phase1_out_df_0, phase1_out_df_1, phase1_out_df_2):
    input_shape_gnb0 = phase1_out_df_0.iloc[:, :-1].shape[1]
    input_shape_gnb1 = phase1_out_df_1.iloc[:, :-1].shape[1]
    input_shape_gnb2 = phase1_out_df_2.iloc[:, :-1].shape[1]

    # Define the inputs for each head
    input_gnb0 = Input(shape=(input_shape_gnb0,), name='input_gnb1')
    input_gnb1 = Input(shape=(input_shape_gnb1,), name='input_gnb2')
    input_gnb2 = Input(shape=(input_shape_gnb2,), name='input_gnb3')

    # Define the MLP structure for each head
    head_gnb0 = Dense(64, activation='relu')(input_gnb0)
    head_gnb1 = Dense(64, activation='relu')(input_gnb1)
    head_gnb2 = Dense(64, activation='relu')(input_gnb2)

    # Concatenate the outputs of each head
    merged = concatenate([head_gnb0, head_gnb1, head_gnb2])

    # Add a final dense layer and output layer
    hidden1 = Dense(128, activation='relu')(merged)
    hidden2 = Dense(64, activation='relu')(hidden1)

    # Output layer
    output = Dense(1, activation='sigmoid')(hidden2)

    # Create the model
    model = Model(inputs=[input_gnb0, input_gnb1, input_gnb2], outputs=output)

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Print the model summary
    #model.summary()

    return model

model = multi_headed_mlp_model(phase1_out_df_0, phase1_out_df_1, phase1_out_df_2)

X0 = phase1_out_df_0.iloc[:, :-1].values
X1 = phase1_out_df_1.iloc[:, :-1].values
X2 = phase1_out_df_2.iloc[:, :-1].values

y = phase1_out_df_0.iloc[:, -1].values

# Fit the model
model.fit([X0, X1, X2], y, epochs=100, batch_size=32, verbose=2)

########################################## Phase 3 TEST ##########################################
test_dataset_walking = new_exp(data_walking)
spoofed_test_data_walking = spoof_dataframe(test_dataset_walking, percentage, Spoof_magnitude, 1)

for gnb_id, group in spoofed_test_data_walking.groupby('gnb_id'):
    group = rolling_mean_std_mad(group, window_size)
    if gnb_id == 0:
        phase3_out_df_0 = phase_1(window_size, group, group)             
    elif gnb_id == 1:
        phase3_out_df_1 = phase_1(window_size, group, group)
    elif gnb_id == 2:
        phase3_out_df_2 = phase_1(window_size, group, group)

# Define a list of columns to remove
cols_to_remove = [f'spoofed_{i}' for i in range(window_size)]

# Remove the columns from each DataFrame
for i, df in enumerate([phase3_out_df_0, phase3_out_df_1, phase3_out_df_2]):
    df.drop(columns=cols_to_remove, inplace=True)

phase3_out_df_0, phase3_out_df_1, phase3_out_df_2 = same_spoof_index(phase3_out_df_0, phase3_out_df_1, phase3_out_df_2)

phase3_out_df_0 = move_columns(phase3_out_df_0, 'y_pred', 'spoofed')
phase3_out_df_1 = move_columns(phase3_out_df_1, 'y_pred', 'spoofed')
phase3_out_df_2 = move_columns(phase3_out_df_2, 'y_pred', 'spoofed')

phase3_out_df_0.to_csv('phase3_out_df_0.csv', index=False)  
phase3_out_df_1.to_csv('phase3_out_df_1.csv', index=False)
phase3_out_df_2.to_csv('phase3_out_df_2.csv', index=False) 

y_new_pred = model.predict([phase3_out_df_0.iloc[:, :-1].values, phase3_out_df_1.iloc[:, :-1].values, phase3_out_df_2.iloc[:, :-1].values])

print(f'len(y_new_pred): {len(y_new_pred)}')
print(f'len(phase3_out_df_0): {len(phase3_out_df_0)}')
print(f'len(phase1_out_df_0): {len(phase1_out_df_0)}')

accuracy = accuracy_score(phase3_out_df_0.iloc[:, -1], y_new_pred.round())
report = classification_report(phase3_out_df_0.iloc[:, -1], y_new_pred.round())

print("Accuracy: \n", accuracy)
print("Classification Report: \n", report)
