import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pyod.models.knn import KNN
from scipy.optimize import minimize
import tkinter
import matplotlib

matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
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
from matplotlib.dates import date2num
import numpy as np
from scipy.optimize import least_squares
from scipy import signal
import scipy.optimize as opt
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.signal import cwt, morlet2
import pywt

def DBConnect():
    conn = sqlite3.connect('/home/ipalama/work/OAI-colosseum-ric-integration-paloma/xapp-oai/base-xapp/toa_measurements.db')
    return conn

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

def fetch_data(conn, table_name):
    cur = conn.cursor()
    cur.execute(f"SELECT * FROM {table_name}")
    rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=['gnb_id', 'rnti', 'toa_val', 'snr', 'timestamp'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df[['gnb_id', 'rnti']] = df[['gnb_id', 'rnti']].astype(int)
    df[['toa_val', 'snr']] = df[['toa_val', 'snr']].astype(float)
    return df.dropna()

# positioning
import numpy as np

def trilateration_solver(gnb_positions, distances):
    # Compute the differences between gNB positions
    p1, p2, p3 = [np.array(pos) for pos in gnb_positions]
    d1, d2, d3 = distances
    
    ex = (p2 - p1) / np.linalg.norm(p2 - p1)
    i = np.dot(ex, p3 - p1)
    ey = (p3 - p1 - i * ex) / np.linalg.norm(p3 - p1 - i * ex)
    
    # Compute the position of the UE
    x = (d1**2 - d2**2 + np.linalg.norm(p2 - p1)**2) / (2 * np.linalg.norm(p2 - p1))
    y = (d1**2 - d3**2 + np.linalg.norm(p3 - p1)**2 - 2 * i * x) / (2 * np.linalg.norm(p3 - p1 - i * ex))
    
    return p1 + x * ex + y * ey


def trilateration_objective(params, *args):
    x, y = params
    p1, p2, p3, d1, d2, d3 = args

    # Calculate the actual distances between the UE and the gNBs
    actual_d1 = np.linalg.norm(np.array([x, y]) - p1)
    actual_d2 = np.linalg.norm(np.array([x, y]) - p2)
    actual_d3 = np.linalg.norm(np.array([x, y]) - p3)
    
    # Calculate the residuals as the difference between measured and actual distances
    residuals = np.array([actual_d1 - d1, actual_d2 - d2, actual_d3 - d3])
    
    # Check if residuals are finite, if not, return a large value
    if not np.all(np.isfinite(residuals)):
        return np.full_like(residuals, 1e10)
    
    return residuals

def trilateration_non_linear_least_squares(gnb_positions, distances):
    p1, p2, p3 = gnb_positions
    d1, d2, d3 = distances
    
    initial_guess = np.array([0, 0])
    result = least_squares(trilateration_objective, initial_guess, args=(p1, p2, p3, d1, d2, d3))
    
    return result.x



# miscellanea

def toa_val_to_distance(toa_vals, sampling_rate):
    #given samples and sampling rate, return distance in meters
    return (toa_vals * (299792458)) / (sampling_rate)

def distance_to_toa_val(distance, sampling_rate):
    #given distance in meters and sampling rate, return toa_val
    return (distance * (sampling_rate)) / (299792458)

def enforce_same_number_of_rows(dataframes):

    min_rows = min(len(df) for table_name, df in dataframes.items())
    
    #take only last min_rows rows for each dataframe
    for table_name, df in dataframes.items():
        dataframes[table_name] = df.tail(min_rows)
    return dataframes

def distance_between_points(p, q):
    """
    Return the Euclidean distance between points p and q
    assuming both have the same number of dimensions
    """
    s_sq_difference = 0
    for p_i, q_i in zip(p, q):
        s_sq_difference += (p_i - q_i)**2
    
    distance = s_sq_difference**0.5
    return distance

#############################################################################
# One approach to determine the target level is to analyze the wavelet coefficients 
# and look for a level where the majority of the energy is concentrated. You can then 
# remove the detail coefficients at that level to eliminate the unwanted cyclical component. 
# To do this, you can calculate the sum of the absolute values of the coefficients at each 
# level and find the level with the highest sum.
#############################################################################    
def find_target_level(coeffs):
    level_sums = np.zeros(len(coeffs))
    level = 0

    for i in range(len(coeffs)):
        level_sums[i] = np.sum(np.abs(coeffs[i]))
        if level_sums[i] > level_sums[level]:
            level = i

    return level

def wavelet_filtering(df, plot_wavelet_effect):
    # Perform the discrete wavelet transform
    coeffs = pywt.wavedec(df['cal_toa_val_window'].to_numpy(), 'db2', mode='symmetric')

    # Set the detail coefficients at a specific level to zero
    #target_level = 2  # Adjust this according to the level corresponding to the unwanted cyclical component
    target_level = find_target_level(coeffs)
    coeffs[target_level] = np.zeros_like(coeffs[target_level])

    # Reconstruct the signal without the unwanted cyclical component
    filtered_signal = pywt.waverec(coeffs, 'db2')

    # select only first 1000 rows of the filtered signal and df['cal_toa_val_window']
    filtered_signal = pd.Series(filtered_signal[:len(df['cal_toa_val_window'])], index=df['cal_toa_val_window'].index)
    df_gnb = df.copy()
    # create a new column in df_gnb called 'detrended_signal' which is the difference between df_gnb['cal_toa_val_window'] and filtered_signal
    df_gnb['detrended_signal'] = df_gnb['cal_toa_val_window'] - filtered_signal
    if plot_wavelet_effect:
        # Plot the original and filtered signals
        plt.figure(figsize=(15, 6))
        plt.plot(df_gnb['cal_toa_val_window'].to_numpy(), label='Original Signal')
        plt.plot(filtered_signal, label='Filtered Signal', linestyle='--')
        #plot detrended signal
        plt.plot(df_gnb['detrended_signal'].to_numpy(), label='Detrended Signal', linestyle='--')
        plt.legend()
        plt.show()
    return df_gnb