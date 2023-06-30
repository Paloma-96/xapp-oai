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
    conn = sqlite3.connect('/home/ipalama/work/OAI-colosseum-ric-integration-paloma/xapp-oai/base-xapp/toa_measurements.db')
    return conn

# Connect to the SQLite database
conn = DBConnect()
cur = conn.cursor()
cur.execute("SELECT * FROM measurements")
rows = cur.fetchall()
conn.close()

# Convert fetched data to a suitable format (e.g., Pandas DataFrame)
df_raw = pd.DataFrame(rows, columns=['gnb_id', 'rnti', 'toa_val', 'snr', 'timestamp'])

#save the dataframe to a csv file
#df_raw.to_csv('toa_measurements.csv', index=False)

#print(df_raw)

#elimina valori nan, None e <=0
df = df_raw.dropna()
df = df[df['toa_val'] != None]
df = df[df['snr'] != None]
df = df[df['toa_val'] != 'None']
df = df[df['snr'] != 'None']
df = df[df['toa_val'] > 0]
df = df[df['snr'] > 0]
df = df.reset_index(drop=True)


#set gnb_id and rnti as integers and toa_val and snr as floats
df['gnb_id'] = df['gnb_id'].astype(int)
df['rnti'] = df['rnti'].astype(int)
df['toa_val'] = df['toa_val'].astype(float)
df['snr'] = df['snr'].astype(float)

#df = df.to_numpy()
#sns.pairplot(df_raw)
#plt.show()
print(df)


#plot histograms of toa_val, snr in the same plot but with different figures for each gnb_id.  add labels, title, legend to better undestand the plots
df.hist(column='toa_val', by='gnb_id', bins=100, figsize=(12,8))
df.hist(column='snr', by='gnb_id', bins=100, figsize=(12,8))
plt.show()



