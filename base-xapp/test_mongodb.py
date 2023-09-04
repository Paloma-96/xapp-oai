import pandas as pd
from pymongo import MongoClient
import numpy as np

import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import scipy.optimize as opt
import pywt
import math
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

wavelet_filtering_flag = False

def mongodb_init(collection_name):
    client = MongoClient('mongodb://root:rootpassword@localhost:27017/')
    db = client['toa_measurements']
    collection = db[collection_name]
    return collection

def calculate_distance(point1, point2):
    return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

def error(x, locations, distances):
    residuals = []  # List to store the residuals for each gNB

    for loc, dist in zip(locations, distances):
        # Calculate the squared difference in the x and y coordinates
        square_diff = np.square(np.subtract(x, loc))
        
        # Sum the squared differences to get the squared Euclidean distance
        squared_distance = np.sum(square_diff)
        
        # Take the square root of the sum to get the Euclidean distance
        calculated_distance = np.sqrt(squared_distance)
        
        # Subtract the known distance to get the residual
        residual = calculated_distance - dist
        
        # Add the residual to the list
        residuals.append(residual)
    
    return residuals

# Define the nonlinear function to fit
def nonlinear_func(x, a, b, c):
    return a * np.sin(b * x) + c

def wavelet_filtering(distances, plot_wavelet_effect, real_distance, wavelet='db2'):\

    distances_old = distances
    #take only first 30% of distances
    distances = distances[:int(len(distances)*0.3)]
    real_distance = real_distance[:int(len(real_distance)*0.3)]

    # Perform the discrete wavelet transform
    coeffs = pywt.wavedec(distances, wavelet, mode='symmetric')

    # Set the detail coefficients at a specific level to zero
    #target_level = 2  # Adjust this according to the level corresponding to the unwanted cyclical component
    target_level = find_target_level(coeffs)
    coeffs[target_level] = np.zeros_like(coeffs[target_level])

    # Reconstruct the signal without the unwanted cyclical component
    filtered_signal = pywt.waverec(coeffs, wavelet, mode='symmetric')

    # create a new column in df_gnb called 'detrended_signal' which is the difference between df_gnb['cal_toa_val_window'] and filtered_signal
    detrended_distances = distances - filtered_signal[:len(distances)]

    diff = np.abs(real_distance - detrended_distances)
    total_difference = np.sum(diff)
    # o
    rmse = np.sqrt(np.mean((real_distance-detrended_distances)**2))

    #print(f"Total difference: {total_difference}")
    #print(f"RMSE: {rmse}")

    #apply rolling mean to detrended_distances
    rolling_mean_window = 2
    df_filtered_signal = pd.DataFrame(filtered_signal)
    filtered_signal_mean = df_filtered_signal.rolling(rolling_mean_window).mean().iloc[rolling_mean_window-1:].values
    
    # Generate polynomial features
    degree = 10  # Adjust the degree of the polynomial as needed
    poly_features = PolynomialFeatures(degree=degree)
    X_poly = poly_features.fit_transform(np.arange(len(filtered_signal_mean)).reshape(-1, 1))

    # Fit the polynomial regression model
    model = LinearRegression()
    model.fit(X_poly, filtered_signal_mean)

    # Predict the filtered_signal values using the fitted model
    filtered_signal_approx = model.predict(X_poly)

    next_value = model.predict(poly_features.transform([[len(filtered_signal_approx)]]))

    # Plot the original and approximated filtered_signal curves and the next value
    #plt.plot(filtered_signal_mean, label='original')
    #plt.plot(filtered_signal_approx, label='approximated')
    #plt.plot(len(filtered_signal_approx), next_value, 'ro', label='next value')
    #plt.legend()
    #plt.show()

    if plot_wavelet_effect:
        # Plot the original and filtered distances
        plt.plot(distances, label='original')
        plt.plot(filtered_signal, label='ciclycal effect')
        plt.plot(detrended_distances, label='detrended')
        plt.plot(real_distance, label='real_distance')
        plt.legend()
        plt.show()
    
    return next_value

def find_target_level(coeffs):
    level_sums = np.zeros(len(coeffs))
    level = 0

    for i in range(len(coeffs)):
        level_sums[i] = np.sum(np.abs(coeffs[i]))
        if level_sums[i] > level_sums[level]:
            level = i

    return level


speed_of_light = 299792458 # in meters per second
sampling_rate = 61.44e6 # in samples per second
gnbs_positions = [[0, 9], [3.5, 0], [7, 9]] # in meters

collection_names = ['toa_measurements_exp1','toa_measurements_exp2','toa_measurements_exp3','toa_measurements_exp4','toa_measurements_exp5']
ue_real_position = [[0, 0],[7, 0],[3.5, 9],[7, 4.5],[3.5, 12]]

# iterare through all collections, take also index of collection
for collection_name, ue_real_position in zip(collection_names, ue_real_position):

    collection = mongodb_init(collection_name)

    # Fetch the toaVal values from MongoDB and store them in a list
    toa_values = []
    for doc in collection.find():
        toa_values.append(doc['paramMap'][1]['toa']['toaVal'])

    # convert toa_values to numpy array
    toa_values = np.array(toa_values)

    #assuming that toa_values is a numpy array of arrays of shape (1,3), remove the outliers from the inner arrays
    toa_values = np.array([x for x in toa_values if (x[0] > 0 and x[1] > 0 and x[2] > 0)])
    toa_values = np.array([x for x in toa_values if (x[0] < 100 and x[1] < 100 and x[2] < 100)])


    # Calculate distances
    distances = [val * speed_of_light / sampling_rate for val in toa_values]
    real_distances = [calculate_distance(gnb_position, ue_real_position) for gnb_position in gnbs_positions]

    #create a train dataset of distances and a test dataset of distances (30% and 70%)
    train_distances = distances[:int(len(distances)*0.3)]
    test_distances = distances[int(len(distances)*0.3):]

    mean_train_distances = np.mean(train_distances, axis=0)

    #apply offset to the distances
    cal_offset = real_distances - mean_train_distances

    #print(f"Calibration offset: {cal_offset}")

    #print(f"Train distances: {train_distances}")

    train_distances_cal = train_distances + cal_offset

    
    # Initial guess for the user's position
    x0 = [0,0]

    #apply calibration offset to test_distances
    test_distances_cal = test_distances + cal_offset

    # apply rolling mean to test_distances
    rolling_mean_window = 10
    df_test_distances_cal = pd.DataFrame(test_distances_cal)
    test_distances_cal_mean = df_test_distances_cal.rolling(rolling_mean_window).mean().iloc[rolling_mean_window-1:].values

    # for each entry in cal_test_distances use least squares to estimate the user's position 
    estimated_positions = []
    for dist in test_distances_cal_mean: 
        result = least_squares(error, x0, args=(gnbs_positions, dist))
        estimated_positions.append(result.x)

    #plot the estimated positions of user, the real position of user and the positions of the gNBs
    plot = True
    if (plot):
        plt.scatter([x[0] for x in estimated_positions], [x[1] for x in estimated_positions], label='estimated positions', marker='o')
        plt.scatter(ue_real_position[0], ue_real_position[1], label='real position', marker='o')
        plt.scatter([x[0] for x in gnbs_positions], [x[1] for x in gnbs_positions], label='gNBs', marker='x')
        plt.legend()
        plt.show()

