import datetime
import pandas as pd
import math
from pymongo import MongoClient
import numpy as np
import matplotlib
i = 0
while i < 10:
    i += 1
    try:
        #matplotlib.use('TkAgg')
        matplotlib.use('Agg')
        break
    except:
        print(i)
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import scipy.optimize as opt
import pywt
import math
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import pickle

wavelet_filtering_flag = True
save_fig = True

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
        #square_diff = np.square(np.subtract(x, loc))
        abs_diff = np.absolute(np.subtract(x, loc))
        
        # Sum the squared differences to get the squared Euclidean distance
        #squared_distance = np.sum(square_diff)
        manhattan_distance = np.sum(abs_diff)
        
        # Take the square root of the sum to get the Euclidean distance
        #calculated_distance = np.sqrt(squared_distance)
        
        # Subtract the known distance to get the residual
        #residual = calculated_distance - dist
        residual = manhattan_distance - dist
        
        # Add the residual to the list
        residuals.append(residual)
    
    return residuals

def jacobian_function(params, *args):
    # Unpack any additional arguments needed by the error function.
    locations, distances = args

    # Number of parameters (x-coordinates in this case)
    num_params = len(params)
    num_residuals = len(distances)

    # Initialize the Jacobian matrix with zeros
    jac = np.zeros((num_residuals, num_params))

    for i in range(num_residuals):
        for j in range(num_params):
            # Calculate the derivative of the i-th residual with respect to the j-th parameter
            if params[j] < locations[i][j]:
                jac[i, j] = -1
            else:
                jac[i, j] = 1

    return jac


# Define the nonlinear function to fit
def nonlinear_func(x, a, b, c):
    return a * np.sin(b * x) + c

def wavelet_filtering(distances, plot_wavelet_effect, real_distance, wavelet='db2'):

    distances_old = distances
    #take only first 30% of distances
    #distances = distances[:int(len(distances)*0.3)]
    #real_distance = real_distance[:int(len(real_distance)*0.3)]

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
    #print(f"Next value: {next_value}")

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

def predict_cyclical_offset(distances, real_distances):
    # apply transpose to distances
    distances = np.transpose(distances)

    # create an array of real_distances[0] with the same length as distances[0]
    real_distances_0 = np.full(len(distances[0]), real_distances[0])
    real_distances_1 = np.full(len(distances[1]), real_distances[1])
    real_distances_2 = np.full(len(distances[2]), real_distances[2])

    # apply wavelet filtering to distances
    plot = False
    next_offset_0 = wavelet_filtering(distances[0], plot, real_distances_0)
    next_offset_1 = wavelet_filtering(distances[1], plot, real_distances_1)
    next_offset_2 = wavelet_filtering(distances[2], plot, real_distances_2)

    next_offset = [next_offset_0.squeeze(), next_offset_1.squeeze(), next_offset_2.squeeze()]

    return next_offset

def process_distances(toa_values, ue_real_position, gnbs_positions, error_est_pos_list, sorted_errors_list, cdf_values_list, estimated_positions_list, spoofing_index):
    
    distances = [val * speed_of_light / sampling_rate for val in toa_values]
    real_distances = [calculate_distance(gnb_position, ue_real_position) for gnb_position in gnbs_positions]

    #create a train dataset of distances and a test dataset of distances (20% and 80%)
    train_distances = distances[:int(len(distances)*spoofing_index)]
    test_distances = distances[int(len(distances)*spoofing_index):]

    # apply rolling mean to train_distances and test_distances
    rolling_mean_window = 5
    df_train_distances = pd.DataFrame(train_distances)
    df_test_distances = pd.DataFrame(test_distances)
    train_distances = df_train_distances.rolling(rolling_mean_window).mean().iloc[rolling_mean_window-1:].values
    test_distances = df_test_distances.rolling(rolling_mean_window).mean().iloc[rolling_mean_window-1:].values


    #for each entry in test_distances use least squares to estimate the user's position
    estimated_positions = []
    train_distances_tmp = train_distances.copy()

    lower_bounds = [0, 0]  # Lower bounds for x and y
    upper_bounds = [40, 40]  # Upper bounds for x and y

    for dist in test_distances:
        x0 = [0,0]

        mean_train_distances = np.mean(train_distances_tmp, axis=0)

        cal_offset = real_distances - mean_train_distances

        train_distances_cal = train_distances_tmp + cal_offset

        if wavelet_filtering_flag:

            cyclical_offset = predict_cyclical_offset(train_distances_cal, real_distances)
            train_distances_tmp = np.append(train_distances_tmp, [dist], axis=0)

        else:
            cyclical_offset = np.zeros(len(dist))

        test_distance_cal = dist + cal_offset - cyclical_offset

        #result = least_squares(error, x0, args=(gnbs_positions, test_distance_cal), loss='soft_l1', bounds=(lower_bounds, upper_bounds), method='dogbox', jac=jacobian_function)
        #result = least_squares(error, x0, args=(gnbs_positions, test_distance_cal), method='lm', jac=jacobian_function)
        result = least_squares(error, x0, args=(gnbs_positions, test_distance_cal), loss='soft_l1', bounds=(lower_bounds, upper_bounds), method='trf', jac=jacobian_function)
        estimated_positions.append(result.x)

    error_est_pos = np.array([calculate_distance(estimated_position, ue_real_position) for estimated_position in estimated_positions])
    error_est_pos_list.append(error_est_pos)

    sorted_errors = np.sort(error_est_pos)
    sorted_errors_list.append(sorted_errors)

    cdf_values = 1. * np.arange(len(sorted_errors)) / (len(sorted_errors) - 1)
    cdf_values_list.append(cdf_values)

    estimated_positions_list.append(estimated_positions)
    #plot the estimated positions of user, the real position of user and the positions of the gNBs
    plot = False
    if (plot):
        plt.scatter([x[0] for x in estimated_positions], [x[1] for x in estimated_positions], label='estimated positions', marker='o')
        plt.scatter(ue_real_position[0], ue_real_position[1], label='real position', marker='o')
        plt.scatter([x[0] for x in gnbs_positions], [x[1] for x in gnbs_positions], label='gNBs', marker='x')
        plt.legend()
        plt.show()

    return train_distances_tmp, error_est_pos_list, sorted_errors_list, cdf_values_list, estimated_positions_list

speed_of_light = 299792458 # in meters per second
sampling_rate = 61.44e6 # in samples per second
gnbs_positions = [[0, 9], [3.5, 0], [7, 9]] # in meters

collection_names = ['toa_measurements_exp1','toa_measurements_exp2','toa_measurements_exp3','toa_measurements_exp4','toa_measurements_exp5']
ue_real_positions = [[0, 0],[7, 0],[3.5, 9],[3.5, 4.5],[3.5, 12]]

figs = []
axes = []

spoofing_index = 0.3
cdf_values_total_list = []
cdf_values_total_spoofed_list = []
cdf_values_total_spoofed_list_2 = []

sorted_errors_total_list = []
sorted_errors_total_spoofed_list = []
sorted_errors_total_spoofed_list_2 = []

spoofing_offset = np.array([2,5,10,20])
total_iterations = len(spoofing_offset)

for index, spoofing_value in enumerate(spoofing_offset):
    print(f'Completion: {(index / total_iterations) * 100}%')    
    print(f"spoofing_offset: {spoofing_value}")

    cdf_values_list = []
    cdf_values_list_spoofed = []
    cdf_values_list_spoofed_2 = []

    error_est_pos_list = []
    error_est_pos_list_spoofed = []
    error_est_pos_list_spoofed_2 = []

    sorted_errors_list = []
    sorted_errors_list_spoofed = []
    sorted_errors_list_spoofed_2 = []

    estimated_positions_list = []
    estimated_positions_list_spoofed = []
    estimated_positions_list_spoofed_2 = []

    # iterare through all collections, take also index of collection
    for collection_name, ue_real_position in zip(collection_names, ue_real_positions):

        collection = mongodb_init(collection_name)

        # Fetch the toaVal values from MongoDB and store them in a list
        toa_values = []
        for doc in collection.find():
            # if collection_name == 'toa_measurements_exp4': and doc['timestamp'] > 2023-07-05T11:59:52.641260 skip appending toa values
            if collection_name == 'toa_measurements_exp4' and datetime.datetime.fromisoformat(doc['timestamp']) > datetime.datetime.fromisoformat("2023-07-05T11:59:52.641260"):
                continue

            toa_values.append(doc['paramMap'][1]['toa']['toaVal'])

        toa_values_spoofed = np.copy(toa_values)
        toa_values_spoofed_2 = np.copy(toa_values)

        # convert toa_values to numpy array
        toa_values = np.array(toa_values)



        # apply the spoofing value offset to first row of the entries of toa_values_spoofed inner elements
        for i in range(math.floor(len(toa_values_spoofed)*(1-spoofing_index))):
            toa_values_spoofed[-i][0] = toa_values_spoofed[-i][0] + spoofing_value

            toa_values_spoofed_2[-i][0] = toa_values_spoofed_2[-i][0] + spoofing_value
            toa_values_spoofed_2[-i][1] = toa_values_spoofed_2[-i][1] + spoofing_value


        train_distances_tmp, error_est_pos_list, sorted_errors_list, cdf_values_list, estimated_positions_list = process_distances(toa_values, ue_real_position, gnbs_positions, error_est_pos_list, sorted_errors_list, cdf_values_list, estimated_positions_list, spoofing_index)

        train_distances_tmp_spoofed, error_est_pos_list_spoofed, sorted_errors_list_spoofed, cdf_values_list_spoofed, estimated_positions_list_spoofed = process_distances(toa_values_spoofed, ue_real_position, gnbs_positions, error_est_pos_list_spoofed, sorted_errors_list_spoofed, cdf_values_list_spoofed, estimated_positions_list_spoofed, spoofing_index)

        train_distances_tmp_spoofed_2, error_est_pos_list_spoofed_2, sorted_errors_list_spoofed_2, cdf_values_list_spoofed_2, estimated_positions_list_spoofed_2 = process_distances(toa_values_spoofed_2, ue_real_position, gnbs_positions, error_est_pos_list_spoofed_2, sorted_errors_list_spoofed_2, cdf_values_list_spoofed_2, estimated_positions_list_spoofed_2, spoofing_index)
        
    #compute the cdf  global using all the error values from all the experiments
    sorted_errors_total = np.sort(np.concatenate(error_est_pos_list))
    sorted_errors_total_spoofed = np.sort(np.concatenate(error_est_pos_list_spoofed))
    sorted_errors_total_spoofed_2 = np.sort(np.concatenate(error_est_pos_list_spoofed_2))

    sorted_errors_total_list.append(sorted_errors_total)
    sorted_errors_total_spoofed_list.append(sorted_errors_total_spoofed)
    sorted_errors_total_spoofed_list_2.append(sorted_errors_total_spoofed_2)

    cdf_values_total = 1. * np.arange(len(sorted_errors_total)) / (len(sorted_errors_total) - 1)
    cdf_values_total_spoofed = 1. * np.arange(len(sorted_errors_total_spoofed)) / (len(sorted_errors_total_spoofed) - 1)
    cdf_values_total_spoofed_2 = 1. * np.arange(len(sorted_errors_total_spoofed_2)) / (len(sorted_errors_total_spoofed_2) - 1)

    cdf_values_total_list.append(cdf_values_total)
    cdf_values_total_spoofed_list.append(cdf_values_total_spoofed)
    cdf_values_total_spoofed_list_2.append(cdf_values_total_spoofed_2)

'''
    # Loop over sorted_errors_list using enumerate to get the index of each list

    
    # Create a new figure and two subplots for each iteration
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('UE position estimation errors')

    # First subplot
    ax1.plot(sorted_errors_total, cdf_values_total, label='Mean')
    # add the cdf odf all experiments to the plot
    for i in range(len(collection_names)):
        ax1.plot(sorted_errors_list[i], cdf_values_list[i], label=f'Exp {i+1}')
    ax1.legend(loc='lower right', bbox_to_anchor=(1.0, 0.0))
    ax1.set_xlabel('Error (m)')
    ax1.set_ylabel('CDF')
    ax1.set_title('CDF of UE position estimation errors')
    ax1.grid()

    # Second subplot
    ax2.plot(sorted_errors_total_spoofed, cdf_values_total_spoofed, label='Mean spoofed')
    for i in range(len(collection_names)):
        ax2.plot(sorted_errors_list_spoofed[i], cdf_values_list_spoofed[i], label=f'Exp {i+1}')
    
    #set legend near the edge down right
    ax2.legend(loc='lower right', bbox_to_anchor=(1.0, 0.0))
    ax2.set_xlabel('Error (m)')
    ax2.set_ylabel('CDF')
    # set title of subplot specifing the spoofing offset
    ax2.set_title(f'CDF of UE position estimation errors, spoofing offset: {spoofing_offset[index]}')
    ax2.grid()
    
    #plt.show()
    if save_fig:
        #save fig to file appending spoofing offset to filename
        pickle.dump(fig,open(f'./plots/CDF_ue_pos_est_errors_spoofing_offset_{spoofing_offset[index]}_gnb_0.fig','wb'))
        # Append the figure and axes to the lists
    figs.append(fig)
    axes.append((ax1, ax2))
    
'''

#plot the cdf total spoofed and not spoofed for each spoofing offset in the same plot
fig, ax = plt.subplots(1, 1)
ax.plot(sorted_errors_total_list[0], cdf_values_total_list[0], label=f'Not spoofed')
for index, spoofing_value in enumerate(spoofing_offset):
    ax.plot(sorted_errors_total_spoofed_list[index], cdf_values_total_spoofed_list[index], label=f'gNB 0 spoofed, spoofing offset: {spoofing_value}')
    ax.plot(sorted_errors_total_spoofed_list_2[index], cdf_values_total_spoofed_list_2[index], label=f'gNB 0 and 1 spoofed, spoofing offset: {spoofing_value}')
ax.legend(loc='lower right', bbox_to_anchor=(1.0, 0.0))
ax.set_xlabel('Error (m)')
ax.set_ylabel('CDF')
#set x axis limit to 120
#ax.set_xlim([0, 120])
ax.set_title(f'CDF of UE position estimation errors')
ax.grid()
#save fig to file
if save_fig:
    fig.set_size_inches(19.2, 10.8)
    fig.savefig(f"./plots/CDF_ue_pos_est_errors_mean_spoofing.png", dpi=100)

    #save fig to file appending spoofing offset to filename
    pickle.dump(fig,open(f'./plots/CDF_ue_pos_est_errors_mean_spoofing.fig','wb'))



# Now you can display or save all the figures
plot = True
if plot:
    plt.show()





