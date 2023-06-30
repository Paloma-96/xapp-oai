from func import *

plot_calibration = False
plot_rolling_mean = True
plot_wavelet_effect = False
plot_trilateration = False
############################################# connect DB
conn = DBConnect()
cur = conn.cursor()
cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cur.fetchall()

############################################ Ask user to select 3 tables 
print("Tables in database:")
for i, table in enumerate(tables):
    print(f"{i+1}. {table[0]}")

selected_tables = [1,3,4]
#selected_tables.append(input("Select first table and press enter to continue "))
#selected_tables.append(input("Select second table and press enter to continue "))
#selected_tables.append(input("Select third table and press enter to continue "))


table_names = [tables[int(table)-1][0] for table in selected_tables]
dataframes = {table_name: fetch_data(conn, table_name) for table_name in table_names}

dataframes = enforce_same_number_of_rows(dataframes)

############################################# sync dataframes
for table_name, df in dataframes.items():
    #Subtract the first timestamp of each 'gnb_id' group from the corresponding timestamps and convert to seconds
    df['timestamp_in_seconds'] = df['timestamp'].transform(lambda x: (x - x.iloc[0]).dt.total_seconds())


############################################# set variables
gnb_ids = [0, 1, 2]
window_size = 20
gnb_positions = [[0, 9], [12, 9], [6, 0]]
ue_real_position = [15, 9]


############################################# process dataframes

# remove rows with toa_val > 20 and < 0
for table_name, df in dataframes.items():
    df = df[(df['toa_val'] < 20) & (df['toa_val'] > 0)]
    dataframes[table_name] = df


#knowing the ue_real position and the gnb_positions, perform calibration of the toa_val

# using gnb_positions and ue_real_position, calculate the ideal toa_val between ue and each gnb
ideal_toa_val_0 = distance_to_toa_val(distance_between_points(gnb_positions[0], ue_real_position), 61.44 * 1e6)
ideal_toa_val_1 = distance_to_toa_val(distance_between_points(gnb_positions[1], ue_real_position), 61.44 * 1e6)
ideal_toa_val_2 = distance_to_toa_val(distance_between_points(gnb_positions[2], ue_real_position), 61.44 * 1e6)

ideal_toa = [ideal_toa_val_0, ideal_toa_val_1, ideal_toa_val_2]

# compute the mean error between the real toa_val and the toa_val measured by the gnb for each gnb_id
mean_error_0 = dataframes[table_names[0]]['toa_val'][350:450].mean() - ideal_toa_val_0
mean_error_1 = dataframes[table_names[1]]['toa_val'][350:450].mean() - ideal_toa_val_1
mean_error_2 = dataframes[table_names[2]]['toa_val'][350:450].mean() - ideal_toa_val_2


for table_name, df in dataframes.items():
    df['calibrated_toa_val'] = df['toa_val'].apply(lambda x: x - mean_error_0 if table_name == table_names[0] else x - mean_error_1 if table_name == table_names[1] else x - mean_error_2)

if plot_calibration:
    #plot the toa_val, calibrated toa_val and ideal toa_val for each dataframe on a different axes
    plt.figure(figsize=(10, 6))
    for i, (table_name, df) in enumerate(dataframes.items()):
        #plt.plot(df['timestamp_in_seconds'].to_numpy(), df['toa_val'].to_numpy(), label=f"toa_val {table_name}")
        plt.plot(df['timestamp_in_seconds'].to_numpy(), df['calibrated_toa_val'].to_numpy(), label=f"calibrated_toa_val {table_name}")
        plt.plot(df['timestamp_in_seconds'].to_numpy(), [ideal_toa[i]] * len(df), label=f"ideal_toa_val {table_name}")
    plt.xlabel("Time (seconds)")
    plt.ylabel("toa_val")
    plt.legend()
    plt.title("Effect of the calibration")
    plt.show()


############################################# apply calibrated_toa_val rolling mean to each dataframe
for table_name, df in dataframes.items():

    df.loc[:, 'cal_toa_val_window'] = df['calibrated_toa_val'].rolling(window=window_size).mean()
    df.loc[:, 'timestamps_window'] = df['timestamp_in_seconds'].rolling(window=window_size).mean()

    #clean SMA_window_size column from NaN values and corresponding MA_timestamps
    df = df.dropna()

# Plot the toa_val and SMA_window_size values for each dataframe on a different axes

if plot_rolling_mean:
    plt.figure(figsize=(10, 6))
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    # Loop through each df and plot the moving average values
    for i, (table_name, df) in enumerate(dataframes.items()):

        # Plot the data
        axes[i].plot(df['timestamp_in_seconds'].to_numpy(), df['calibrated_toa_val'].to_numpy(), label=f"calibrated_toa_val {table_name}")
        axes[i].plot(df['timestamps_window'].to_numpy(), df['cal_toa_val_window'].to_numpy(), label=f"cal_toa_val_window {table_name}")
        axes[i].set_ylabel("toa_val")
        axes[i].legend()

    plt.tight_layout()
    plt.title("Effect of the rolling mean")
    plt.show()



# save to files the dataframes
#dataframes[table_names[0]].to_csv(f'{table_names[0]}.csv')
#dataframes[table_names[1]].to_csv(f'{table_names[1]}.csv')
#dataframes[table_names[2]].to_csv(f'{table_names[2]}.csv')

############################################# cyclical effect removal
for table_name, df in dataframes.items():
    df = df.dropna()
    #detrended_data_signal_detrend = signal.detrend(df['cal_toa_val_window'].to_numpy())

    df = wavelet_filtering(df, plot_wavelet_effect)
    dataframes[table_name] = df


dataframes[table_names[0]]['dis_val'] = dataframes[table_names[0]]['detrended_signal'].apply(lambda x: toa_val_to_distance(x, 61.44 * 1e6))
dataframes[table_names[1]]['dis_val'] = dataframes[table_names[1]]['detrended_signal'].apply(lambda x: toa_val_to_distance(x, 61.44 * 1e6))
dataframes[table_names[2]]['dis_val'] = dataframes[table_names[2]]['detrended_signal'].apply(lambda x: toa_val_to_distance(x, 61.44 * 1e6))

# apply a window mean to each dataframe and save the results in a new column called 'dis_val_rolling_mean'
#dataframes[table_names[0]]['dis_val_rolling_mean'] = dataframes[table_names[0]]['dis_val'].rolling(window=window_size).mean()
#dataframes[table_names[1]]['dis_val_rolling_mean'] = dataframes[table_names[1]]['dis_val'].rolling(window=window_size).mean()
#dataframes[table_names[2]]['dis_val_rolling_mean'] = dataframes[table_names[2]]['dis_val'].rolling(window=window_size).mean()


ue_positions = []
# iterate through each row of the dataframe and apply the trilateration_solver function
for index, row in dataframes[table_names[0]].iterrows():
    if all(0 <= index < len(df) for df in dataframes.values()):
        distances = [
            dataframes[table_names[0]].iloc[index]['dis_val'],
            dataframes[table_names[1]].iloc[index]['dis_val'],
            dataframes[table_names[2]].iloc[index]['dis_val']
        ]
        ue_positions.append(trilateration_non_linear_least_squares(gnb_positions, distances))


ue_positions_df = pd.DataFrame(ue_positions, columns=['x', 'y'])

if plot_trilateration:
    #plot the UE positions and the gNB positions on the same figure using labels gnb_0, gnb_1, gnb_2
    plt.figure(figsize=(10, 6))
    plt.scatter(ue_positions_df['x'], ue_positions_df['y'], label="UE positions")
    plt.scatter(gnb_positions[0][0], gnb_positions[0][1], label="gNB 0")
    plt.scatter(gnb_positions[1][0], gnb_positions[1][1], label="gNB 1")
    plt.scatter(gnb_positions[2][0], gnb_positions[2][1], label="gNB 2")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.title("Trilateration")
    plt.show()

