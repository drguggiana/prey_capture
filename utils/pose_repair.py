import paths
import numpy as np
import functions_bondjango as bd
import pandas as pd
import processing_parameters
import os
from snakemake.sub_preprocess_S1 import process_corners
import random
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam

# define the likelihood threshold for the DLC points
likelihood_threshold = 0.1

# define the search string
search_string = processing_parameters.search_string

# define the target model
if 'miniscope' in search_string:
    target_model = 'video_experiment'
else:
    target_model = 'vr_experiment'

# get the queryset
file_set = bd.query_database(target_model, search_string)[:2]

# allocate memory to accumulate the trajectories
all_points = []

# run through the files
for files in file_set:
    raw_path = files['bonsai_path']
    calcium_path = files['bonsai_path'][:-4] + '_calcium.hdf5'

    file_path_dlc = files['bonsai_path'].replace('.csv', '_dlc.h5')
    # load the bonsai info
    raw_h5 = pd.read_hdf(file_path_dlc)
    # get the column names
    column_names = raw_h5.columns
    # take only the relevant columns
    # DLC in small arena
    filtered_traces = pd.DataFrame(raw_h5[[
        [el for el in column_names if ('mouseSnout' in el) and ('x' in el)][0],
        [el for el in column_names if ('mouseSnout' in el) and ('y' in el)][0],
        [el for el in column_names if ('mouseBarL' in el) and ('x' in el)][0],
        [el for el in column_names if ('mouseBarL' in el) and ('y' in el)][0],
        [el for el in column_names if ('mouseBarR' in el) and ('x' in el)][0],
        [el for el in column_names if ('mouseBarR' in el) and ('y' in el)][0],
        [el for el in column_names if ('mouseHead' in el) and ('x' in el)][0],
        [el for el in column_names if ('mouseHead' in el) and ('y' in el)][0],
        [el for el in column_names if ('mouseBody1' in el) and ('x' in el)][0],
        [el for el in column_names if ('mouseBody1' in el) and ('y' in el)][0],
        [el for el in column_names if ('mouseBody2' in el) and ('x' in el)][0],
        [el for el in column_names if ('mouseBody2' in el) and ('y' in el)][0],
        [el for el in column_names if ('mouseBody3' in el) and ('x' in el)][0],
        [el for el in column_names if ('mouseBody3' in el) and ('y' in el)][0],
        [el for el in column_names if ('mouseBase' in el) and ('x' in el)][0],
        [el for el in column_names if ('mouseBase' in el) and ('y' in el)][0],
        [el for el in column_names if ('cricketHead' in el) and ('x' in el)][0],
        [el for el in column_names if ('cricketHead' in el) and ('y' in el)][0],
        [el for el in column_names if ('cricketBody' in el) and ('x' in el)][0],
        [el for el in column_names if ('cricketBody' in el) and ('y' in el)][0],
    ]].to_numpy(), columns=['mouse_snout_x', 'mouse_snout_y', 'mouse_barl_x', 'mouse_barl_y',
                            'mouse_barr_x', 'mouse_barr_y', 'mouse_head_x', 'mouse_head_y',
                            'mouse_x', 'mouse_y', 'mouse_body2_x', 'mouse_body2_y',
                            'mouse_body3_x', 'mouse_body3_y', 'mouse_base_x', 'mouse_base_y',
                            'cricket_0_head_x', 'cricket_0_head_y', 'cricket_0_x', 'cricket_0_y'])

    # get the likelihoods
    likelihood_frame = pd.DataFrame(raw_h5[[
        [el for el in column_names if ('mouseHead' in el) and ('likelihood' in el)][0],
        [el for el in column_names if ('mouseBody1' in el) and ('likelihood' in el)][0],
        [el for el in column_names if ('mouseBody2' in el) and ('likelihood' in el)][0],
        [el for el in column_names if ('mouseBody3' in el) and ('likelihood' in el)][0],
        [el for el in column_names if ('mouseBase' in el) and ('likelihood' in el)][0],
        [el for el in column_names if ('cricketHead' in el) and ('likelihood' in el)][0],
        [el for el in column_names if ('cricketBody' in el) and ('likelihood' in el)][0],
    ]].to_numpy(), columns=['mouse_head', 'mouse', 'mouse_body2',
                            'mouse_body3', 'mouse_base',
                            'cricket_0_head', 'cricket_0'])

    # nan the trace where the likelihood is too low
    # for all the columns
    for col in likelihood_frame.columns:
        # get the vector for nans
        nan_vector = likelihood_frame[col] < likelihood_threshold
        # nan the points
        filtered_traces.loc[nan_vector, col+'_x'] = np.nan
        filtered_traces.loc[nan_vector, col+'_y'] = np.nan

    corner_info = pd.DataFrame(raw_h5[[
        [el for el in column_names if ('corner_UL' in el) and ('x' in el)][0],
        [el for el in column_names if ('corner_UL' in el) and ('y' in el)][0],
        [el for el in column_names if ('corner_BL' in el) and ('x' in el)][0],
        [el for el in column_names if ('corner_BL' in el) and ('y' in el)][0],
        [el for el in column_names if ('corner_BR' in el) and ('x' in el)][0],
        [el for el in column_names if ('corner_BR' in el) and ('y' in el)][0],
        [el for el in column_names if ('corner_UR' in el) and ('x' in el)][0],
        [el for el in column_names if ('corner_UR' in el) and ('y' in el)][0],
    ]].to_numpy(), columns=['corner_UL_x', 'corner_UL_y', 'corner_BL_x', 'corner_BL_y',
                            'corner_BR_x', 'corner_BR_y', 'corner_UR_x', 'corner_UR_y'])
    # get the corners
    corner_points = process_corners(corner_info)

    # accumulate the points
    all_points.append(filtered_traces)

# define the amount of delay to include
delay = 2
# define the learning rate
learning_rate = 0.001
# define the test train split
validation_split = 0.2
# get the target column names
column_list = [el for el in all_points[0].columns if 'mouse' in el]
# get the number of features
number_features = len(column_list)
# allocate memory for the data
data_matrix = []

# for all the files
for files in all_points:
    # select the features to use
    current_points = np.array(files.loc[:, column_list])
    # get the number of timepoints
    number_timepoints = current_points.shape[0]

    # allocate memory for the design matrix
    design_matrix = np.zeros((number_timepoints, number_features*delay + number_features))
    # pad the data according to the delay
    current_points = np.concatenate((np.zeros((delay, number_features)), current_points), axis=0)
    # assemble the design matrix
    # for all the points
    for points in np.arange(number_timepoints):
        design_matrix[points, :] = current_points[points:points+delay+1, :].reshape([1, -1])
    # save
    data_matrix.append(design_matrix)

# concatenate the data
data_matrix = np.concatenate(data_matrix, axis=0)

# get a vector with the rows with NaN
nan_vector = np.any(np.isnan(data_matrix), axis=1)
# eliminate the points with NaNs in them
nonan_matrix = data_matrix[~nan_vector, :]
nan_matrix = data_matrix[nan_vector, :]

# separate the fitting data and the target
X = nonan_matrix[:, number_features:]
y = nonan_matrix[:, :number_features]

# split into training and test sets

# # get a selection vector based on the number of points and the desired split
# train_idx = random.sample(list(np.arange(X.shape[0])), np.int(np.round(test_train_split*X.shape[0])))
# train_vector = np.zeros([X.shape[0]])
# train_vector[train_idx] = 1
# train_vector = train_vector == 1
# validation_vector = ~train_vector
#
# train_X = X[train_vector, :]
# train_y = y[train_vector, :]
#
# validation_X = X[validation_vector, :]
# validation_y = y[validation_vector, :]

# create the network
model = tf.keras.Sequential()
model.add(layers.Dense(50, activation='relu'))
model.add(layers.Dense(16, activation='softmax'))

model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')

model.fit(X, y, validation_split=validation_split, batch_size=128, epochs=500, shuffle=True, verbose=2)

print('yay')
