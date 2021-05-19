# imports
from tkinter import filedialog
# import functions_plotting as fplot
from functions_misc import tk_killwindow
import functions_preprocessing as fp
from functions_io import parse_path
import functions_kinematic as fk
import numpy as np
import pandas as pd
import datetime
import paths
import tensorflow as tf
import os


# run inference on the incomplete pose points
def pose_repair(data_in, model_path, delay=4):
    """Use a pre-trained neural network to infer missing points in the data"""
    # import the model
    model = tf.keras.models.load_model(model_path)

    # copy the data
    data_out = data_in.copy()

    # get a vector with the nan rows
    # nan_vector = np.any(np.isnan(data_in), axis=1)
    nan_vector = np.isnan(data_in)
    # select the points with NaNs for inference
    # current_points = data_in[nan_vector, :]
    current_points = data_in
    # assemble the design matrix
    # define the constants (based on model trained)
    number_features = current_points.shape[1]
    # get the number of timepoints
    number_timepoints = current_points.shape[0]

    # allocate memory for the design matrix
    design_matrix = np.zeros((number_timepoints, number_features*delay + number_features))
    # # pad the data according to the delay
    # current_points = np.concatenate((np.zeros((delay, number_features)), current_points), axis=0)
    # # assemble the design matrix
    # # for all the points
    # for points in np.arange(number_timepoints):
    #     design_matrix[points, :] = current_points[points:points+delay+1, :].reshape([1, -1])
    #
    # # get the input data
    # # x = design_matrix[:, number_features:]
    # x = design_matrix

    current_points = np.concatenate((np.zeros((int(delay / 2), number_features)), current_points,
                                     np.zeros((int(delay / 2), number_features))), axis=0)
    # assemble the design matrix
    # for all the points
    for points in np.arange(number_timepoints):
        design_matrix[points, :] = current_points[points:points + delay + 1, :].reshape([1, -1])

    # extract the prediction and validation matrices
    design_matrix[np.isnan(design_matrix)] = 0

    # eval_X = design_matrix[:, number_features:]
    # eval_X = design_matrix
    # eval_y = design_matrix[:, :number_features]
    x = np.concatenate((design_matrix[:, :number_features * int(delay / 2)],
                        design_matrix[:, -number_features * int(delay / 2):]), axis=1)
    # turn NaNs into 0
    x[np.isnan(x)] = 0
    # run the inference
    predicted_points = model.predict(x)
    # replace the points in the input data
    # data_out[nan_vector, :] = predicted_points[nan_vector, :]
    data_out[nan_vector] = predicted_points[nan_vector]
    return data_out


def run_preprocess(file_path_bonsai, save_file, file_info,
                   kernel_size=21, max_step=300, max_length=50):
    """Preprocess the bonsai file"""
    # parse the bonsai file
    parsed_data = fp.parse_bonsai(file_path_bonsai)
    # define the target columns
    tar_columns = ['cricket_0_x', 'cricket_0_y']

    # parse the path
    parsed_path = parse_path(file_path_bonsai)
    # animal_data_bonsai.append(np.array(parsed_data))
    files = np.array(parsed_data)

    # get the time
    files, time, dates = fp.get_time(files)

    # trim the trace
    if parsed_path['rig'] == 'miniscope':
        files, time = fp.trim_bounds(files, time, dates)
    else:
        # TODO: add proper bound trimming based on ML
        print('yay')

    # assemble a data frame with the data
    data = pd.DataFrame(files, columns=['mouse_x', 'mouse_y', 'cricket_0_x', 'cricket_0_y'])

    # now remove the discontinuities in the trace

    # median filter only the cricket trace
    filtered_traces = fp.median_discontinuities(data, tar_columns, kernel_size)

    # eliminate isolated points
    filtered_traces = fp.eliminate_singles(filtered_traces)

    # eliminate discontinuities before interpolating
    filtered_traces = fp.nan_large_jumps(filtered_traces, tar_columns, max_step, max_length)

    # interpolate the NaN stretches
    filtered_traces = fp.interpolate_segments(filtered_traces, np.nan)

    # add the time field to the dataframe
    filtered_traces['time_vector'] = time

    # also the mouse and the date
    filtered_traces['mouse'] = parsed_path['animal']
    filtered_traces['datetime'] = parsed_path['datetime']

    # eliminate the cricket if there is no real cricket or this is a VScreen experiment
    if ('nocricket' in file_info['notes'] and 'VR' in file_info['rig']) or \
            ('real' not in file_info['notes'] and 'VPrey' in file_info['rig']) or \
            ('VScreen' in file_info['rig']):
        # for all the columns
        for column in filtered_traces.columns:
            if 'cricket' in column:
                filtered_traces.drop([column], inplace=True, axis=1)

    # create the file name
    filtered_traces.to_hdf(save_file, key='full_traces', mode='w', format='table')

    # append to the outpath list
    out_path = save_file

    return out_path, filtered_traces


def run_dlc_preprocess(file_path_bonsai, file_path_dlc, save_file, file_info, kernel_size=5):
    """Extract the relevant columns from the dlc file and rename"""
    # define the threshold for trimming the trace (in pixels for now)
    trim_cutoff = 200
    # define the likelihood threshold for the DLC points
    likelihood_threshold = 0.8
    # define the threshold for max amount of cm allowed between points within an animal
    animal_threshold = 4
    cricket_threshold = 10
    # just return the output path
    out_path = save_file
    # load the bonsai info
    raw_h5 = pd.read_hdf(file_path_dlc)
    # get the column names
    column_names = raw_h5.columns
    # take only the relevant columns
    try:
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
            [el for el in column_names if ('mouseSnout' in el) and ('likelihood' in el)][0],
            [el for el in column_names if ('mouseBarL' in el) and ('likelihood' in el)][0],
            [el for el in column_names if ('mouseBarR' in el) and ('likelihood' in el)][0],
            [el for el in column_names if ('mouseHead' in el) and ('likelihood' in el)][0],
            [el for el in column_names if ('mouseBody1' in el) and ('likelihood' in el)][0],
            [el for el in column_names if ('mouseBody2' in el) and ('likelihood' in el)][0],
            [el for el in column_names if ('mouseBody3' in el) and ('likelihood' in el)][0],
            [el for el in column_names if ('mouseBase' in el) and ('likelihood' in el)][0],
            [el for el in column_names if ('cricketHead' in el) and ('likelihood' in el)][0],
            [el for el in column_names if ('cricketBody' in el) and ('likelihood' in el)][0],
        ]].to_numpy(), columns=['mouse_snout', 'mouse_barl', 'mouse_barr', 'mouse_head', 'mouse', 'mouse_body2',
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
        corner_points = fp.process_corners(corner_info)
        # if file_info['result'] != 'habi':
        #     # interpolate the position of the cricket assuming stationarity
        #     filtered_traces = fp.interpolate_animals(filtered_traces, np.nan)
        # # eliminate points that separate from either mouse or cricket more than a threshold
        # filtered_traces = \
        #     fp.maintain_animals(filtered_traces, animal_threshold,
        #                         corner_points, paths.arena_coordinates['miniscope'])
        # add the cricket position under the mouse when it is not visible and last position was close to the mouse
        # filtered_traces = \
        #     fp.infer_cricket_position(filtered_traces, cricket_threshold,
        #                               corner_points, paths.arena_coordinates['miniscope'])
        # set final position of cricket, assume that cricket is under mouse whenever it is not observed
        # # for succ trials, it stops when mouse stops
        # if 'succ' in file_path_dlc:
        #     # check if the last point for the cricket is a NaN
        #     if np.isnan(filtered_traces['cricket_0_x'].iloc[-1]):
        #         # if so, make it the position of the mouse head
        #         filtered_traces[['cricket_0_x', 'cricket_0_y']].iloc[-1] = \
        #             filtered_traces[['mouse_head_x', 'mouse_head_y']].iloc[-1]

    except IndexError:
        # DLC in VR arena
        filtered_traces = pd.DataFrame(raw_h5[[
            [el for el in column_names if ('head' in el) and ('x' in el)][0],
            [el for el in column_names if ('head' in el) and ('y' in el)][0],
            [el for el in column_names if ('body_center' in el) and ('x' in el)][0],
            [el for el in column_names if ('body_center' in el) and ('y' in el)][0],
            [el for el in column_names if ('tail_base' in el) and ('x' in el)][0],
            [el for el in column_names if ('tail_base' in el) and ('y' in el)][0],
            [el for el in column_names if ('cricket' in el) and ('x' in el)][0],
            [el for el in column_names if ('cricket' in el) and ('y' in el)][0],
        ]].to_numpy(), columns=['mouse_head_x', 'mouse_head_y', 'mouse_x', 'mouse_y', 'mouse_base_x', 'mouse_base_y',
                                'cricket_0_x', 'cricket_0_y'])

        # The camera that records video in the VR arena flips the video about the
        # horizontal axis when saving. To correct, flip the y coordinates from DLC
        filtered_traces = fp.flip_DLC_y(filtered_traces)
        # output an empty for the corners
        corner_points = []

    # eliminate the cricket if there is no real cricket or this is a VScreen experiment
    if ('nocricket' in file_info['notes'] and 'VR' in file_info['rig']) or \
            ('nocricket' in file_info['notes'] and 'miniscope' in file_info['rig']) or \
            ('test' in file_info['result'] and 'VPrey' in file_info['rig']) or \
            ('VScreen' in file_info['rig']) or \
            (file_info['result'] == 'habi'):
        # for all the columns
        for column in filtered_traces.columns:
            if 'cricket' in column:
                filtered_traces.drop([column], inplace=True, axis=1)

    # # trim the trace at the last large jump of the mouse trajectory (i.e when the mouse enters the arena)
    # # do it after median filtering to prevent the single point errors to trim the video too much
    # # calculate the displacement of the mouse center
    # mouse_displacement = np.sqrt((np.diff(filtered_traces['mouse_x']))**2 + (np.diff(filtered_traces['mouse_y']))**2)
    # cutoff_frame = np.argwhere(mouse_displacement > trim_cutoff)
    # # check if it's empty. if so, don't cutoff anything
    # if cutoff_frame.shape[0] > 0:
    #     cutoff_frame = cutoff_frame[-1][0]
    # else:
    #     cutoff_frame = 0

    # save a copy of the untrimmed traces for later
    untrimmed = filtered_traces.copy()
    # if it's not the miniscope rig, use the simpler trimming
    if file_info['rig'] != 'miniscope':
        # define the frame bounds as empty
        frame_bounds = pd.DataFrame(np.array([0, filtered_traces.shape[0],
                                    filtered_traces.shape[0]]).reshape([1, 3]),
                                    columns=['start', 'end', 'original_length'])

        # trim the trace at the first mouse main body not nan
        cutoff_frame = np.argwhere(~np.isnan(filtered_traces['mouse_x'].to_numpy()))
        # if no cutoff is found, don't cutoff anything
        if cutoff_frame.shape[0] > 0:
            cutoff_frame = cutoff_frame[0][0]
        else:
            cutoff_frame = 0
        # save the cutoff frame
        frame_bounds.loc[0, 'start'] = cutoff_frame
        # perform the trimming and reset index
        filtered_traces = filtered_traces.iloc[cutoff_frame:, :].reset_index(drop=True)
    else:  # use the neural net and alternative trimming

        # trim the trace based on the onset and offset of movement
        filtered_traces, frame_bounds = fp.trim_to_movement(file_info['result'],
                                                            filtered_traces, paths.arena_coordinates['miniscope'],
                                                            corner_points)

        # repair the mouse trajectory using a neural net

        # get only the mouse data
        column_list = [el for el in filtered_traces.columns if ('mouse' in el) and ('_x' in el)]
        column_list += [el for el in filtered_traces.columns if ('mouse' in el) and ('_y' in el)]
        mouse_data = filtered_traces.loc[:, column_list].to_numpy()

        # repair the trajectory
        mouse_data = pose_repair(mouse_data, os.path.join(paths.pose_repair_path, 'mouse'), delay=2)

        # put the values back in the main df
        filtered_traces.loc[:, column_list] = mouse_data

    if file_info['result'] != 'habi':
        # interpolate the position of the cricket assuming stationarity
        filtered_traces = fp.interpolate_animals(filtered_traces, np.nan,
                                                 paths.arena_coordinates[file_info['rig']], corner_points, untrimmed)
    # filtered_traces = \
    #     fp.infer_cricket_position(filtered_traces, cricket_threshold,
    #                               corner_points, paths.arena_coordinates['miniscope'])
    # # interpolate the position of the cricket assuming stationarity
    # filtered_traces = fp.interpolate_animals(filtered_traces, np.nan)

    # # get only the cricket data
    # column_list = [el for el in filtered_traces.columns if ('cricket' in el) and ('_x' in el)]
    # column_list += [el for el in filtered_traces.columns if ('cricket' in el) and ('_y' in el)]
    # cricket_data = filtered_traces.loc[:, column_list].to_numpy()
    #
    # # repair the trajectory
    # cricket_data = pose_repair(cricket_data, os.path.join(paths.pose_repair_path, 'cricket'), delay=2)
    #
    # # put the values back in the main df
    # filtered_traces.loc[:, column_list] = cricket_data

    # # use an ARIMA to infer the low likelihood points
    # filtered_traces = fp.median_arima(filtered_traces, filtered_traces.columns)
    # interpolate the NaN stretches
    # filtered_traces = fp.interpolate_segments(filtered_traces, np.nan)
    # filtered_traces = fp.interpolate_animals(filtered_traces, np.nan)

    # median filter the traces
    filtered_traces = fp.median_discontinuities(filtered_traces, filtered_traces.columns, kernel_size)

    # trim the trace before and after the places with speed above threshold
    # calculate speed trace for the mouse

    # filtered_traces

    # find the places where there is no pixel movement in any axis and NaN those

    # cricket_nonans_x = filtered_traces['cricket_x']
    # cricket_nonans_y = filtered_traces['cricket_y']
    # find the places with tracking off via delta pixel and NaN them
    # filtered_traces = find_frozen_tracking(filtered_traces, stretch_length=15)
    # eliminate discontinuities before interpolating
    # filtered_traces = nan_large_jumps(filtered_traces, ['cricket_x', 'cricket_y'], max_step=200, max_length=150)
    # eliminate large jumps
    # filtered_traces = nan_jumps_dlc(filtered_traces)
    # # eliminate isolated points
    # filtered_traces = eliminate_singles(filtered_traces)
    # interpolate the NaN stretches
    # filtered_traces = interpolate_segments(filtered_traces, np.nan)
    # parse the bonsai file for the time stamps
    timestamp = []
    bonsai_data = []
    with open(file_path_bonsai) as f:
        for ex_line in f:
            ex_list = ex_line.split(' ')
            ex_list.remove('\n')
            timestamp.append(ex_list.pop())
            bonsai_data.append([float(el) for el in ex_list])

    # parse the path
    parsed_path = parse_path(file_path_bonsai)
    # add the time stamps to the main dataframe
    time = [datetime.datetime.strptime(el[:-7], '%Y-%m-%dT%H:%M:%S.%f')
            for el in timestamp[frame_bounds.loc[0, 'start']:frame_bounds.loc[0, 'end']-1]]
    # if time is missing frames, skip them from the end and show a warning (checked comparing the traces)
    if len(time) < filtered_traces.shape[0]:
        # calculate the delta
        delta_frame = filtered_traces.shape[0] - len(time)
        # show the warning
        print('Extra frames in video: %i' % delta_frame)
        # trim filtered traces
        filtered_traces = filtered_traces.iloc[:-delta_frame, :]

    filtered_traces['time_vector'] = [(el - time[0]).total_seconds() for el in time]

    # also the mouse and the date
    filtered_traces['mouse'] = parsed_path['animal']
    filtered_traces['datetime'] = parsed_path['datetime']

    coordinate_columns = [el for el in filtered_traces.columns if 'mouse_' in el]
    coordinate_columns += [el for el in filtered_traces.columns if 'cricket_' in el]
    # check for nans
    if np.any(np.isnan(filtered_traces[coordinate_columns].to_numpy())):
        raise ValueError(f'NaN value found in file {file_info["slug"]}')
    # turn corner points into a dataframe
    corner_frame = pd.DataFrame(corner_points.T, columns=['UL', 'BL', 'BR', 'UR'])

    return out_path, filtered_traces, corner_frame, frame_bounds


def extract_motive(file_path_motive, rig):
    """Extract the encoded traces in the current motive file"""

    # parse the path
    parsed_path = parse_path(file_path_motive)

    # set up empty variables for arena corner and obstacle positions
    arena_corners = []
    obstacle_positions = []

    # read the data
    try:
        raw_data = pd.read_csv(file_path_motive, header=None)

        # parse the path
        parsed_path = parse_path(file_path_motive)
        # select the appropriate header
        if rig == 'VR':
            # if it's before the sync files, exclude the last column
            if parsed_path['datetime'] <= datetime.datetime(year=2019, month=11, day=10):
                column_names = ['time_m', 'mouse_y_m', 'mouse_z_m', 'mouse_x_m',
                                'mouse_yrot_m', 'mouse_zrot_m', 'mouse_xrot_m',
                                'vrcricket_0_y_m', 'vrcricket_0_z_m', 'vrcricket_0_x_m'
                                ]
            elif parsed_path['datetime'] <= datetime.datetime(year=2020, month=6, day=22):
                column_names = ['time_m', 'mouse_y_m', 'mouse_z_m', 'mouse_x_m'
                                , 'mouse_yrot_m', 'mouse_zrot_m', 'mouse_xrot_m'
                                , 'vrcricket_0_y_m', 'vrcricket_0_z_m', 'vrcricket_0_x_m'
                                , 'color_factor'
                                ]
            else:
                column_names = ['time_m', 'mouse_y_m', 'mouse_z_m', 'mouse_x_m'
                                , 'mouse_yrot_m', 'mouse_zrot_m', 'mouse_xrot_m'
                                , 'color_factor'
                                ]
        else:
            # get the number of vr crickets
            # TODO: make this not arbitrary
            cricket_number = (raw_data.shape[1] - 8)/10
            # define the cricket template
            cricket_template = ['_y', '_z', '_x', '_yrot', '_zrot', '_xrot',
                                '_speed', '_state', '_motion', '_encounter']
            # assemble the cricket fields
            cricket_fields = ['vrcricket_'+str(int(number))+el
                              for number in np.arange(cricket_number) for el in cricket_template]

            column_names = ['time_m', 'mouse_y_m', 'mouse_z_m', 'mouse_x_m'
                            , 'mouse_yrot_m', 'mouse_zrot_m', 'mouse_xrot_m'
                            ] + cricket_fields + [
                            'color_factor'
                            ]
        # create the column name dictionary
        column_dict = {idx: column for idx, column in enumerate(column_names)}

        # # read the data
        # raw_data = pd.read_csv(file_path_motive, names=column_names)
        raw_data.rename(columns=column_dict, inplace=True)

    except:
        # This occurs for files that have more complicated headers
        arena_corners, obstacle_positions, df_line = fp.read_motive_header(file_path_motive)
        raw_data = pd.read_csv(file_path_motive, header=0, skiprows=df_line)

        # Correct for mistakes in coordinate convention
        if rig == 'VScreen':
            if parsed_path['datetime'] <= datetime.datetime(year=2020, month=11, day=24):
                column_names = ['time_m', 'trial_num',
                                'mouse_y_m', 'mouse_z_m', 'mouse_x_m',
                                'mouse_yrot_m', 'mouse_zrot_m', 'mouse_xrot_m',
                                'target_y_m', 'target_z_m', 'target_x_m', 'color_factor']

                # create the column name dictionary
                column_dict = {old_col: column for old_col, column in zip(raw_data.columns, column_names)}
                raw_data.rename(columns=column_dict, inplace=True)

                # Arena coordinates need to be put into a format that aligns with DLC tracking.
                # Need to negate the x coordinate to match video mirroring, then flip the x and z coordinates
                arena_corners_temp = arena_corners.copy()
                arena_corners = [[corner[1], corner[0]] for corner in arena_corners_temp]

    return raw_data, arena_corners, obstacle_positions


# if __name__ == '__main__':
#     # get rid of the tk main window
#     tk_killwindow()
#
#     # define the save path
#     save_path = pre_processed_path
#     # define the base loading path
#     base_path_bonsai = bonsai_path
#     # select the files to process
#     file_path = filedialog.askopenfilenames(initialdir=base_path_bonsai, filetypes=(("csv files", "*.csv"),))
#
#     # run the preprocessing
#     run_preprocess(
#         file_path,
#         save_path,
#         # define the kernel size for the median filter
#         kernel_size=21,
#         # define the maximum amount of an allowed jump in the trajectory per axis, in pixels
#         max_step=300,
#         # define the maximum length of a jump to be interpolated
#         max_length=50
#     )

