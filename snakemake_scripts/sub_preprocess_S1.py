# imports
from tkinter import filedialog
import functions_plotting as fp
from paths import *
from functions_misc import tk_killwindow
from functions_preprocessing import trim_bounds, median_discontinuities, interpolate_segments, eliminate_singles, \
    nan_large_jumps, find_frozen_tracking, nan_jumps_dlc, get_time, parse_bonsai, read_motive_header, flip_DLC_y
from functions_io import parse_path
import numpy as np
import pandas as pd
import datetime


def run_preprocess(file_path_bonsai, save_file, file_info,
                   kernel_size=21, max_step=300, max_length=50):
    """Preprocess the bonsai file"""
    # parse the bonsai file
    parsed_data = parse_bonsai(file_path_bonsai)
    # define the target columns
    tar_columns = ['cricket_0_x', 'cricket_0_y']

    # parse the path
    parsed_path = parse_path(file_path_bonsai)
    # animal_data_bonsai.append(np.array(parsed_data))
    files = np.array(parsed_data)

    # get the time
    files, time, dates = get_time(files)

    # trim the trace
    if parsed_path['rig'] == 'miniscope':
        files, time = trim_bounds(files, time, dates)
    else:
        # TODO: add proper bound trimming based on ML
        print('yay')

    # assemble a data frame with the data
    data = pd.DataFrame(files, columns=['mouse_x', 'mouse_y', 'cricket_0_x', 'cricket_0_y'])

    # now remove the discontinuities in the trace

    # median filter only the cricket trace
    filtered_traces = median_discontinuities(data, tar_columns, kernel_size)

    # eliminate isolated points
    filtered_traces = eliminate_singles(filtered_traces)

    # eliminate discontinuities before interpolating
    filtered_traces = nan_large_jumps(filtered_traces, tar_columns, max_step, max_length)

    # interpolate the NaN stretches
    filtered_traces = interpolate_segments(filtered_traces, np.nan)

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


def run_dlc_preprocess(file_path_bonsai, file_path_dlc, save_file, file_info, kernel_size=21):
    """Extract the relevant columns from the dlc file and rename"""
    # TODO: convert everything to real distance
    # define the threshold for trimming the trace (in pixels for now)
    trim_cutoff = 200
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
            [el for el in column_names if ('mouseHead' in el) and ('x' in el)][0],
            [el for el in column_names if ('mouseHead' in el) and ('y' in el)][0],
            [el for el in column_names if ('mouseBody' in el) and ('x' in el)][0],
            [el for el in column_names if ('mouseBody' in el) and ('y' in el)][0],
            [el for el in column_names if ('mouseBase' in el) and ('x' in el)][0],
            [el for el in column_names if ('mouseBase' in el) and ('y' in el)][0],
            [el for el in column_names if ('cricketHead' in el) and ('x' in el)][0],
            [el for el in column_names if ('cricketHead' in el) and ('y' in el)][0],
            [el for el in column_names if ('cricketBody' in el) and ('x' in el)][0],
            [el for el in column_names if ('cricketBody' in el) and ('y' in el)][0],
        ]].to_numpy(), columns=['mouse_head_x', 'mouse_head_y', 'mouse_x', 'mouse_y', 'mouse_base_x', 'mouse_base_y',
                                'cricket_0_head_x', 'cricket_0_head_y', 'cricket_0_x', 'cricket_0_y'])
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
        filtered_traces = flip_DLC_y(filtered_traces)

    # eliminate the cricket if there is no real cricket or this is a VScreen experiment
    if ('nocricket' in file_info['notes'] and 'VR' in file_info['rig']) or \
            ('test' in file_info['result'] and 'VPrey' in file_info['rig']) or \
            ('VScreen' in file_info['rig']):
        # for all the columns
        for column in filtered_traces.columns:
            if 'cricket' in column:
                filtered_traces.drop([column], inplace=True, axis=1)

    # TODO: need to take advantage of the constraints between points to filter the data
    # median filter the traces
    filtered_traces = median_discontinuities(filtered_traces, filtered_traces.columns, kernel_size)
    # trim the trace at the last large jump of the mouse trajectory (i.e when the mouse enters the arena)
    # do it after median filtering to prevent the single point errors to trim the video too much
    # calculate the displacement of the mouse center
    mouse_displacement = np.sqrt((np.diff(filtered_traces['mouse_x']))**2 + (np.diff(filtered_traces['mouse_y']))**2)
    cutoff_frame = np.argwhere(mouse_displacement > trim_cutoff)
    # check if it's empty. if so, don't cutoff anything
    if cutoff_frame.shape[0] > 0:
        cutoff_frame = cutoff_frame[-1][0]
    else:
        cutoff_frame = 0

    # perform the trimming and reset index
    filtered_traces = filtered_traces.iloc[cutoff_frame:, :].reset_index(drop=True)
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
    # turn bonsai_data into an array
    # bonsai_data = np.array(bonsai_data)
    # # plot trajectories
    # fp.plot_2d([[filtered_traces['mouse_x'].to_numpy(),
    #              cricket_nonans_x,
    #              filtered_traces['cricket_x'].to_numpy()],
    #
    #              # bonsai_data[:, 0]],
    #             [filtered_traces['mouse_y'].to_numpy(),
    #              cricket_nonans_y,
    #              filtered_traces['cricket_y'].to_numpy()]
    #             ], rows=2, dpi=100)
    # parse the path
    parsed_path = parse_path(file_path_bonsai)
    # add the time stamps to the main dataframe
    time = [datetime.datetime.strptime(el[:-7], '%Y-%m-%dT%H:%M:%S.%f')
            for el in timestamp[cutoff_frame:filtered_traces.shape[0]+cutoff_frame]]
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

    return out_path, filtered_traces


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
        arena_corners, obstacle_positions, df_line = read_motive_header(file_path_motive)
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

