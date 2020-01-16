# imports
from tkinter import filedialog
from tkinter import Tk
import matplotlib.pyplot as plt
import numpy as np
import paths
from functions_preprocessing import median_discontinuities, nan_large_jumps, interpolate_segments
from functions_io import parse_line, get_file_date
import os
import pandas as pd
import datetime
import statistics
# Create Tk root
root = Tk()
# Hide the main window
root.withdraw()
root.call('wm', 'attributes', '.', '-topmost', True)

# define loading path and select file
base_path = paths.motive_path
file_path = filedialog.askopenfilenames(initialdir=base_path)


# # parse the file
# def parse_line(single_line):
#     parsed_line = [float(number) for number in single_line[:-1].split(',')]
#     # parsed_line = re.findall('[+-]?\d+[.]\d+',single_line)
#     # parsed_line = [float(s) for s in re.findall('[+-]?\d+.\d+',single_line)]
#     return parsed_line


# allocate a list for all the animals
# animal_data = []

for animal in file_path:
    parsed_data = []
    with open(file_path[0]) as f:
        for idx, line in enumerate(f):
            if line[0] == '0':
                continue
            # elif idx == 30:
            #     break
            parsed_data.append(parse_line(line))

    target_data = np.array(parsed_data)

    # check if there are lists inside the array
    if isinstance(target_data[0], list):
        # if so, get their sizes
        list_sizes = np.array([len(el) for el in target_data])
        # get the mode of the sizes
        size_mode = statistics.mode(list_sizes)
        # get the time points with non-mode sizes
        non_mode_points = np.argwhere(list_sizes != size_mode)
        # allocate memory to nan/unpack
        unpack_data = []
        # nan the non-mode time points that are not the timestamp, unpack the others
        for idx, line in enumerate(target_data):
            # if it's a non-mode line
            if idx in non_mode_points:
                new_line = [target_data[idx][0]] + list(np.zeros(size_mode-1)*np.nan)
            else:
                new_line = target_data[idx]
            unpack_data.append(new_line)
        # turn the data into an array
        target_data = np.array(unpack_data)

    # based on the name of the file, parse it into a dataframe
    # get the name of the file
    file_name = os.path.basename(animal)
    # get the file date
    file_date = get_file_date(animal)
    # if social is present, parse the second mouse too
    if 'social' in file_name:
        col_names = ['time', 'mouse_position_x', 'mouse_position_z', 'mouse_position_y'
                     , 'mouse_rotation_x', 'mouse_rotation_z', 'mouse_rotation_y'
                     , 'mouse2_position_x', 'mouse2_position_z', 'mouse2_position_y'
                     , 'mouse2_rotation_x', 'mouse2_rotation_z', 'mouse2_rotation_y'
                     , 'color_factor']
        # define a factor to correctly calculate the frame rate based on the time stamping
        frame_rate_factor = 1
    # check if the file is before the sync files
    elif file_date <= datetime.datetime(year=2019, month=11, day=10):
        col_names = ['time', 'mouse_position_x', 'mouse_position_z', 'mouse_position_y'
                     , 'mouse_rotation_x', 'mouse_rotation_z', 'mouse_rotation_y'
                     , 'cricket_position_x', 'cricket_position_z', 'cricket_position_y']
        # define a factor to correctly calculate the frame rate based on the time stamping
        frame_rate_factor = 1e-7
    # otherwise, read as if a cricket is present
    else:
        col_names = ['time', 'mouse_position_x', 'mouse_position_z', 'mouse_position_y'
                     , 'mouse_rotation_x', 'mouse_rotation_z', 'mouse_rotation_y'
                     , 'cricket_position_x', 'cricket_position_z', 'cricket_position_y'
                     , 'color_factor']
        # define a factor to correctly calculate the frame rate based on the time stamping
        frame_rate_factor = 1

    # turn into dataframe
    target_data = pd.DataFrame(target_data, columns=col_names)

    # # define target file
    # target_file = 0
    #
    # # get the corresponding data
    # target_data = animal_data[target_file]

    print(len(target_data))
    timestamp = target_data['time']
    # print('Frame rate motive:' + str(1 / np.mean(np.diff(timestamp * 1e-7))) + ' fps')
    print('Frame rate motive:' + str(1 / np.mean(np.diff(timestamp * frame_rate_factor))) + ' fps')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.diff(timestamp))

    # median the discontinuities
    filtered_traces = median_discontinuities(target_data, ['mouse_position_x', 'mouse_position_y'], 21)
    # detect if two mice are present. If so plot also
    if 'social' in file_name:
        filtered_traces = median_discontinuities(target_data, ['mouse2_position_x', 'mouse2_position_y'], 21)

    # eliminate discontinuities before interpolating
    filtered_traces = nan_large_jumps(filtered_traces, ['mouse_position_x', 'mouse_position_y'], 0.05, 300)
    if 'social' in file_name:
        filtered_traces = nan_large_jumps(filtered_traces, ['mouse2_position_x', 'mouse2_position_y'], 0.05, 300)

    # interpolate the NaN stretches
    target_data = interpolate_segments(filtered_traces, np.nan)

    # 2D mouse movement
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(target_data.mouse_position_y, target_data.mouse_position_x)

    # detect if two mice are present. If so plot also
    if 'social' in file_name:
        ax.plot(target_data.mouse2_position_y, target_data.mouse2_position_x)

    plt.gca().invert_xaxis()
    # ax.plot(target_data[:number_points,7], target_data[:number_points,9])
    ax.autoscale()
    ax.axis('equal')

    # plot the individual axes
    fig2 = plt.figure()
    ax = fig2.add_subplot(411)
    ax.plot(target_data.mouse_position_y)
    ax = fig2.add_subplot(412)
    ax.plot(target_data.mouse_position_x)

    if target_data.shape[1] > 11:
        ax = fig2.add_subplot(413)
        ax.plot(target_data.mouse2_position_y)
        ax = fig2.add_subplot(414)
        ax.plot(target_data.mouse2_position_x)


plt.show()
