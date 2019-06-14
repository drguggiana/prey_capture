# imports
from tkinter import filedialog
from tkinter import Tk
# import re
import matplotlib.pyplot as plt
import numpy as np
# from mpl_toolkits.mplot3d import Axes3D
# from mpl_toolkits.mplot3d.art3d import Line3DCollection
# from pykalman import KalmanFilter
from scipy.ndimage.measurements import label
# from scipy.signal import decimate
# from scipy.interpolate import interp1d
# from matplotlib import animation, rc
# from matplotlib.collections import LineCollection
# import matplotlib.colors as colors
# from IPython.display import HTML
# import math
import datetime
from os import path
import csv
# rc('animation', html='html5')


# Create Tk root
root = Tk()
# Hide the main window
root.withdraw()
root.call('wm', 'attributes', '.', '-topmost', True)

# # define loading path and select file
# base_path = r'C:\Users\drguggiana\Documents\Motive_test1\etc'
# file_path = filedialog.askopenfilenames(initialdir=base_path)
#
#
# # parse the file
# def parse_line(single_line):
#     parsed_line = [float(number) for number in single_line[:-1].split(',')]
#     # parsed_line = re.findall('[+-]?\d+[.]\d+',single_line)
#     # parsed_line = [float(s) for s in re.findall('[+-]?\d+.\d+',single_line)]
#     return parsed_line
#
#
# # allocate a list for all the animals
# animal_data = []
#
# for animal in file_path:
#     parsed_data = []
#     with open(file_path[0]) as f:
#         for line in f:
#             if line[0] == '0':
#                 continue
#             parsed_data.append(parse_line(line))
#
#     animal_data.append(np.array(parsed_data))

# define the save path
save_path = r'J:\Drago Guggiana Nilo\Prey_capture\Pre_processed'
# define the base loading path
base_path_bonsai = r'J:\Drago Guggiana Nilo\Prey_capture\Bonsai'
# select the files to process
file_path_bonsai = filedialog.askopenfilenames(initialdir=base_path_bonsai, filetypes=(("csv files", "*.csv"), ))
# define loading path and select file
# allocate a list for all the animals
animal_data_bonsai = []

for animal in file_path_bonsai:
    parsed_data = []
    last_nan = 0
    with open(animal) as f:
        for ex_line in f:
            ex_list = ex_line.split(' ')
            ex_list.remove('\n')
            if (float(ex_list[0]) < 110 or ex_list[0] == 'NaN') and last_nan == 0:
                continue
            else:
                last_nan = 1

            timestamp = ex_list.pop()
            ex_list = [float(el) for el in ex_list]

            parsed_data.append([ex_list, timestamp])

    animal_data_bonsai.append(np.array(parsed_data))

# interpolate between NaNs

# define the target columns
tar_columns = [[0, 1], [2, 3]]
# define the arena left boundary
left_bound_mouse = 110
left_bound_cricket = 100
# tar_columns = [[2, 3]]

# allocate memory for the preprocessed file
preproc_list = []
time_list = []
number_points = None

for files in animal_data_bonsai:

    # get the time
    time = [datetime.datetime.strptime(el[1][:-7], '%Y-%m-%dT%H:%M:%S.%f') for el in files]

    time = [(el - time[0]).total_seconds() for el in time]    # print the frame rate
    print('Frame rate:' + str(1 / np.mean(np.diff(time))) + 'fps')

    # get just the coordinate data
    files = np.vstack(np.array([el[0] for el in files]))
    # eliminate any pieces of the trace until the mouse track doesn't have NaNs
    if sum(np.isnan(files[:, 0])) > 0:
        nan_pointer = np.max(np.argwhere(np.isnan(files[:, 0])))
        files = files[nan_pointer + 1:, :]
        time = time[nan_pointer + 1:]
    # eliminate any remaining pieces captured outside the arena
    if sum(files[:, 0] < left_bound_mouse) > 0:
        nan_pointer = np.max(np.argwhere(files[:, 0] < left_bound_mouse))
        files = files[nan_pointer + 1:, :]
        time = time[nan_pointer + 1:]
    # same as above with the cricket
    if sum(files[:, 2] < left_bound_cricket) > 0:
        nan_pointer = np.max(np.argwhere(files[:, 2] < left_bound_cricket))
        files = files[nan_pointer + 1:, :]
        time = time[nan_pointer + 1:]
    # save the time vector
    time_list.append(time)
    # allocate memory for a per file list
    perfile_array = np.zeros_like(files)

    # for the mouse and the cricket columns
    for idx, animal in enumerate([tar_columns[1]]):
        # allocate a list to store the marked traces
        marked_traces = []
        col = animal[0]
        # get the data
        result = files[:, col].copy()

        # remove the NaN regions
        result[np.isnan(result) == 0] = 0
        result[np.isnan(result)] = 1

        # label the NaN positions
        result, number_regions = label(result)

        # run through the columns again
        for idx2, col in enumerate(animal):

            # copy the original vector
            target_vector = files[:number_points, col]
            # for all the regions
            for segment in range(2, number_regions + 1):
                # get the edges of the labeled region
                indexes = np.nonzero(result == segment)[0]
                start = indexes[0] - 1
                end = indexes[-1] + 1
                # skip the segment if the end of the segment is also the end of the whole trace
                if end == target_vector.shape:
                    continue

                target_vector[result == segment] = np.interp(indexes, [start, end], target_vector[[start, end]])
            perfile_array[:, idx * idx2 + idx2] = target_vector

            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(time, files[:number_points, col])
            #         ax.plot(result)
            ax.plot(time, target_vector, marker='*')
            plt.xlabel('Time (s)')
    preproc_list.append(files)

    # define the start of plotting
    plot_start = 0
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.gca().invert_yaxis()

    ax.plot(files[plot_start:number_points, tar_columns[0][0]],
            files[plot_start:number_points, tar_columns[0][1]], marker='*', linestyle='None')
    ax.plot(files[plot_start:number_points, tar_columns[1][0]],
            files[plot_start:number_points, tar_columns[1][1]], marker='*', linestyle='-')
    plt.close('all')

# now remove the large jumps
# define the maximum step
max_step_euc = 30

# define the target columns
tar_columns = [[0, 1], [2, 3]]
# tar_columns = [[2, 3]]

number_points = None

proc_list = []

for idx_file, files in enumerate(preproc_list):
    time = time_list[idx_file]
    # for the mouse and the cricket columns
    for animal in tar_columns:
        # allocate a list to store the marked traces
        marked_traces = []

        for idx, col in enumerate(animal):
            # get the data
            curr_data = files[:, col].copy()

            # take the absolute derivative trace
            result = np.absolute(np.diff(curr_data[:number_points]))

            result = np.hstack((0, result))

            # append the result to the list
            marked_traces.append(result)

        # combine the marked traces and rebinarize
        pre_result = []
        for idx, el in enumerate(marked_traces[0]):
            current_xy = np.array([el, marked_traces[1][idx]])
            pre_result.append(np.sqrt(current_xy[0] ** 2 + current_xy[1] ** 2))

        result = np.array(pre_result)
        # kill the remaining NaNs
        result[np.isnan(result)] = 0

        result[result < max_step_euc] = 0
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(result)

        # label the relevant regions
        result, number_regions = label(result)

        # run through the columns again
        for col in animal:

            # copy the original vector
            target_vector = files[:number_points, col]
            # for all the regions
            for segment in range(2, number_regions + 1):
                # get the edges of the labeled region
                indexes = np.nonzero(result == segment)[0]
                start = indexes[0] - 1
                end = indexes[-1] + 1
                # skip the segment if the end of the segment is also the end of the whole trace
                if end == target_vector.shape:
                    continue

                target_vector[result == segment] = np.interp(indexes, [start, end],
                                                             target_vector[[start, end]])

            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(time, files[:number_points, col])
            #         ax.plot(result)
            ax.plot(time, target_vector, marker='*')
            plt.xlabel('Time (s)')
    proc_list.append(files)
    # define the start of plotting
    plot_start = 0
    fig_final = plt.figure()
    ax = fig_final.add_subplot(111)
    plt.gca().invert_yaxis()

    ax.plot(files[plot_start:number_points, tar_columns[0][0]],
            files[plot_start:number_points, tar_columns[0][1]], marker='*', linestyle='None')
    ax.plot(files[plot_start:number_points, tar_columns[1][0]],
            files[plot_start:number_points, tar_columns[1][1]], marker='*', linestyle='-')
    fig_final.savefig(path.join(save_path, path.basename(file_path_bonsai[idx_file])[:-4]+'.png'), bbox_inches='tight')
# plt.show()
    plt.close('all')

# save the results

# for all the files
for idx, files in enumerate(proc_list):
    times = time_list[idx]
    # assemble the file name
    save_file = path.join(save_path, path.basename(file_path_bonsai[idx])[:-4] + '_preproc.csv')
    with open(save_file, mode='w', newline='') as f:
        file_writer = csv.writer(f, delimiter=',')
        for el, t in zip(files, times):
            file_writer.writerow(np.hstack((el, t)))
