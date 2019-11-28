# imports
from tkinter import filedialog
import matplotlib.pyplot as plt
from os import path
import csv
from paths import *
from functions_misc import tk_killwindow
from functions_preprocessing import trim_bounds, median_discontinuities, interpolate_segments, eliminate_singles, \
    nan_large_jumps
import numpy as np

# get rid of the tk main window
tk_killwindow()

# define whether to plot the intermediate steps
plot_flag = 0
# define the kernel size for the median filter
kernel_size = 21
# define the target columns (0 and 1 are mouse, 2 and 3 are cricket at the moment)
tar_columns = [[0, 1], [2, 3]]
# define the maximum amount of an allowed jump in the trajectory per axis, in pixels
max_step = 300
# define the maximum length of a jump to be interpolated
max_length = 50

# define the save path
save_path = pre_processed_path
# define the base loading path
base_path_bonsai = bonsai_path
# select the files to process
file_path_bonsai = filedialog.askopenfilenames(initialdir=base_path_bonsai, filetypes=(("csv files", "*.csv"), ))
# define loading path and select file
# allocate a list for all the animals
animal_data_bonsai = []

for current_path in file_path_bonsai:
    parsed_data = []
    last_nan = 0
    with open(current_path) as f:
        for ex_line in f:
            ex_list = ex_line.split(' ')
            ex_list.remove('\n')
            # TODO: check what happens with older data using the new line
            # if (float(ex_list[0]) < 110 or ex_list[0] == 'NaN') and last_nan == 0:
            if (ex_list[0] == 'NaN') and last_nan == 0:
                continue
            else:
                last_nan = 1

            timestamp = ex_list.pop()
            ex_list = [float(el) for el in ex_list]

            parsed_data.append([ex_list, timestamp])

    # animal_data_bonsai.append(np.array(parsed_data))
    files = np.array(parsed_data)

    # trim the trace
    files, time = trim_bounds(files)

    # if plotting is enabled
    if plot_flag > 0:
        # define the start of plotting
        plot_start = 0
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.gca().invert_yaxis()

        ax.plot(files[plot_start:, tar_columns[0][0]],
                files[plot_start:, tar_columns[0][1]], marker='*', linestyle='None')
        ax.plot(files[plot_start:, tar_columns[1][0]],
                files[plot_start:, tar_columns[1][1]], marker='*', linestyle='-')

    # now remove the discontinuities in the trace

    # median filter only the cricket trace
    filtered_traces = median_discontinuities(files, [[2, 3]], kernel_size)

    # eliminate isolated points
    filtered_traces = eliminate_singles(filtered_traces)

    # eliminate discontinuities before interpolating
    filtered_traces = nan_large_jumps(filtered_traces, tar_columns, max_step, max_length)

    # interpolate the NaN stretches
    filtered_traces = interpolate_segments(filtered_traces, np.nan)

    # if plotting is enabled
    if plot_flag > 0:
        # define the start of plotting
        plot_start = 0
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.gca().invert_yaxis()
        # plot the unfiltered trace
        ax.plot(files[plot_start:, tar_columns[0][0]],
                files[plot_start:, tar_columns[0][1]], marker='*', linestyle='None')
        ax.plot(files[plot_start:, tar_columns[1][0]],
                files[plot_start:, tar_columns[1][1]], marker='*', linestyle='-')

        # plot the filtered trace
        ax.plot(filtered_traces[plot_start:, tar_columns[0][0]],
                filtered_traces[plot_start:, tar_columns[0][1]], marker='o', linestyle='None')
        ax.plot(filtered_traces[plot_start:, tar_columns[1][0]],
                filtered_traces[plot_start:, tar_columns[1][1]], marker='o', linestyle='-')

        # plot the individual original and unfiltered x and y separately
        fig = plt.figure()
        ax = fig.add_subplot(411)
        ax.plot(files[plot_start:, tar_columns[0][0]])
        ax.plot(filtered_traces[plot_start:, tar_columns[0][0]], marker='.', linestyle='-')
        ax2 = fig.add_subplot(412)
        ax2.plot(files[plot_start:, tar_columns[0][1]])
        ax2.plot(filtered_traces[plot_start:, tar_columns[0][1]], marker='.', linestyle='-')
        ax = fig.add_subplot(413)
        ax.plot(files[plot_start:, tar_columns[1][0]])
        ax.plot(filtered_traces[plot_start:, tar_columns[1][0]], marker='.', linestyle='-')
        ax2 = fig.add_subplot(414)
        ax2.plot(files[plot_start:, tar_columns[1][1]])
        ax2.plot(filtered_traces[plot_start:, tar_columns[1][1]], marker='.', linestyle='-')

        plt.show()

    # save the filtered trace
    fig_final = plt.figure()
    ax = fig_final.add_subplot(111)
    plt.gca().invert_yaxis()

    # plot the filtered trace
    ax.plot(filtered_traces[:, tar_columns[0][0]],
            filtered_traces[:, tar_columns[0][1]], marker='o', linestyle='-')
    ax.plot(filtered_traces[:, tar_columns[1][0]],
            filtered_traces[:, tar_columns[1][1]], marker='o', linestyle='-')
    fig_final.savefig(path.join(save_path, path.basename(current_path)[:-4] + '.png'),
                      bbox_inches='tight')

    plt.close('all')
    # save the results
    # assemble the file name
    save_file = path.join(save_path, path.basename(current_path)[:-4] + '_preproc.csv')
    # write the file
    with open(save_file, mode='w', newline='') as f:
        file_writer = csv.writer(f, delimiter=',')
        for el, t in zip(files, time):
            file_writer.writerow(np.hstack((el, t)))
