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
import pandas as pd


def run_preprocess(file_path_bonsai, save_path_bonsai, tar_columns, plot_flag=0,
                   kernel_size=21, max_step=300, max_length=50):
    """Preprocess the bonsai file"""
    # initialize a list for the path output
    out_path = []

    # for all the files
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

        # assemble a data frame with the data
        data = pd.DataFrame(files, columns=['mouse_x', 'mouse_y', 'cricket_x', 'cricket_y'])

        # if plotting is enabled
        if plot_flag > 0:
            # define the start of plotting
            fig = plt.figure()
            ax = fig.add_subplot(111)
            plt.gca().invert_yaxis()

            ax.plot(data.mouse_x,
                    data.mouse_y, marker='*', linestyle='None')
            ax.plot(data.cricket_x,
                    data.cricket_y, marker='*', linestyle='-')

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
        filtered_traces['time'] = time

        # if plotting is enabled
        if plot_flag > 0:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            plt.gca().invert_yaxis()
            # plot the unfiltered trace
            ax.plot(data.mouse_x,
                    data.mouse_y, marker='*', linestyle='None')
            ax.plot(data.cricket_x,
                    data.cricket_y, marker='*', linestyle='-')

            # plot the filtered trace
            ax.plot(filtered_traces.mouse_x,
                    filtered_traces.mouse_y, marker='o', linestyle='None')
            ax.plot(filtered_traces.cricket_x,
                    filtered_traces.cricket_y, marker='o', linestyle='-')

            # plot the individual original and unfiltered x and y separately
            fig = plt.figure()
            ax = fig.add_subplot(411)
            ax.plot(data.mouse_x)
            ax.plot(filtered_traces.mouse_x, marker='.', linestyle='-')
            ax2 = fig.add_subplot(412)
            ax2.plot(data.mouse_y)
            ax2.plot(filtered_traces.mouse_y, marker='.', linestyle='-')
            ax = fig.add_subplot(413)
            ax.plot(data.cricket_x)
            ax.plot(filtered_traces.cricket_x, marker='.', linestyle='-')
            ax2 = fig.add_subplot(414)
            ax2.plot(data.cricket_y)
            ax2.plot(filtered_traces.cricket_y, marker='.', linestyle='-')

            plt.show()

        # save the filtered trace
        fig_final = plt.figure()
        ax = fig_final.add_subplot(111)
        plt.gca().invert_yaxis()

        # plot the filtered trace
        ax.plot(filtered_traces.mouse_x,
                filtered_traces.mouse_y, marker='o', linestyle='-')
        ax.plot(filtered_traces.cricket_x,
                filtered_traces.cricket_y, marker='o', linestyle='-')
        fig_final.savefig(path.join(save_path_bonsai, path.basename(current_path)[:-4] + '.png'),
                          bbox_inches='tight')

        plt.close('all')
        # save the results
        # assemble the file name
        save_file = path.join(save_path_bonsai, path.basename(current_path)[:-4] + '_preproc.csv')
        # write the file
        filtered_traces.to_csv(save_file)

        # append to the outpath list
        out_path.append(save_file)
        # with open(save_file, mode='w', newline='') as f:
        #     file_writer = csv.writer(f, delimiter=',')
        #     for el, t in zip(files, time):
        #         file_writer.writerow(np.hstack((el, t)))
    return out_path, filtered_traces


if __name__ == '__main__':
    # get rid of the tk main window
    tk_killwindow()

    # define the save path
    save_path = pre_processed_path
    # define the base loading path
    base_path_bonsai = bonsai_path
    # select the files to process
    file_path = filedialog.askopenfilenames(initialdir=base_path_bonsai, filetypes=(("csv files", "*.csv"),))

    # run the preprocessing
    run_preprocess(
        file_path,
        save_path,
        # define whether to plot the intermediate steps
        plot_flag=0,
        # define the kernel size for the median filter
        kernel_size=21,
        # define the target columns
        tar_columns=['cricket_x', 'cricket_y'],
        # define the maximum amount of an allowed jump in the trajectory per axis, in pixels
        max_step=300,
        # define the maximum length of a jump to be interpolated
        max_length=50
    )
