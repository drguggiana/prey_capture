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
import datetime


def run_preprocess(file_path_bonsai, save_file, tar_columns,
                   kernel_size=21, max_step=300, max_length=50):
    """Preprocess the bonsai file"""
    # initialize a list for the path output
    parsed_data = []
    last_nan = 0
    with open(file_path_bonsai) as f:
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
    # parse the path
    parsed_path = parse_path(file_path_bonsai)
    # animal_data_bonsai.append(np.array(parsed_data))
    files = np.array(parsed_data)

    # trim the trace
    files, time = trim_bounds(files)

    # assemble a data frame with the data
    data = pd.DataFrame(files, columns=['mouse_x', 'mouse_y', 'cricket_x', 'cricket_y'])

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

    # also the mouse and the date
    filtered_traces['mouse'] = parsed_path['animal']
    filtered_traces['datetime'] = parsed_path['datetime']

    # create the file name
    filtered_traces.to_hdf(save_file, key='full_traces', mode='w', format='table')

    # append to the outpath list
    out_path = save_file

    return out_path, filtered_traces


def parse_path(in_path):
    """Parse the input path into a dict"""
    path_parts = path.basename(in_path)[:-4].split('_')

    # check whether the rig is miniscope or social
    if path_parts[6] == 'miniscope':
        rig = 'miniscope'
        counter = 7
    elif path_parts[6] == 'social':
        rig = 'social'
        counter = 7
    else:
        rig = 'vr'
        counter = 6

    out_path = {'datetime': datetime.datetime.strptime('_'.join((path_parts[:6])), '%m_%d_%Y_%H_%M_%S'),
                'rig': rig,
                'animal': '_'.join((path_parts[counter:counter+3])),
                'result': path_parts[counter+3]}
    return out_path


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
        # define the kernel size for the median filter
        kernel_size=21,
        # define the target columns
        tar_columns=['cricket_x', 'cricket_y'],
        # define the maximum amount of an allowed jump in the trajectory per axis, in pixels
        max_step=300,
        # define the maximum length of a jump to be interpolated
        max_length=50
    )
