import subprocess as sp
import paths
import functions_io
from functions_misc import tk_killwindow
from tkinter import filedialog
import pandas as pd
import h5py
import numpy as np
import os
import functions_data_handling as fd


# get rid of the tk main window
tk_killwindow()

# prompt the user for file selection
# define the base loading path
base_path = paths.videoexperiment_path
# select the files to process
all_files = filedialog.askopenfilenames(initialdir=base_path, filetypes=(("tif files", "*.tif"), ))
all_files = [el for el in all_files if 'CAT' not in el]
# all_files = [el for el in os.listdir(base_path) if (el.endswith('.tif')) and ('CAT' not in el)]
# group the files by date and mouse
# parse the paths
parsed_paths = [fd.parse_experiment_name(el) for el in all_files]
# get the mice and dates
mouse_list = [el['mouse'] for el in parsed_paths]
date_list = [el['date'][:10] for el in parsed_paths]

# get the unique mice and dates
unique_mice = np.unique(mouse_list)
unique_days = np.unique(date_list)

# for all the dates
for day in unique_days:
    for mouse in unique_mice:

        # get the files corresponding to this day and mouse
        file_path_bonsai = [os.path.join(base_path, el) for el in all_files if (day in el) and (mouse in el)]
        # if it's empty, skip the iteration
        if len(file_path_bonsai) == 0:
            continue
        # combine the selected files into a single tif
        out_path_tif, out_path_log = functions_io.combine_tif(file_path_bonsai, paths.miniscope_path)
        # out_path_log = filedialog.askopenfilename(initialdir=base_path, filetypes=(("log files", "*_CAT.csv"), ))

        min1pipe_process = sp.Popen([r'D:\Code Repos\environments\matlab_env\Scripts\python.exe',
                                     r'D:\Code Repos\prey_capture\minpipe_runner.py',
                                     out_path_tif], stdout=sp.PIPE)

        stdout = min1pipe_process.communicate()[0]
        print(stdout.decode())

        # get the path for the ca file
        calcium_path = out_path_log.replace('.csv', '_data_processed.mat')
        # if there are no ROIs detected, skip the file and print the name
        try:
            # load the contents of the ca file
            with h5py.File(calcium_path) as f:
                calcium_data = np.array((f['sigfn'])).T

            # grab the processed files and split them based on the log file

            # read the log file
            files_list = pd.read_csv(out_path_log)
            # initialize a counter for the frames
            frame_counter = 0

            # for all the rows in the dataframe
            for index, row in files_list.iterrows():
                # get the frames from this file
                current_calcium = calcium_data[:, frame_counter:row['frame_number']+frame_counter]

                # assemble the save path for the file
                new_calcium_path = os.path.join(base_path,
                                                os.path.basename(
                                                    row['filename'].replace('.tif', '_calcium_data.h5')))

                # save the data as an h5py
                with h5py.File(new_calcium_path) as file:
                    file.create_dataset('calcium_data', data=current_calcium)

                # update the frame counter
                frame_counter += row['frame_number']
        except KeyError:
            print('This file did not contain any ROIs: ' + calcium_path)

