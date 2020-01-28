import subprocess
import paths
import functions_io
from functions_misc import tk_killwindow
from tkinter import filedialog
import pandas as pd
import h5py
import numpy as np


# get rid of the tk main window
tk_killwindow()

# prompt the user for file selection
# define the base loading path
base_path = paths.miniscope_path
# select the files to process
# TODO: automate file selection so it picks all files from a given date
file_path_bonsai = filedialog.askopenfilenames(initialdir=base_path, filetypes=(("tif files", "*.tif"), ))

# # combine the selected files into a single tif
out_path_tif, out_path_log = functions_io.combine_tif(file_path_bonsai)
# out_path_log = filedialog.askopenfilename(initialdir=base_path, filetypes=(("log files", "*_CAT.csv"), ))

# TODO: run min1PIPE on the newly created tif file from python
# min1pipe_process = subprocess.Popen([paths.minpipe_path])

# get the path for the ca file
calcium_path = out_path_log.replace('.csv', '_data_processed.mat')
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
    new_calcium_path = row['filename'].replace('.tif', '_calcium_data.h5')

    # save the data as an h5py
    with h5py.File(new_calcium_path) as file:
        file.create_dataset('calcium_data', data=current_calcium)

    # update the frame counter
    frame_counter += row['frame_number']
