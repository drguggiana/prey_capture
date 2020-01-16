from tkinter import filedialog
from functions_misc import tk_killwindow, error_logger
import paths
import os
import datetime
import numpy as np

# get rid of the tk main window
tk_killwindow()

# define the save path
save_path = paths.motive_path
# define the base loading path
base_path_bonsai = paths.bonsai_path
# define the target motive path
target_path = paths.motive_path
# # select the files to process
# file_path = filedialog.askopenfilenames(initialdir=target_path, filetypes=(("text files", "*.txt"),))

# get a list of the bonsai file times

# allocate a list with the identity of each path
path_id = []
# also lists for the motive and bonsai paths
motive_names = []
bonsai_names = []
# for all the files in the target folder
for files in os.listdir(base_path_bonsai):
    # TODO: update the conditions depending on the types of file
    # detect the type of path
    if files[8] == 'T':
        path_id.append(0)
        motive_names.append(files)

    elif 'sync' in files:
        path_id.append(2)
    elif '.avi' in files:
        path_id.append(3)
    else:
        path_id.append(1)
        bonsai_names.append(files)

# calculate the bonsai times
bonsai_times = [datetime.datetime.strptime(el[:18], '%m_%d_%Y_%H_%M_%S') for el in bonsai_names]
# initialize a list to contain errors
error_log = []
# for all the motive files
for name in motive_names:
    # get the motive timestamp
    motive_time = datetime.datetime.strptime(name[:15], '%Y%m%dT%H%M%S')

    # get the delta times between the bonsai files and this motive file
    delta_withbonsai = np.abs([el - motive_time
                               for el in bonsai_times])
    bonsai_idx = np.argmin(delta_withbonsai)
    try:
        assert bonsai_idx is not np.ndarray, 'More than 1 matching file found'
        assert delta_withbonsai[bonsai_idx].seconds < 100, 'Delay too long, probably unmatched file'
    except AssertionError as e:
        error_logger(error_log, '_'.join((name, 'File matching problem', str(e.args))))
        # still rename the file to the bonsai time nomenclature
        # define the new name
        new_name = motive_time.strftime('%m_%d_%Y_%H_%M_%S') + name[15:]
        os.rename(os.path.join(base_path_bonsai, name),
                  os.path.join(base_path_bonsai, new_name))
        continue

    # if the match is legit, rename the file
    os.rename(os.path.join(base_path_bonsai, name),
              os.path.join(base_path_bonsai, bonsai_names[bonsai_idx].replace('.csv', '.txt')))
