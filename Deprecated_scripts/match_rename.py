from tkinter import filedialog
from functions_misc import tk_killwindow, error_logger
import paths
import os
import datetime
import numpy as np

# get rid of the tk main window
tk_killwindow()

# define the base loading path
base_path = paths.vrexperiment_path

# # select the files to process
# file_path = filedialog.askopenfilenames(initialdir=target_path, filetypes=(("text files", "*.txt"),))

# get a list of the bonsai file times

# allocate a list with the identity of each path
path_id = []
# also lists for the motive and bonsai paths
motive_names = []
bonsai_names = []
# for all the files in the target folder
for files in os.listdir(base_path):
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
# initialize a counter for the files replaced
replace_counter = 1
# for all the motive files
for name in motive_names:
    try:
        # get the motive timestamp
        motive_time = datetime.datetime.strptime(name[:15], '%Y%m%dT%H%M%S')
    except ValueError as e:
        # if it's not a valid time stamp, log and skip the file
        error_logger(error_log, '_'.join((name, 'Invalid time stamp', str(e.args))))
        continue

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
        os.rename(os.path.join(base_path, name),
                  os.path.join(base_path, new_name))
        continue

    # if the match is legit, rename the file
    os.rename(os.path.join(base_path, name),
              os.path.join(base_path, bonsai_names[bonsai_idx].replace('.csv', '.txt')))

    # print the value of the counter
    print('%s files renamed' % (str(replace_counter)))
    # update the counter
    replace_counter += 1

# create the path for the error log
error_path = os.path.join(base_path, '_'.join((datetime.datetime.now().strftime('%m_%d_%Y_%H_%M_%S'), 'errorlog.txt')))
# save the error_log
with open(error_path, 'w') as f:
    f.writelines(error_log)
