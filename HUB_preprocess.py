from tkinter import filedialog
import functions_misc
import functions_matching
import functions_plotting
import functions_io
import paths

import tracking_preprocessBonsai
import matplotlib.pyplot as plt
import kinematic_S1_calculations
import datetime
import os

# get rid of the tk main window
functions_misc.tk_killwindow()

# prompt the user for file selection
# define the base loading path
base_path_bonsai = paths.bonsai_path
# select the files to process
file_path_bonsai = filedialog.askopenfilenames(initialdir=base_path_bonsai, filetypes=(("csv files", "*.csv"), ))

# run through the files and select the appropriate analysis script
for files in file_path_bonsai:
    # get the file date
    file_date = functions_io.get_file_date(files)
    # get the miniscope flag
    miniscope_flag = 'miniscope' in files
    # get the nomini flag
    nomini_flag = 'nomini' in files
    # decide the analysis path based on the file name and date
    # if miniscope with the _nomini flag, run bonsai only
    if miniscope_flag and nomini_flag:
        # run the first stage of preprocessing
        out_path, filtered_traces = tracking_preprocessBonsai.run_preprocess([files], paths.pre_processed_path,
                                                                             ['cricket_x', 'cricket_y'])
        # TODO: add corner detection to calibrate the coordinate to real size
        # in the meantime, add a rough manual correction based on the size of the arena and the number of pixels

        # run the preprocessing kinematic calculations
        kinematics_data = kinematic_S1_calculations.kinematic_calculations(out_path, paths.kinematics_path)

    # if miniscope regular, run with the matching of miniscope frames
    if miniscope_flag and not nomini_flag:
        # run the first stage of preprocessing
        out_path, filtered_traces = tracking_preprocessBonsai.run_preprocess([files], paths.pre_processed_path,
                                                                             ['cricket_x', 'cricket_y'])

        # run the preprocessing kinematic calculations
        kinematics_data = kinematic_S1_calculations.kinematic_calculations(out_path, paths.kinematics_path)

        # get the calcium file path
        calcium_path = os.path.join(paths.miniscope_path, os.path.basename(files).replace('.csv', '_calcium_data.h5'))

        # find the sync file
        sync_path = os.path.join(paths.sync_path, os.path.basename(files).replace('miniscope', 'syncMini', 1))

        # get a dataframe with the calcium data matched to the bonsai data
        matched_calcium = functions_matching.match_calcium(calcium_path, sync_path, filtered_traces, kinematics_data)

        # assemble the save path
        save_file = os.path.join(paths.kinematics_path, os.path.basename(files)[:-4] + '_preprocMini.csv')
        # save the data frame
        matched_calcium.to_csv(save_file)

        # also plot the output in a matrix
        fig_final = functions_plotting.plot_image([functions_misc.normalize_matrix(matched_calcium.to_numpy().T,
                                                                                   axis=1)])
        # save the figure
        fig_final.savefig(os.path.join(paths.kinematics_figs, os.path.basename(files)[:-4] + '_preprocMini.png'),
                          bbox_inches='tight')

    if not miniscope_flag and file_date <= datetime.datetime(year=2019, month=11, day=10):
        # run the first stage of preprocessing
        out_path, filtered_traces = tracking_preprocessBonsai.run_preprocess([files], paths.pre_processed_path,
                                                                             ['cricket_x', 'cricket_y'])
        # TODO: add corner detection to calibrate the coordinate to real size
        # in the meantime, add a rough manual correction based on the size of the arena and the number of pixels

        # TODO: add the old motive-bonsai alignment as a function

        # run the preprocessing kinematic calculations
        kinematics_data = kinematic_S1_calculations.kinematic_calculations(out_path, paths.kinematics_path)
    # TODO: if no miniscope and after sync, run the new analysis

print('yay')
print('<3')
