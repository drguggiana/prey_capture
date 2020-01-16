from tkinter import filedialog
from functions_misc import tk_killwindow
import paths
import functions_io
import tracking_preprocessBonsai
from PIL import Image
import pandas as pd
import numpy as np
import functions_plotting
import functions_misc
import h5py
import matplotlib.pyplot as plt
import kinematic_S1_calculations
import datetime

# get rid of the tk main window
tk_killwindow()

# prompt the user for file selection
# define the base loading path
base_path_bonsai = paths.bonsai_path
# select the files to process
file_path_bonsai = filedialog.askopenfilenames(initialdir=base_path_bonsai, filetypes=(("csv files", "*.csv"), ))
# file_path_bonsai = [r'D:\Code Repos\miniscope_processing\12_07_2019_15_01_31_miniscope_MM_191108_a_succ.csv']

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

        # kinematics_path = r'J:\Drago Guggiana Nilo\Prey_capture\Kinematics\12_07_2019_15_01_31_miniscope_MM_191108_a_succ_kinematics.csv'

        # # load the kinematics file
        # kinematics_data = pd.read_csv(kinematics_path, index_col=0)

        # run the preprocessing kinematic calculations
        kinematics_data = kinematic_S1_calculations.kinematic_calculations(out_path, paths.kinematics_path)

        # import the calcium data
        calcium_path = r'D:\Code Repos\miniscope_processing\07122019_MM_191108_a_000_data_processed.mat'
        with h5py.File(calcium_path) as f:

            calcium_data = np.array((f['sigfn'])).T

        # get the time vector from bonsai
        bonsai_time = filtered_traces.time
        # get the number of frames from the bonsai file
        n_frames_bonsai_file = filtered_traces.shape[0]

        # # find the miniscope min1PIPE output
        # mini_path = r'D:\Code Repos\miniscope_processing\07122019_MM_191108_a_000.tif'
        # # get the number of frames
        # with Image.open(mini_path) as img:
        #     # meta_dict = {TAGS[key]: img.tag[key] for key in img.tag.iterkeys()}
        #     n_frames_mini_tif = img.n_frames

        # find the sync file
        sync_path = r'D:\Code Repos\miniscope_processing\12_07_2019_15_01_31_syncMini_MM_191108_a_succ.csv'
        # load the sync data
        sync_data = pd.read_csv(sync_path, names=['Time', 'mini_frames', 'bonsai_frames'])
        # get the number of miniscope frames on the sync file
        n_frames_mini_sync = np.sum(np.diff(sync_data.mini_frames) > 0)
        # match the sync frames with the actual miniscope frames
        frame_times_mini_sync = sync_data.loc[np.concatenate(([0], np.diff(sync_data.mini_frames) > 0)) > 0, 'Time'].to_numpy()

        # find the gaps between bonsai frames, take the frames only after the background subtraction gap
        bonsai_ifi = np.argwhere(np.diff(sync_data.bonsai_frames) > 0)
        bonsai_start_frame = bonsai_ifi[np.argwhere(np.diff(bonsai_ifi, axis=0) > 1000)[0][0] + 1][0]
        n_frames_bonsai_sync = np.sum(np.diff(sync_data.bonsai_frames.to_numpy()[bonsai_start_frame:]) > 0)
        frame_times_bonsai_sync = sync_data.loc[np.concatenate(([0], np.diff(sync_data.bonsai_frames) > 0)) > 0, 'Time'].to_numpy()[-n_frames_bonsai_file:]
        # functions_plotting.plot_2d([[sync_data.bonsai_frames]], rows=2, columns=1)

        # interpolate the bonsai traces to match the mini frames
        matched_bonsai = kinematics_data.drop(['time_vector'], axis=1).apply(functions_misc.interp_trace, raw=False,
                                                                      args=(frame_times_bonsai_sync,
                                                                            frame_times_mini_sync))

        # trim the data to the frames within the experiment
        calcium_data = calcium_data[:, :n_frames_mini_sync]
        # print a single dataframe with the calcium matched positions and timestamps
        calcium_data_data = pd.DataFrame()

        # fig = functions_plotting.plot_2d([[functions_plotting.normalize_matrix(matched_bonsai.mouse_cricket_distance.to_numpy())]])
        # plt.plot(functions_plotting.normalize_matrix(np.mean(calcium_data, axis=0)))
        # plt.show()
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
