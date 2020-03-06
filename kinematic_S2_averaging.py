# imports
from tkinter import filedialog
from os.path import join, basename
from functions_io import *
from functions_plotting import *
from functions_misc import *
from functions_kinematic import *
from paths import *
from scipy.ndimage.measurements import label
import pandas as pd

# prevent the appearance of the tk main window
tk_killwindow()

# define the outcome keyword to search for
outcome_keyword = 'succ'
# define the condition keyword to search for
condition_keyword = 'miniscope'

# load the data
base_path = kinematics_path
file_path = filedialog.askopenfilenames(initialdir=base_path, filetypes=(("kine files", "*.csv"),))

# define the figure save path
save_path = kinematics_figs

# parse the file names for the desired trait
file_path = file_parser(file_path, outcome_keyword, condition_keyword, mse_threshold=0.01)

# actually load the data
data_all = [pd.read_csv(el, index_col=0) for el in file_path]

# generate histograms of the distance, speed and acceleration
kinematic_parameters = pd.concat(data_all)

# wrap angles
kinematic_parameters.mouse_heading = wrap(kinematic_parameters.mouse_heading)
kinematic_parameters.cricket_heading = wrap(kinematic_parameters.cricket_heading)

# normalize trial time and bin it
# define the number of time bins
number_timebins = 30
# allocate memory for the binned time
binned_trials = []
# for all the trials
for idx, data in enumerate(data_all):
    # get the time vector
    time_vector = data.time_vector.to_numpy()
    # bin it evenly in 10 bins
    time_bins = np.digitize(time_vector, np.histogram(time_vector, bins=number_timebins)[1], right=True)
    # bin the data correspondingly
    # binned_trials.append(np.array([np.mean(data.iloc[time_bins == (el+1), :-1], axis=0)
    # for el in range(number_timebins)]))
    binned_trials.append(data.groupby(time_bins).mean())
    # add a label column with the number of the trial to concatenate the whole thing
    binned_trials[-1]['trial_id'] = idx
    # add a label to the frames for grouping
    binned_trials[-1]['frame'] = np.arange(number_timebins+1)
# binned_trials = np.array(binned_trials)
# concatenate into a dataframe
binned_trials = pd.concat(binned_trials)

# define the time window width, centered on the encounter (in seconds)
encounter_window = 5
# allocate memory for the animal encounters
encounter_pertrial = []

# for all the trials
for idx, data in enumerate(data_all):

    # identify the regions with encounters
    [encounter_idx, encounter_number] = label(data.mouse_cricket_distance.to_numpy() < 100)
    # for all the encounters
    for encounters in range(1, encounter_number):
        # get the first coordinate of the encounter and grab the surroundings
        encounter_hit = np.nonzero(encounter_idx == encounters)[0][0]
        # get the starting time point of the encounter
        time_start = data.loc[encounter_hit, 'time_vector']
        # get the number of positions for this encounter
        encounter_start = np.argmin(np.abs(data.loc[:, 'time_vector'].to_numpy() - (time_start-encounter_window/2)))
        encounter_end = np.argmin(np.abs(data.loc[:, 'time_vector'].to_numpy() - (time_start+encounter_window/2)))

        if encounter_end == data.shape[0]:
            continue
        # store the distance and the speed around the encounter
        # encounter_pertrial.append(np.array(data[encounter_indexes, :]))
        encounter_pertrial.append(data.iloc[encounter_start:encounter_end+1, :].copy())
        # correct the time axis
        encounter_pertrial[-1].loc[:, 'time_vector'] -= time_start
        # add the trial id
        encounter_pertrial[-1]['trial_id'] = idx
        # add the encounter id
        encounter_pertrial[-1]['encounter_id'] = encounters

# interpolate the traces to match the one with the most points
# determine the trace with the most points
size_array = [el.shape[0] for el in encounter_pertrial]
max_coord = np.argmax(size_array)

max_time = encounter_pertrial[max_coord].time_vector.to_numpy()
# allocate memory for the interpolated traces
encounter_interpolated = []
# for all the traces, if it doesn't have the same number of points as the max, interpolate to it
for index, traces in enumerate(encounter_pertrial):
    if index != max_coord:

        encounter_interpolated.append(traces.drop(['time_vector', 'trial_id', 'encounter_id'],
                                                  axis=1).apply(interp_trace, raw=False,
                                                                args=(traces.time_vector.to_numpy(), max_time)))
        # # add the time and id vectors back
        # encounter_interpolated[-1]['time_vector'] = max_time
        # encounter_interpolated[-1]['trial_id'] = traces.trial_id.iloc[0]
        # encounter_interpolated[-1]['encounter_id'] = traces.encounter_id.iloc[0]
        # # add the frame
        # encounter_interpolated[-1]['frame'] = np.arange(max_time.shape[0])

    else:
        encounter_interpolated.append(encounter_pertrial[index].drop(['time_vector', 'trial_id', 'encounter_id'], axis=1))

        # encounter_interpolated[-1]['time_vector'] = max_time
        # encounter_interpolated[-1]['trial_id'] = traces.trial_id.iloc[0]
        # encounter_interpolated[-1]['encounter_id'] = traces.encounter_id.iloc[0]
        # # add the frame
        # encounter_interpolated[-1]['frame'] = np.arange(max_time.shape[0])

    # add the time and id vectors back
    encounter_interpolated[-1]['time_vector'] = max_time
    encounter_interpolated[-1]['trial_id'] = traces.trial_id.iloc[0]
    encounter_interpolated[-1]['encounter_id'] = traces.encounter_id.iloc[0]
    # add the frame
    encounter_interpolated[-1]['frame'] = np.arange(max_time.shape[0])
# average and sem across encounters
encounter_matrix = pd.concat(encounter_interpolated)

# save the results
# create the file name
file_name = join(save_path, 'kinematicVariables_' + outcome_keyword + '_' + condition_keyword + '.hdf5')
kinematic_parameters.to_hdf(file_name, key='kinematic_parameters', mode='w', format='table')
binned_trials.to_hdf(file_name, key='timecourse_parameters', mode='a', format='table')
encounter_matrix.to_hdf(file_name, key='encounter_parameters', mode='a', format='table')
