# TODO: set up file loading
# TODO: plot average heading towards cricket, make it a function of distance
# TODO: plot the time sequence of the acceleration, speed of mouse and cricket
# imports
from tkinter import filedialog
from os.path import join, basename
from io_functions import *
from plotting_functions import *
from misc_functions import *
from kinematics_functions import *
from paths import *
from scipy.stats import sem
from scipy.ndimage.measurements import label
import h5py


# prevent the appearance of the tk main window
tk_killwindow()

# define the outcome keyword to search for
outcome_keyword = 'all'
# define the condition keyword to search for
condition_keyword = 'dark'

# load the data
base_path = kinematics_path
file_path = filedialog.askopenfilenames(initialdir=base_path, filetypes=(("kine files", "*.csv"),))
# file_path = test_kinematics

# define the figure save path
save_path = kinematics_figs

# parse the file names for the desired trait
file_path = file_parser(file_path, outcome_keyword, condition_keyword, mse_threshold=0.01)

# actually load the data
data_all = load_preprocessed(file_path)

# generate histograms of the distance, speed and acceleration
kinematic_parameters = np.vstack(data_all)

# for all three parameters
fig = histogram([[kinematic_parameters[:, el]] for el in range(5, 11)], rows=2, columns=3)
fig.axes[1].set_yscale("log")
fig.axes[2].set_yscale("log")
fig.axes[3].set_yscale("log")
fig.axes[4].set_yscale("log")
fig.axes[5].set_yscale("log")
fig.savefig(join(save_path, 'histograms_' + outcome_keyword + '_' + condition_keyword + '.png'), bbox_inches='tight')


# wrap angles
kinematic_parameters[:, 3] = wrap(kinematic_parameters[:, 3])
kinematic_parameters[:, 4] = wrap(kinematic_parameters[:, 4])

# prepare the polar heading plot
polar_coord = bin_angles(kinematic_parameters[:, 3])
heading_fig = plot_polar(polar_coord)
heading_fig.savefig(join(save_path, 'heading_' + outcome_keyword + '_' + condition_keyword + '.png'),
                    bbox_inches='tight')


polar_coord = bin_angles(kinematic_parameters[:, 4])
plot_polar(polar_coord)

# polar_coord = bin_angles(kinematic_parameters[kinematic_parameters[:, 7] > 0.25, 3])
# plot_polar(polar_coord)
#
# polar_coord = bin_angles(kinematic_parameters[kinematic_parameters[:, 7] > 0.25, 4])
# plot_polar(polar_coord)

# normalize trial time and bin it
# define the number of time bins
number_timebins = 30
# allocate memory for the binned time
binned_trials = []
# for all the trials
for data in data_all:
    # get the time vector
    time_vector = data[:, -1]
    # bin it evenly in 10 bins
    time_bins = np.digitize(time_vector, np.histogram(time_vector, bins=number_timebins)[1], right=True)
    # bin the data correspondingly
    binned_trials.append(np.array([np.mean(data[time_bins == (el+1), :-1], axis=0) for el in range(number_timebins)]))

binned_trials = np.array(binned_trials)
# average across trials
# trial_average = np.nanmean(np.array(binned_trials), axis=0)
# trial_sem = sem(np.array(binned_trials), axis=0, nan_policy='omit')

trial_average = np.hstack((unwrap(circmean_deg(binned_trials[:, :, :5], axis=0), axis=0),
                          np.nanmean(binned_trials[:, :, 5:], axis=0)))
trial_sem = np.hstack((circstd_deg(binned_trials[:, :, :5], axis=0)/np.sqrt(binned_trials.shape[0]),
                      sem(binned_trials[:, :, 5:], axis=0, nan_policy='omit')))

timecourse_fig = plot_2d([[trial_average[:, el]] for el in range(3, 11)], rows=2, columns=4, yerr=[[trial_sem[:, el]]
                                                                                                for el in range(3, 11)])
timecourse_fig.savefig(join(save_path, 'timecourse_' + outcome_keyword + '_' + condition_keyword + '.png'),
                       bbox_inches='tight')

# calculate the encounter triggered averages
# isolate attacks by distance
# define the number of positions to capture per encounter (split evenly)
encounter_positions = 500
# allocate memory for the animal encounters
encounter_pertrial = []
# for all the trials
for data in data_all:

    # identify the regions with encounters
    [encounter_idx, encounter_number] = label(data[:, 5] < 0.02)
    # for all the encounters
    for encounters in range(1, encounter_number):
        # get the first coordinate of the encounter and grab the surroundings
        encounter_start = np.nonzero(encounter_idx == encounters)[0][0]
        # get the vector of actual indexes
        encounter_indexes = np.linspace(encounter_start-encounter_positions/2,
                                        encounter_start+encounter_positions/2 - 1,
                                        encounter_positions, dtype=int)
        if np.max(encounter_indexes) >= data.shape[0]:
            continue
        # store the distance and the speed around the encounter
        encounter_pertrial.append(np.array(data[encounter_indexes, :]))
    # map_permouse, binx, biny = density_map(trials, bin_number, extrema_list)
    # map_list.append(map_permouse)

# average and sem across encounters
encounter_matrix = np.array(encounter_pertrial)

encounter_average = np.hstack((unwrap(circmean_deg(encounter_matrix[:, :, :5], axis=0), axis=0),
                               np.nanmean(encounter_matrix[:, :, 5:], axis=0)))
encounter_sem = np.hstack((circstd_deg(encounter_matrix[:, :, :5], axis=0)/np.sqrt(encounter_matrix.shape[0]),
                          sem(encounter_matrix[:, :, 5:], axis=0, nan_policy='omit')))

# plot the results
encounter_fig = plot_2d([[encounter_average[:, el]] for el in range(3, 11)], rows=2, columns=4, yerr=[[encounter_sem[:, el]]
                                                                                   for el in range(3, 11)])
encounter_fig.savefig(join(save_path, 'encounters_' + outcome_keyword + '_' + condition_keyword + '.png'),
                      bbox_inches='tight')

with h5py.File(join(save_path, 'kinematicVariables_' + outcome_keyword + '_'
                               + condition_keyword + '.hdf5'), 'w') as savefile:
    savefile.create_dataset('kinematic_parameters', data=kinematic_parameters)
    savefile.create_dataset('timecourse_parameters', data=binned_trials)
    savefile.create_dataset('encounter_parameters', data=encounter_matrix)

plt.close(fig='all')
# plt.show()
