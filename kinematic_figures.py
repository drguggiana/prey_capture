# imports
from tkinter import filedialog
import h5py
from paths import *
from plotting_functions import *
from os.path import join, split
from misc_functions import *
from kinematics_functions import *
from scipy.stats import sem

# prevent the appearance of the tk main window
tk_killwindow()

# use it also for saving figures
save_path = kinematics_figs

# define the variable names
variable_names = ['Heading angle', 'Head direction', 'Prey angle', 'Prey to mouse angle',
                  'Prey to mouse head angle', 'Distance to prey', 'Mouse speed',
                  'Mouse acceleration', 'Prey speed', 'Prey acceleration', 'Mouse head height',
                  'Time']
units = [' [deg]', ' [deg]', ' [deg]', ' [deg]', ' [deg]', ' [m]', ' [cm/s]', ' [cm/s2]',
         ' [cm/s]', ' [cm/s2]', ' [cm]', ' [s]']

# select the files to load
base_path = kinematics_figs
file_path = filedialog.askopenfilenames(initialdir=base_path, filetypes=(("kine files", "*.hdf5"),))

# allocate memory for the data
data_all = []
# for all the selected files
for files in file_path:
    # extract the outcome and condition terms
    head, tail = split(files)
    outcome_term = tail[:-5].split('_')[1]
    condition_term = tail[:-5].split('_')[2]
    # load the processed data
    with h5py.File(files, 'r') as loadfile:
        kinematic_parameters = loadfile['kinematic_parameters'][:]
        timecourse_parameters = loadfile['timecourse_parameters'][:]
        encounter_parameters = loadfile['encounter_parameters'][:]
    # append the data to the list
    data_all.append([outcome_term, condition_term, kinematic_parameters, timecourse_parameters, encounter_parameters])

# allocate memory for the figure lists
histogram_figs = []
polar_figs = []
timecourse_figs = []
encounter_figs = []
# allocate memory to store the keywords
outcome_keyword = ''
condition_keyword = ''
legend_list = []
color = []
# define a dictionary to map the keywords to the legend terms
legend_dict = {'succ': 'Successful Hunt', 'fail': 'Failure to capture', '': 'Light conditions', 'dark': 'Darkness'}
# color_dict = {'succ': [0, 0, 0, 0.5], 'fail': [0, 0, 0, 0.5]}
# linestyle_dict = {'': '-', 'dark': '--'}
color_dict = {'succ_': [0., 0., 1., 0.5], 'fail_': [1, 0., 0., 0.5], 'succ_dark': [0.0, 0.0, 0.0, 0.5],
              'fail_dark': [0., 1., 0., 0.5]}

plt.close('all')
# for all data sets
for counter, data in enumerate(data_all):

    # get the current outcome and condition keywords
    outcome_keyword += '_' + data[0]
    condition_keyword += '_' + data[1]
    legend_list.append(legend_dict[data[0]] + '_' + legend_dict[data[1]])
    color.append(color_dict[str(data[0]) + '_' + str(data[1])])
    # load the actual data
    kinematic_parameters = data[2]

    # plot the histograms
    # define the variables to plot histograms from
    histogram_variables = np.arange(5, 11)
    # for all the variables
    for var_count, variables in enumerate(histogram_variables):
        if counter == 0:
            histogram_figs.append(histogram([[kinematic_parameters[:, variables]]], color=[color[counter]]))
        else:
            histogram([[kinematic_parameters[:, variables]]], fig=histogram_figs[var_count], color=[color[counter]])
        curr_histogram = histogram_figs[var_count]
        # change to log y axis if plot is in the selected variables
        if variables in [6, 7, 8, 9, 10]:
            curr_histogram.axes[0].set_yscale('log')

        plt.xlabel(variable_names[variables] + units[variables])

        if counter == len(data_all) - 1:
            histogram_figs[var_count].legend(legend_list)
            curr_histogram.savefig(join(save_path, 'histogram_' + variable_names[variables] +
                                        '_' + outcome_keyword + '_' + condition_keyword + '.png'), bbox_inches='tight')
plt.close('all')
# for all data sets
for counter, data in enumerate(data_all):

    # load the actual data
    kinematic_parameters = data[2]
    # plot polar graphs
    # define the variables to plot histograms from
    polar_variables = [3]
    # for all the variables
    for var_count, variables in enumerate(polar_variables):
        polar_coord = bin_angles(kinematic_parameters[:, variables])

        if counter == 0:
            polar_figs.append(plot_polar(polar_coord, color=color[counter]))
        else:
            plot_polar(polar_coord, fig=polar_figs[var_count], color=color[counter])
        curr_polar = polar_figs[var_count]
        plt.title(variable_names[variables] + units[variables])
        if counter == len(data_all) - 1:
            polar_figs[var_count].legend(legend_list)
            curr_polar.savefig(join(save_path, 'polarPlot_' + variable_names[variables] +
                                    '_' + outcome_keyword + '_' + condition_keyword + '.png'), bbox_inches='tight')
plt.close('all')
# for all data sets
for counter, data in enumerate(data_all):

    # plot across trial graphs
    # get the timecourse data
    timecourse_parameters = data[3]
    # compute the averages and errors
    trial_average = np.hstack((unwrap(circmean_deg(timecourse_parameters[:, :, :5], axis=0), axis=0),
                               np.nanmean(timecourse_parameters[:, :, 5:], axis=0)))
    trial_sem = np.hstack(
        (circstd_deg(timecourse_parameters[:, :, :5], axis=0) / np.sqrt(timecourse_parameters.shape[0]),
         sem(timecourse_parameters[:, :, 5:], axis=0, nan_policy='omit')))
    # define the variables to plot from
    timecourse_variables = np.arange(3, 11)
    # for all the variables
    for var_count, variables in enumerate(timecourse_variables):
        if counter == 0:
            timecourse_figs.append(plot_2d([[trial_average[:, variables]]], yerr=[[trial_sem[:, variables]]],
                                           color=[color[counter]]))
        else:
            plot_2d([[trial_average[:, variables]]], yerr=[[trial_sem[:, variables]]], fig=timecourse_figs[var_count],
                    color=[color[counter]])
        curr_timecourse = timecourse_figs[var_count]
        plt.ylabel(variable_names[variables] + units[variables])
        plt.xlabel('Normalized Time [a.u.]')
        if counter == len(data_all) - 1:
            timecourse_figs[var_count].legend(legend_list)
            curr_timecourse.savefig(join(save_path, 'timecourse_' + variable_names[variables] +
                                    '_' + outcome_keyword + '_' + condition_keyword + '.png'), bbox_inches='tight')
plt.close('all')
# for all data sets
for counter, data in enumerate(data_all):

    # plot encounter graphs
    # get the encounter data
    encounter_parameters = data[4]
    # compute the averages and errors
    encounter_average = np.hstack((unwrap(circmean_deg(encounter_parameters[:, :, :5], axis=0), axis=0),
                                   np.nanmean(encounter_parameters[:, :, 5:], axis=0)))
    encounter_sem = np.hstack((circstd_deg(encounter_parameters[:, :, :5], axis=0) / np.sqrt(encounter_parameters.shape[0]),
                               sem(encounter_parameters[:, :, 5:], axis=0, nan_policy='omit')))

    # plot the results
    # define the variables to plot from
    encounter_variables = np.arange(3, 11)
    # for all the variables
    for var_count, variables in enumerate(encounter_variables):
        if counter == 0:
            encounter_figs.append(plot_2d([[encounter_average[:, variables]]], yerr=[[encounter_sem[:, variables]]],
                                          color=[color[counter]]))
        else:
            plot_2d([[encounter_average[:, variables]]], yerr=[[encounter_sem[:, variables]]],
                    fig=encounter_figs[var_count], color=[color[counter]])
        curr_encounter = encounter_figs[var_count]
        plt.ylabel(variable_names[variables] + units[variables])
        plt.xlabel('Time [a.u.]')
        if counter == len(data_all) - 1:
            encounter_figs[var_count].legend(legend_list)
            curr_encounter.savefig(join(save_path, 'encounter_' + variable_names[variables] +
                                        '_' + outcome_keyword + '_' + condition_keyword + '.png'), bbox_inches='tight')
plt.close('all')
# plt.show()
