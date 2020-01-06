# imports
from tkinter import filedialog
import h5py
from paths import *
from functions_plotting import *
from os.path import join, split
from functions_misc import *
from functions_kinematic import *
from scipy.stats import sem
import pandas as pd

# prevent the appearance of the tk main window
tk_killwindow()

# use it also for saving figures
save_path = kinematics_figs

# define the variable names
variable_names = {'mouse_heading': 'Heading angle', 'head_direction': 'Head direction', 'cricket_heading': 'Prey angle',
                  'delta_heading': 'Mouse to prey angle', 'delta_head_heading': 'Mouse head to prey angle',
                  'mouse_cricket_distance': 'Distance to prey', 'mouse_speed': 'Mouse speed',
                  'mouse_acceleration': 'Mouse acceleration', 'cricket_speed': 'Prey speed',
                  'cricket_acceleration': 'Prey acceleration', 'head_height': 'Mouse head height', 'time': 'Time'}

units = {'mouse_heading': ' [deg]', 'head_direction': ' [deg]', 'cricket_heading': ' [deg]',
         'delta_heading': ' [deg]', 'mouse_cricket_angle': ' [deg]',
         'mouse_cricket_distance': ' [m]', 'mouse_speed': ' [m/s]',
         'mouse_acceleration': ' [m/s2]', 'cricket_speed': ' [m/s]',
         'cricket_acceleration': ' [m/s2]', 'head_height': ' [cm]', 'time': ' [s]'}


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
    # with h5py.File(files, 'r') as loadfile:
    kinematic_parameters = pd.read_hdf(files, 'kinematic_parameters')
    timecourse_parameters = pd.read_hdf(files, 'timecourse_parameters')
    encounter_parameters = pd.read_hdf(files, 'encounter_parameters')
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
legend_dict = {'succ': 'Successful trial', 'fail': 'Failed trial', '': 'Normal lighting', 'dark': 'Darkness',
               'all': 'All trials', 'miniscope': 'Small arena'}
# color_dict = {'succ': [0, 0, 0, 0.5], 'fail': [0, 0, 0, 0.5]}
# linestyle_dict = {'': '-', 'dark': '--'}
color_dict = {'succ_': [0., 0., 1., 0.5], 'fail_': [1, 0., 0., 0.5], 'succ_dark': [0.0, 0.0, 0.0, 0.5],
              'fail_dark': [0., 1., 0., 0.5], 'all_': [1, 0., 1, 0.5], 'succ_miniscope': [0.5, 0., 0.5, 0.5]}


def figure_histograms(histogram_variables):
    # PLOT HISTOGRAMS
    font = {'family': 'arial',
            'weight': 'normal',
            'size': 20}
    matplotlib.rc('font', **font)
    # for all data sets
    for counter, data in enumerate(data_all):

        # get the current outcome and condition keywords
        out_keyword = '_' + data[0]
        cond_keyword = '_' + data[1]
        legend_list.append(legend_dict[data[0]] + ', ' + legend_dict[data[1]])
        color.append(color_dict[str(data[0]) + '_' + str(data[1])])
        # load the actual data
        kin_parameters = data[2]

        # plot the histograms
        # define the variables to plot histograms from
        # histogram_variables = np.arange(5, 11)
        # histogram_variables = ['mouse_cricket_distance', 'mouse_speed',
        #                        'mouse_acceleration', 'cricket_speed', 'cricket_acceleration', 'Mouse head height']
        # for all the variables
        for var_count, variables in enumerate(histogram_variables):
            if counter == 0:
                histogram_figs.append(
                    histogram([[kin_parameters.loc[:, variables]]], color=[color[counter]], bins=100))
            else:
                histogram([[kin_parameters.loc[:, variables]]], fig=histogram_figs[var_count], color=[color[counter]],
                          bins=100)
            curr_histogram = histogram_figs[var_count]
            # change to log y axis if plot is in the selected variables
            if variables in ['mouse_cricket_distance', 'mouse_speed',
                             'mouse_acceleration', 'cricket_speed', 'cricket_acceleration']:
                curr_histogram.axes[0].set_yscale('log')

            plt.xlabel(variable_names[variables] + units[variables])
            plt.ylabel('Density [a.u.]')
            # plt.text(0.8, 0.8 - counter*0.1, str(kinematic_parameters.shape[0]) + ' trials',
            #          transform=curr_histogram.axes[0].transAxes)

            if counter == len(data_all) - 1:
                if counter > 0:
                    histogram_figs[var_count].legend(legend_list, fontsize=12)
                curr_histogram.savefig(join(save_path, 'histogram_' + variable_names[variables] +
                                            '_' + out_keyword + '_' + cond_keyword + '.png'),
                                       bbox_inches='tight')
    return histogram_figs


def figure_polar(polar_variables):
    # PLOT POLAR PLOTS
    font = {'family': 'arial',
            'weight': 'normal',
            'size': 20}
    matplotlib.rc('font', **font)
    # for all data sets
    for counter, data in enumerate(data_all):
        # get the current outcome and condition keywords
        out_keyword = '_' + data[0]
        cond_keyword = '_' + data[1]

        # load the actual data
        kin_parameters = data[2]
        # plot polar graphs
        # # define the variables to plot histograms from
        # polar_variables = [3]
        # for all the variables
        for var_count, variables in enumerate(polar_variables):
            polar_coord = bin_angles(kin_parameters.loc[:, variables])
            polar_coord[:, 1] = normalize_matrix(polar_coord[:, 1])

            if counter == 0:
                polar_figs.append(plot_polar(polar_coord, color=color[counter]))
            else:
                plot_polar(polar_coord, fig=polar_figs[var_count], color=color[counter])
            curr_polar = polar_figs[var_count]
            plt.title(variable_names[variables] + units[variables], y=1.15)
            if counter == len(data_all) - 1:
                if counter > 0:
                    polar_figs[var_count].legend(legend_list, fontsize=12, bbox_to_anchor=(1.0, 0.15))
                curr_polar.savefig(join(save_path, 'polarPlot_' + variable_names[variables] +
                                        '_' + out_keyword + '_' + cond_keyword + '.png'), bbox_inches='tight')
    return polar_figs


def figure_timecourse(timecourse_angle_variables, timecourse_nonangle_variables):
    # PLOT TIMECOURSES
    font = {'family': 'arial',
            'weight': 'normal',
            'size': 20}
    matplotlib.rc('font', **font)
    # for all data sets
    for counter, data in enumerate(data_all):
        # get the current outcome and condition keywords
        out_keyword = '_' + data[0]
        cond_keyword = '_' + data[1]
        # plot across trial graphs
        # get the timecourse data
        time_parameters = data[3]
        # compute the averages and errors

        angled_average = pd.DataFrame(unwrap(time_parameters.loc[:, timecourse_angle_variables + ['frame']].groupby('frame').agg(
            lambda x: circmean_deg(x))), columns=timecourse_angle_variables)
        nonangled_average = time_parameters.loc[:, timecourse_nonangle_variables + ['frame']].groupby('frame').mean()
        trial_average = pd.concat((angled_average, nonangled_average), axis=1)

        angled_std = pd.DataFrame(unwrap(time_parameters.loc[:, timecourse_angle_variables + ['frame']].groupby('frame').agg(
            lambda x: circstd_deg(x)/np.sqrt(x.shape[0]))), columns=timecourse_angle_variables)
        nonangled_std = time_parameters.loc[:, timecourse_nonangle_variables + ['frame']].groupby('frame').sem()
        trial_sem = pd.concat((angled_std, nonangled_std), axis=1)

        # define the variables to plot from
        # timecourse_variables = np.arange(3, 11)
        timecourse_variables = timecourse_angle_variables + timecourse_nonangle_variables
        # for all the variables
        for var_count, variables in enumerate(timecourse_variables):
            if counter == 0:
                timecourse_figs.append(plot_2d([[trial_average.loc[:, variables]]], yerr=[[trial_sem.loc[:, variables]]],
                                               color=[color[counter]]))
            else:
                plot_2d([[trial_average.loc[:, variables]]], yerr=[[trial_sem.loc[:, variables]]],
                        fig=timecourse_figs[var_count],
                        color=[color[counter]])
            curr_timecourse = timecourse_figs[var_count]
            plt.ylabel(variable_names[variables] + units[variables])
            plt.xlabel('Normalized Time [a.u.]')
            if counter == len(data_all) - 1:
                if counter > 0:
                    timecourse_figs[var_count].legend(legend_list, fontsize=12)
                curr_timecourse.savefig(join(save_path, 'timecourse_' + variable_names[variables] +
                                             '_' + out_keyword + '_' + cond_keyword + '.png'),
                                        bbox_inches='tight')
    return timecourse_figs


def figure_encounter(encounter_angle_variables, encounter_nonangle_variables):
    # PLOT ENCOUNTERS
    font = {'family': 'arial',
            'weight': 'normal',
            'size': 20}
    matplotlib.rc('font', **font)
    # for all data sets
    for counter, data in enumerate(data_all):

        # plot encounter graphs
        # get the current outcome and condition keywords
        out_keyword = '_' + data[0]
        cond_keyword = '_' + data[1]
        # get the encounter data
        enc_parameters = data[4]
        # compute the averages and errors

        angled_average = pd.DataFrame(
            wrap(enc_parameters.loc[:, encounter_angle_variables + ['frame']].groupby('frame').agg(
                lambda x: 180 + circmean_deg(x))), columns=encounter_angle_variables)
        nonangled_average = enc_parameters.loc[:, encounter_nonangle_variables + ['frame']].groupby('frame').mean()
        encounter_average = pd.concat((angled_average, nonangled_average), axis=1)

        angled_std = pd.DataFrame(unwrap(enc_parameters.loc[:, encounter_angle_variables + ['frame']].groupby('frame').agg(
            lambda x: circstd_deg(x)/np.sqrt(x.shape[0]))), columns=encounter_angle_variables)
        nonangled_std = enc_parameters.loc[:, encounter_nonangle_variables + ['frame']].groupby('frame').sem()
        encounter_sem = pd.concat((angled_std, nonangled_std), axis=1)

        # plot the results
        # define the variables to plot from
        encounter_variables = encounter_angle_variables + encounter_nonangle_variables
        # get the time vector
        time_vector = enc_parameters.loc[(enc_parameters['encounter_id'] == 1) & (enc_parameters['trial_id'] == 0),
                                         'time_vector']

        # for all the variables
        for var_count, variables in enumerate(encounter_variables):
            if counter == 0:
                encounter_figs.append(plot_2d([[np.vstack((time_vector, encounter_average.loc[:, variables])).T]],
                                              yerr=[[encounter_sem.loc[:, variables]]],
                                              color=[color[counter]]))
            else:
                plot_2d([[np.vstack((time_vector, encounter_average.loc[:, variables])).T]],
                        yerr=[[encounter_sem.loc[:, variables]]],
                        fig=encounter_figs[var_count], color=[color[counter]])

            curr_encounter = encounter_figs[var_count]
            plt.figure(encounter_figs[var_count].number)
            plt.ylabel(variable_names[variables] + units[variables])
            plt.xlabel('Time [s]')
            # if it's the last data set of the ones selected
            if counter == len(data_all) - 1:
                # if there was more than one plot, plot the legend
                if counter > 0:
                    encounter_figs[var_count].legend(legend_list, fontsize=12)
                # set the target figure as the current figure
                plt.figure(encounter_figs[var_count].number)
                # turn autoscale off so the middle line spans the whole plot
                plt.autoscale(enable=False)
                # draw a line in the middle to indicate the encounter
                plt.plot([0, 0], [plt.ylim()[0], plt.ylim()[1]], color=[0., 1., 0., 1.])

                curr_encounter.savefig(join(save_path, 'encounter_' + variable_names[variables] +
                                            '_' + out_keyword + '_' + cond_keyword + '.png'),
                                       bbox_inches='tight')
    return encounter_figs


def figure_alltraces(target_parameter, sorting_parameter):
    # PLOT ENCOUNTERS AS AN IMAGE
    font = {'family': 'arial',
            'weight': 'normal',
            'size': 12}
    matplotlib.rc('font', **font)
    # initialize the list of figures
    trace_figs = []
    # for all data sets
    for counter, data in enumerate(data_all):
        # get the current outcome and condition keywords
        out_keyword = '_' + data[0]
        cond_keyword = '_' + data[1]
        # get the data
        enc_parameters = data[4]
        # for the target pairs of target and sorting parameters
        for tar, sort in zip(target_parameter, sorting_parameter):
            # sort the traces by a target parameter

            # get the sorting vector by obtaining the desired parameter based on group by and the encounter and trial_id
            sorting_vector = np.argsort(enc_parameters.loc[:, [sort] + ['encounter_id', 'trial_id']].groupby(
                                        ['trial_id', 'encounter_id']).max().to_numpy(), axis=0)
            # then extract the values as a single array to use for indexing
            sorting_vector = np.array([el[0] for el in sorting_vector])
            # get the actual encounters to be sorted in matrix form
            plot_parameters = enc_parameters.loc[:, [tar] + ['encounter_id', 'trial_id']].groupby(
                ['trial_id', 'encounter_id']).agg(list).to_numpy()
            # turn them into an array, sorted by the sorting parameter
            plot_parameters = np.array([el for sublist in plot_parameters for el in sublist])[sorting_vector]
            # plot the results
            trace_figs.append(plot_image([plot_parameters], colorbar=[variable_names[tar]]))

            trace_figs[counter].savefig(join(save_path, 'trials_param' + variable_names[tar] + '_sortedby'
                                             + variable_names[sort] +
                                             '_' + out_keyword + '_' + cond_keyword + '.png'), bbox_inches='tight')
    return trace_figs


def figure_alllines(target_parameter, sorting_parameter):
    # PLOT ENCOUNTERS AS TRACES
    font = {'family': 'arial',
            'weight': 'normal',
            'size': 12}
    matplotlib.rc('font', **font)
    # initialize the list of figures
    trace_figs = []
    # for all data sets
    for counter, data in enumerate(data_all):
        # get the current outcome and condition keywords
        out_keyword = '_' + data[0]
        cond_keyword = '_' + data[1]
        # get the data
        enc_parameters = data[4]
        # for the target pairs of target and sorting parameters
        for tar, sort in zip(target_parameter, sorting_parameter):

            # get the sorting vector by obtaining the desired parameter based on group by and the encounter and trial_id
            sorting_vector = np.argsort(enc_parameters.loc[:, [sort] + ['encounter_id', 'trial_id']].groupby(
                                        ['trial_id', 'encounter_id']).max().to_numpy(), axis=0)
            # then extract the values as a single array to use for indexing
            sorting_vector = np.array([el[0] for el in sorting_vector])
            # get the actual encounters to be sorted in matrix form
            plot_parameters = enc_parameters.loc[:, [tar] + ['encounter_id', 'trial_id']].groupby(
                ['trial_id', 'encounter_id']).agg(list).to_numpy()
            # turn them into an array, sorted by the sorting parameter
            plot_parameters = np.array([el for sublist in plot_parameters for el in sublist])[sorting_vector]
            # create the figure
            fig = plt.figure()
            trace_figs.append(fig)
            ax = fig.add_subplot(111)
            # plot them one by one
            for traces in plot_parameters:
                ax.plot(traces)

            trace_figs[counter].savefig(join(save_path, 'lines_param' + variable_names[tar] + '_sortedby'
                                             + variable_names[sort] +
                                             '_' + out_keyword + '_' + cond_keyword + '.png'), bbox_inches='tight')
    return trace_figs


# plt.close('all')
histogram_vars = ['mouse_cricket_distance', 'mouse_speed',
                  'mouse_acceleration', 'cricket_speed', 'cricket_acceleration']
histogram_figures = figure_histograms(histogram_vars)
plt.close('all')

# polar_vars = ['delta_heading']
# polar_figures = figure_polar(polar_vars)
# plt.close('all')
#
# timecourse_angle_vars = ['mouse_heading', 'cricket_heading', 'delta_heading']
# timecourse_nonangle_vars = ['mouse_cricket_distance', 'mouse_speed', 'mouse_acceleration', 'cricket_speed',
#                             'cricket_acceleration']
# timecourse_figures = figure_timecourse(timecourse_angle_vars, timecourse_nonangle_vars)
# plt.close('all')
#
# encounter_angle_vars = ['mouse_heading', 'cricket_heading', 'delta_heading']
# encounter_nonangle_vars = ['mouse_cricket_distance', 'mouse_speed', 'mouse_acceleration', 'cricket_speed',
#                            'cricket_acceleration']
# encounter_figures = figure_encounter(encounter_angle_vars, encounter_nonangle_vars)
# plt.close('all')


trace_figures = figure_alltraces(target_parameter=['mouse_speed'], sorting_parameter=['mouse_cricket_distance'])
plt.close('all')

# line_figures = figure_alllines(target_parameter=['mouse_speed'], sorting_parameter=['mouse_cricket_distance'])
# plt.close('all')

# plt.show()
