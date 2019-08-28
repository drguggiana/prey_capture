import numpy as np
import csv
from os import listdir
from os.path import isfile, join, basename
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.ndimage.measurements import label
from scipy.stats import sem
from sklearn.preprocessing import minmax_scale
import matplotlib.colors as colors


# define the outcome keyword to search for
outcome_keyword = 'succ'
# define the condition keyword to search for
condition_keyword = ''
condition_list = ['dark', 'vr']
# load the data
base_path = r'J:\Drago Guggiana Nilo\Prey_capture\Pre_processed'
# base_path = r'E:\Prey_capture\Pre_processed'
file_path = [join(base_path, f) for f in listdir(base_path) if isfile(join(base_path, f[:-4]+'.csv'))]
# file_path = [r'E:\Prey_capture\Pre_processed\05_24_2019_16_34_35_DG_190417_c_succ_preproc.csv']
# define the figure save path
figure_save = r'C:\Users\drguggiana\Dropbox\Bonhoeffer_things\Presentations\Figures'
# figure_save = r'C:\Users\Drago\Dropbox\Bonhoeffer_things\Presentations\Figures'


def density_map(occ_array, bin_num, bin_ranges):
    # occ_array = target_data[:, [1, 3]]

    # discretize the occupancy matrix
    # # define the number of bins
    # bin_num_x = 20
    # # calculate the bin number in y as a function of the relationship between the arena dimensions
    # bin_num_y = int(bin_num_x * 1.5)
    # generate the bin ranges for both dimensions
    # bins_x = np.linspace(np.min(occ_array[:, 0]), np.max(occ_array[:, 0]), bin_num_x)
    # bins_y = np.linspace(np.min(occ_array[:, 1]), np.max(occ_array[:, 1]), bin_num_y)
    bins_x = np.linspace(bin_ranges[0], bin_ranges[1], bin_num)
    bins_y = np.linspace(bin_ranges[2], bin_ranges[3], bin_num)
    # digitize the dimensions and generate a binned matrix
    occ_matrix = np.vstack((np.digitize(occ_array[:, 0], bins_x), np.digitize(occ_array[:, 1], bins_y))).T
    # allocate memory for the plotting matrix
    # occ_plot = np.zeros((bin_num_x, bin_num_y))
    occ_plot = np.zeros((bin_num, bin_num))
    # generate the density map
    for x, y in occ_matrix:
        occ_plot[x-1, y-1] += 1
    return occ_plot.T, bins_x, bins_y
    # # plot
    # fig = plt.figure()
    # plt.imshow(np.flip(occ_plot.T, axis=0), cmap='Reds', interpolation='nearest', norm=colors.PowerNorm(gamma=0.5))


def load_preprocessed(file_path_in):
    # allocate a list for all the animals
    animal_data = []

    for animal_in in file_path_in:
        parsed_data = []
        with open(animal_in) as f:
            reader = csv.reader(f, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
            for ex_line in reader:
                parsed_data.append(np.array(ex_line))

            animal_data.append(np.array(parsed_data))

    return animal_data


# filter the results by outcome (only use all for performance plot though)
if outcome_keyword != 'all':
    file_path = [file for file in file_path if outcome_keyword in file]

# filter the files by the desired condition
if condition_keyword == '':
    file_path = [file for file in file_path if sum([1 for word in condition_list if word in file]) == 0]
elif condition_keyword != 'all':
    file_path = [file for file in file_path if condition_keyword in file]

# extract the different animals used
animal_names = np.array(['_'.join(basename(el).split(sep='_')[6:9]) for el in file_path])
date_order = [0, 2, 1]
animal_dates = [basename(el).split(sep='_')[0:3] for el in file_path]
animal_dates = np.array(['_'.join([str(el[el2]) for el2 in [2, 0, 1]]) for el in animal_dates])

unique_animals, idx_animals, inv_animals = np.unique(animal_names, return_index=True, return_inverse=True)
unique_dates, idx_dates, inv_dates = np.unique(animal_dates, return_index=True, return_inverse=True)

# count the successes per animal and per date
# allocate memory for the result
performance_list = []
# for all the animals
for animal in unique_animals:
    # allocate a list for the dates
    performance_perdate = []
    for dates in unique_dates:
        # format the date
        date = '_'.join([str(dates.split(sep='_')[idx]) for idx in [1, 2, 0]])
        # get the total number of trials for that day
        date_trials = [trial for trial in file_path if ((date in trial) and (animal in trial))]
        # if it's empty, log a NaN and continue
        if not date_trials:
            performance_perdate.append([dates, np.nan])
            continue
        total_trials = len(date_trials)
        performance_perdate.append([dates, sum([1 if 'succ' in trial else 0 for trial in date_trials])/total_trials])
    # turn the whole list into an array and append the result to the animal list
    performance_list.append([animal, np.array(performance_perdate)])

# plot the performance per animal
fig = plt.figure()
ax = fig.add_subplot(111)
# initialize an accumulator
date_label = []
# for all the animals
for animal in performance_list:
    performance_vector = animal[1][:, 1].astype(float)
    # find the last not nan element and skip the remaining ones
    nonnan_idx = np.nonzero(~np.isnan(performance_vector))[0][-1]
    performance_vector = performance_vector[:nonnan_idx]
    date_vector = [datetime.strptime(el, '%Y_%m_%d') for el in animal[1][:nonnan_idx, 0]]
    date_vector = [(el - date_vector[0]).days for el in date_vector]
    ax.plot(date_vector, performance_vector, label=animal[0], marker='o')
    # ax.legend()
    if len(date_label) < len(date_vector):
        date_label = date_vector

plt.xticks(date_label, date_label, rotation=45)
plt.ylabel('Performance [a.u.]')
plt.xlabel('Days')
fig.savefig(join(figure_save, 'performanceMouse_'+outcome_keyword+'_'+condition_keyword+'.png'), bbox_inches='tight')

# calculate latency to attack and time to capture
# define the length constant (measured approximately from the data)
m_px = 1 / 445
# define the distance to be considered as an encounter and convert to pixels
encounter_distance = 0.02/m_px
# allocate memory for the result
latency_list = []
# for all the animals
for animal in unique_animals:
    # allocate a list for the dates
    latency_perdate = []
    for dates in unique_dates:
        # format the date
        date = '_'.join([str(dates.split(sep='_')[idx]) for idx in [1, 2, 0]])
        # get the total number of trials for that day
        date_trials = [trial for trial in file_path if ((date in trial) and (animal in trial))]
        # load the trial data
        trial_data = load_preprocessed(date_trials)
        # allocate memory for the trial info per date
        trial_perdate = np.zeros([len(trial_data), 3])
        # for all the trials
        for idx, trial in enumerate(trial_data):
            # if it's empty, log a NaN and continue
            if not list(trial):
                trial_perdate[idx, :] = np.array([np.nan, np.nan, np.nan])
                continue
            # calculate the distance vector
            distance_vector = np.array([np.linalg.norm(el[[0, 1]] - el[[2, 3]]) for el in trial])
            # get the time vector
            time_vector = trial[:, 4]

            # find all the encounters
            [encounter_idx, encounter_number] = label(distance_vector < encounter_distance)
            if sum(encounter_idx) == 0:
                # find the time of the beginning of the first one
                delay_to_initiation = np.nan
                # find the time of the end of the last one
                capture_time = np.nan
            else:
                # find the time of the beginning of the first one
                delay_to_initiation = time_vector[np.nonzero(encounter_idx == 1)[0][0]] - time_vector[0]
                # find the time of the end of the last one
                capture_time = time_vector[np.nonzero(encounter_idx == encounter_number)[0][0]] - time_vector[0]
            # save the results
            # trial_perdate.append([encounter_number, delay_to_initiation, capture_time])
            trial_perdate[idx, :] = np.array([encounter_number, delay_to_initiation, capture_time])
        # save the results per date
        latency_perdate.append([dates, trial_perdate])
    # save the data per animal
    latency_list.append([animal, latency_perdate])

# plot the capture time related numbers
fig = plt.figure()
encNumber_plot = fig.add_subplot(131)
initTime_plot = fig.add_subplot(132)
endTime_plot = fig.add_subplot(133)

# put the plot handles in a list
plot_array = [encNumber_plot, initTime_plot, endTime_plot]
plot_labels = ['Encounter number', 'Latency to first encounter [s]', 'Total hunt time [s]']
# initialize an accumulator
date_label = []
# for all the plot types
for idx_plot, plots in enumerate(plot_array):
    # for all the animals
    for animal in latency_list:
        if not animal[1]:
            continue
        # allocate memory for a date vector
        date_vector = []
        # also for the plot vector
        plot_vector = []
        # for all the dates in the list
        for dates in animal[1]:
            # if empty, skip
            if dates[1].shape[0] == 0:
                continue
            # save the date
            date_vector.append(dates[0])
            # get the data
            latency_matrix = dates[1].astype(float)
            # # for all the plot types
            # for idx_plot, plots in enumerate(plot_array):
            # get the slice to use from the matrix
            plot_values = latency_matrix[:, idx_plot]
            # find the last not nan element and skip the remaining ones
            # nonnan_idx = np.nonzero(~np.isnan(plot_values))[0][-1]
            plot_vector.append(np.nanmean(plot_values))

        date_vector = [datetime.strptime(el, '%Y_%m_%d') for el in date_vector]
        date_vector = [(el - date_vector[0]).days for el in date_vector]
        if len(date_label) < len(date_vector):
            date_label = date_vector
        # ax.plot(date_vector, latency_vector, label=animal[0], marker='o')
        plots.plot(date_vector, plot_vector, label=animal[0], marker='o')
        # plots.set_xticks(date_label[:2:])
        # plots.set_xticklabels(date_label[:2:], rotation=45, fontsize=8)
        plots.set_ylabel(plot_labels[idx_plot])
        plots.set_xlabel('Days')
        # plots.set_xlabel(date_label, rotation=45)

plt.subplots_adjust(wspace=0.8, hspace=0.3)
fig.savefig(join(figure_save, 'latencies_'+outcome_keyword+'_'+condition_keyword+'.png'), bbox_inches='tight')

# plot a fitted 2D map of the distance to prey vs acceleration
# get the limits of the distribution for distance and mouse speed

# density map
# allocate memory for the result
speed_distance_list = []
# allocate memory to save the extrema
extrema_list = np.zeros(4)
# for all the animals
for animal in unique_animals:
    # allocate a list for the dates
    speed_distance_perdate = []
    for dates in unique_dates:
        # format the date
        date = '_'.join([str(dates.split(sep='_')[idx]) for idx in [1, 2, 0]])
        # get the total number of trials for that day
        date_trials = [trial for trial in file_path if ((date in trial) and (animal in trial))]
        # load the trial data
        trial_data = load_preprocessed(date_trials)
        # allocate memory for the trial info per date
        # trial_perdate = np.zeros([len(trial_data), 3])
        trial_perdate = []
        # for all the trials
        for idx, trial in enumerate(trial_data):
            # if it's empty, log a NaN and continue
            if not list(trial):
                # trial_perdate[idx, :] = np.array([np.nan, np.nan, np.nan])
                continue
            webcam_perFrame = np.diff(trial[:, 4])[1:]
            # calculate the distance vector
            distance_vector = np.array([np.linalg.norm(el[[0, 1]] - el[[2, 3]]) for el in trial])[2:] * m_px
            distance_vector[distance_vector > 1.96] = np.nan
            # calculate the mouse speed
            mouse_speed_vector = np.linalg.norm(np.diff(trial[:, [0, 1]], axis=0, n=2), axis=1) * m_px / webcam_perFrame
            mouse_speed_vector[mouse_speed_vector > 1.42] = np.nan
            # store the two
            trial_perdate.append(np.vstack((distance_vector, mouse_speed_vector)).T)
            # get the extrema
            extrema_current = [np.nanmin(distance_vector), np.nanmax(distance_vector),
                              np.nanmin(mouse_speed_vector), np.nanmax(mouse_speed_vector)]
            # compare to limits, if higher, use
            # extrema_list = np.array([el1 if np.abs(el1) > np.abs(el2) else el2 for el1, el2
            #                          in zip(extrema_current, extrema_list)])
            extrema_list[0] = np.nanmin([extrema_list[0], extrema_current[0]])
            extrema_list[1] = np.nanmax([extrema_list[1], extrema_current[1]])
            extrema_list[2] = np.nanmin([extrema_list[2], extrema_current[2]])
            extrema_list[3] = np.nanmax([extrema_list[3], extrema_current[3]])

        # save the results per date
        speed_distance_perdate.append([dates, trial_perdate])
    # save the data per animal
    speed_distance_list.append([animal, speed_distance_perdate])

# allocate memory for the maps
map_list = []
# define the bin number
bin_number = 20
# for all the animals
for animal in speed_distance_list:
    if not animal[1]:
        continue
    # for all the dates in the list
    for dates in animal[1]:
        # if empty, skip
        if len(dates[1]) == 0:
            continue
        # for all the trials
        for trials in dates[1]:
            # exclude the NaNs
            # trials = trials[np.isnan(trials[:, 0]) == 0, :]
            map_permouse, binx, biny = density_map(trials, bin_number, extrema_list)
            map_list.append(map_permouse)

# average and plot
map_average = np.log(np.nanmean(np.stack(map_list, axis=2), axis=2))

fig = plt.figure()
# plt.imshow(map_average, cmap='Reds', interpolation='nearest', norm=colors.PowerNorm(gamma=0.5))
plt.imshow(map_average, cmap='Reds', interpolation='nearest')
plt.gca().invert_yaxis()
# # calculate the bin centers
# xcenter = (binx[:-1]+binx[0])/2
# ycenter = (biny[:-1]+biny[0])/2

# write the x and y labels
plt.xticks(range(bin_number), np.round(binx, 2), rotation=45)
plt.yticks(range(bin_number), np.round(biny, 2))

plt.xlabel('Distance to prey [m]')
plt.ylabel('Mouse acceleration [m/s2]')
plt.tight_layout()
fig.savefig(join(figure_save, 'density_'+outcome_keyword+'_'+condition_keyword+'.png'), bbox_inches='tight')


# isolate attacks by distance
# define the number of positions to capture per encounter (split evenly)
encounter_positions = 500
# allocate memory for the encounters
encounter_matrix = []
# for all the animals
for animal in speed_distance_list:
    if not animal[1]:
        continue
    # allocate memory for the animal encounters
    encounter_perdate = []
    # for all the dates in the list
    for dates in animal[1]:
        # if empty, skip
        if len(dates[1]) == 0:
            continue
        # allocate memory for the encounters
        encounter_pertrial = []
        # for all the trials
        for trials in dates[1]:
            # identify the regions with encounters
            [encounter_idx, encounter_number] = label(trials[:, 0] < 0.02)
            # for all the encounters
            for encounters in range(1, encounter_number):
                # get the first coordinate of the encounter and grab the surroundings
                encounter_start = np.nonzero(encounter_idx == encounters)[0][0]
                # get the vector of actual indexes
                encounter_indexes = np.linspace(encounter_start-encounter_positions/2,
                                                encounter_start+encounter_positions/2 - 1,
                                                encounter_positions, dtype=int)
                if np.max(encounter_indexes) >= trials.shape[0]:
                    continue
                # store the distance and the speed around the encounter
                encounter_pertrial.append(np.array([trials[encounter_indexes, :]]))
            # map_permouse, binx, biny = density_map(trials, bin_number, extrema_list)
            # map_list.append(map_permouse)
        if not encounter_pertrial:
            continue
        encounter_perdate.append(np.vstack(encounter_pertrial))
    encounter_matrix.append(encounter_perdate)

# plot the averages

fig = plt.figure()
distance_plot = fig.add_subplot(211)
speed_plot = fig.add_subplot(212)
plot_list = [distance_plot, speed_plot]
title_list = ['Distance to prey [m]', 'Mouse acceleration [m/s2]']
# for all the plots
for idx, plots in enumerate(plot_list):
    # allocate memory to store the trajectories for averaging
    trajectory_list = []
    # for all the animals
    for animal in encounter_matrix:
        # for all the dates
        for date in animal:

            # for both plot types
            # for idx, plots in enumerate(plot_list):
            # for all the encounters
            # for encounter in range(date.shape[0]):
            #     plots.plot(date[encounter, :, idx])
            # plots.plot(np.nanmean(minmax_scale(date[:, :, idx]), axis=0))
            # trajectory_list.append(np.nanmean(minmax_scale(date[:, :, idx]), axis=0))
            trajectory_list.append(np.nanmean(date[:, :, idx], axis=0))
    trajectory_list = np.array(trajectory_list)
    mean_trace = np.nanmean(trajectory_list, axis=0)
    x_trace = range(mean_trace.shape[0])
    plots.plot(x_trace, mean_trace)
    # calculate the error trace
    # error_trace = sem(trajectory_list)
    error_trace = np.nanstd(trajectory_list, axis=0)/np.sqrt(trajectory_list.shape[0])
    plots.fill_between(x_trace, mean_trace-error_trace, mean_trace+error_trace, alpha=0.5)
    plots.set_xlabel('Time')
    plots.set_ylabel(title_list[idx])
    plots.plot([encounter_positions/2, encounter_positions/2], plots.get_ylim())
    # plt.xlabel('Time')
    # plt.ylabel(title_list[idx])
plt.tight_layout()
fig.savefig(join(figure_save, 'huntTriggeredAverage_'+outcome_keyword+'_'+condition_keyword+'.png'), bbox_inches='tight')
# TODO: remove arbitrary mouse speed threshold (i.e. fix preprocessing)
# TODO: correct for frame rates in the acceleration calculations

# fig=plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(distance_vector)
# ax.plot(encounter_idx)
plt.show()

print('yay')
