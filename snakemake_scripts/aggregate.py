# imports
import functions_misc as fm
import functions_kinematic as fk
import functions_data_handling as fd
import paths
from scipy.ndimage.measurements import label
import pandas as pd
import numpy as np
import os
import yaml
import datetime


def aggregate_full_traces(partial_data):
    """Generate a file with the aggregated full traces from the entire queryset concatenated"""

    # create a frame vector for each dataset
    frame = [list(range(el.shape[0])) for el in partial_data]
    frame = np.concatenate(frame, axis=0)
    # concatenate the data
    partial_data = pd.concat(partial_data)

    # wrap angles
    partial_data.mouse_heading = fk.wrap(partial_data.mouse_heading)
    partial_data.cricket_heading = fk.wrap(partial_data.cricket_heading)

    # add the frame as a column
    partial_data['frame'] = frame

    return partial_data


def aggregate_bin_time(data_all):
    """Bin time and aggregate the traces for the queryset"""

    # normalize trial time and bin it
    # define the number of time bins
    number_timebins = 30
    # allocate memory for the binned time
    binned_trials = []
    # for all the trials
    for idx_in, data_in in enumerate(data_all):
        # get the time vector
        time_vector = data_in.time_vector.to_numpy()
        # bin it evenly in 10 bins
        time_bins = np.digitize(time_vector, np.histogram(time_vector, bins=number_timebins)[1], right=True)
        # bin the data correspondingly
        # binned_trials.append(np.array([np.mean(data.iloc[time_bins == (el+1), :-1], axis=0)
        # for el in range(number_timebins)]))
        binned_trials.append(data_in.groupby(time_bins).mean())
        # add a label column with the number of the trial to concatenate the whole thing
        binned_trials[-1]['trial_id'] = idx_in
        # add a label to the frames for grouping
        binned_trials[-1]['frame'] = np.arange(number_timebins+1)
    # binned_trials = np.array(binned_trials)
    # concatenate into a dataframe
    binned_trials = pd.concat(binned_trials)

    return binned_trials


def aggregate_encounters(data_all):
    """Aggregate the traces in the queryset based on encounters"""
    # TODO: fix the constant distance to define an encounter
    # define the time window width, centered on the encounter (in seconds)
    encounter_window = 5
    # allocate memory for the animal encounters
    encounter_pertrial = []

    # for all the trials
    for idx_in, data_in in enumerate(data_all):
        # TODO: improve encounter determination (don't eliminate beginning of a sequence if
        #  it doesn't overlap with the end)
        # identify the regions with encounters
        [encounter_idx, encounter_number] = label(data_in.mouse_cricket_distance.to_numpy() < 100)

        # run through the encounters
        # for encounters in range
        time_temp = data_in.loc[:, 'time_vector'].to_numpy()
        # set an encounter counter
        encounter_counter = 0
        # for all the encounters
        for encounters in range(1, encounter_number):
            # get the first coordinate of the encounter and grab the surroundings
            encounter_hit = np.nonzero(encounter_idx == encounters)[0][0]

            # get the starting time point of the encounter
            time_start = data_in.loc[encounter_hit, 'time_vector']
            # get the number of positions for this encounter
            encounter_start = np.argmin(np.abs(time_temp - (time_start-encounter_window/2)))
            encounter_end = np.argmin(np.abs(time_temp - (time_start+encounter_window/2)))

            if (encounter_end == data_in.shape[0]) or (encounter_start < 0):
                continue

            # also for the next encounter, unless it's the last one
            if encounters < encounter_number:
                encounter_hit2 = np.nonzero(encounter_idx == encounters+1)[0][0]
                time_start2 = data_in.loc[encounter_hit2, 'time_vector']
                encounter_start2 = np.argmin(np.abs(time_temp - (time_start2-encounter_window/2)))

                if encounter_start2 < encounter_end:
                    continue

            # store the distance and the speed around the encounter
            encounter_pertrial.append(data_in.iloc[encounter_start:encounter_end+1, :].copy())
            # correct the time axis
            encounter_pertrial[-1].loc[:, 'time_vector'] -= time_start
            # add the trial id
            encounter_pertrial[-1]['trial_id'] = idx_in
            # add the encounter id
            encounter_pertrial[-1]['encounter_id'] = encounter_counter
            # update the counter
            encounter_counter += 1

    # if there were no encounters, return an empty
    if len(encounter_pertrial) == 0:
        return []
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
                                                      axis=1).apply(fm.interp_trace, raw=False,
                                                                    args=(traces.time_vector.to_numpy(), max_time)))

        else:
            encounter_interpolated.append(encounter_pertrial[index].drop(['time_vector', 'trial_id', 'encounter_id'],
                                                                         axis=1))

        # add the time and id vectors back
        encounter_interpolated[-1]['time_vector'] = max_time
        encounter_interpolated[-1]['trial_id'] = traces.trial_id.iloc[0]
        encounter_interpolated[-1]['encounter_id'] = traces.encounter_id.iloc[0]
        # add the frame
        encounter_interpolated[-1]['frame'] = np.arange(max_time.shape[0])
    # average and sem across encounters
    encounter_matrix = pd.concat(encounter_interpolated)

    return encounter_matrix


# Main script
try:
    # get the raw output_path
    raw_path = snakemake.output[0]
    # get the parsed path
    dict_path = yaml.load(snakemake.params.output_info, Loader=yaml.FullLoader)
    # get the input paths
    paths_all = snakemake.input
    # get the parsed path info
    path_info = [yaml.load(el, Loader=yaml.FullLoader) for el in snakemake.params.file_info]
    # get the analysis type
    analysis_type = dict_path['analysis_type']
    # get a list of all the animals and dates involved
    animal_list = [el['mouse'] for el in path_info]
    date_list = [datetime.datetime.strptime(el['date'], '%Y-%m-%dT%H:%M:%SZ').date() for el in path_info]
except NameError:
    # define the analysis type
    analysis_type = 'aggEncCA'
    # define the search query
    search_query = 'result:succ,lighting:normal,rig:miniscope, imaging:doric'
    # define the origin model
    ori_type = 'preprocessing'
    # get a dictionary with the search terms
    dict_path = fd.parse_search_string(search_query)

    # get the info and paths
    path_info, paths_all, parsed_query, date_list, animal_list = \
        fd.fetch_preprocessing(search_query + ', =analysis_type:' + ori_type)
    # get the raw output_path
    dict_path['analysis_type'] = analysis_type
    basic_name = '_'.join(dict_path.values())
    raw_path = os.path.join(paths.analysis_path, '_'.join((ori_type, basic_name))+'.hdf5')

# get the sub key
sub_key = 'matched_calcium' if 'CA' in analysis_type else 'full_traces'
# read the data
data = [pd.read_hdf(el, sub_key) for el in paths_all]
# get the unique animals and dates
unique_mice = np.unique(animal_list) if 'CA' in analysis_type else [0]
unique_dates = np.unique(date_list) if 'CA' in analysis_type else [0]

# for all the mice
for idx, mouse in enumerate(unique_mice):
    # for all the dates
    for idx2, date in enumerate(unique_dates):

        # define the writing mode so that the file is overwritten in the first iteration
        mode = 'w' if idx+idx2 == 0 else 'a'

        # if there is no calcium, concatenate across the entire queryset
        if mouse == 0 and date == 0:
            sub_data = data
            group_key = analysis_type
        else:
            # get the data from the corresponding mouse and date
            sub_data = [el for idx, el in enumerate(data)
                        if (date_list[idx] == date) & (animal_list[idx] == mouse)]
            # if there is no data for this mouse/date combination, skip it
            if len(sub_data) == 0:
                continue
            # assemble the group key for the hdf5 file (making sure the date is natural)
            group_key = '/'.join(('', mouse, 'd' + str(date)[:10].replace('-', '_'), analysis_type))

        # select the function to run
        if 'aggFull' in analysis_type:
            sub_data = aggregate_full_traces(sub_data)
        elif 'aggBin' in analysis_type:
            sub_data = aggregate_bin_time(sub_data)
        elif 'aggEnc' in analysis_type:
            sub_data = aggregate_encounters(sub_data)
        else:
            raise ValueError('Action not recognized')

        # if the output is empty, print a message
        if len(sub_data) == 0:
            print('No encounters were found in day %s and mouse %s' % (date, mouse))
            continue
        # save to file
        fd.save_create_snake(sub_data, paths_all, raw_path, group_key, dict_path, action='save', mode=mode)

# create the entry
fd.save_create_snake([], paths_all, raw_path, analysis_type, dict_path, action='create')
