# imports
import functions_misc as fm
import functions_kinematic as fk
import functions_data_handling as fd
from scipy.ndimage.measurements import label
import pandas as pd
import numpy as np


def aggregate_full_traces(search_query):
    """Generate a file with the aggregated full traces from the entire queryset concatenated"""
    # define the origin analysis type
    ori_type = 'preprocessing'
    # define the suffix for this function
    analysis_type = 'aggFull'
    # get the data and paths
    data_all, paths_all, parsed_query, _, _ = \
        fd.fetch_data(search_query+', analysis_type='+ori_type, sub_key='full_traces')

    # concatenate the data
    kinematic_parameters = pd.concat(data_all)

    # wrap angles
    kinematic_parameters.mouse_heading = fk.wrap(kinematic_parameters.mouse_heading)
    kinematic_parameters.cricket_heading = fk.wrap(kinematic_parameters.cricket_heading)

    # save the file and create the entry
    fd.save_create(kinematic_parameters, paths_all, analysis_type, parsed_query)

    return None


def aggregate_bin_time(search_query):
    """Bin time and aggregate the traces for the queryset"""
    # define the origin analysis type
    ori_type = 'preprocessing'
    # define the suffix for this function
    analysis_type = 'aggBin'
    # get the data and paths
    data_all, paths_all, parsed_query, _, _ = \
        fd.fetch_data(search_query+', analysis_type='+ori_type, sub_key='full_traces')
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

    # save the file and create the entry
    fd.save_create(binned_trials, paths_all, analysis_type, parsed_query)

    return None


def aggregate_encounters(search_query):
    """Aggregate the traces in the queryset based on encounters"""
    # define the origin analysis type
    ori_type = 'preprocessing'
    # define the suffix for this function
    analysis_type = 'aggEnc'
    # get the data and paths
    data_all, paths_all, parsed_query, _, _ = \
        fd.fetch_data(search_query+', analysis_type='+ori_type, sub_key='full_traces')

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

    # save the file and create the entry
    fd.save_create(encounter_matrix, paths_all, analysis_type, parsed_query)
    return None


if __name__ == '__main__':

    # define the search query
    search_string = 'result=fail, lighting=normal, rig=miniscope, notes=BLANK'
    # run the functions
    aggregate_full_traces(search_string)
    aggregate_bin_time(search_string)
    aggregate_encounters(search_string)
