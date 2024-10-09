# imports
import functions_misc as fm
from src import functions_kinematic as fk
import functions_data_handling as fd
import functions_preprocessing as fp
import paths
from scipy.ndimage.measurements import label
import pandas as pd
import numpy as np
import os
import yaml
import datetime
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# mpl.use('Qt5Agg')
import processing_parameters


def aggregate_full_traces(partial_data):
    """Generate a file with the aggregated full traces from the entire queryset concatenated"""

    # create a frame vector for each dataset
    frame = [list(range(el.shape[0])) for el in partial_data]
    frame = np.concatenate(frame, axis=0)
    # also add the trial id
    trial_id = [[index]*el.shape[0] for index, el in enumerate(partial_data)]
    trial_id = np.concatenate(trial_id, axis=0)

    # concatenate the data
    out_data = pd.concat(partial_data)

    # wrap angles
    # out_data.mouse_heading = fk.wrap(out_data.mouse_heading)
    # out_data.cricket_heading = fk.wrap(out_data.cricket_heading)

    # add the frame as a column
    out_data['frame'] = frame
    out_data['trial_id'] = trial_id

    return out_data


def aggregate_bin_time(data_all):
    """Bin time and aggregate the traces for the queryset"""

    # normalize trial time and bin it
    # define the number of time bins
    number_timebins = 30
    # allocate memory for the binned time
    binned_trials = []
    # for all the trials
    for idx_in, data_in in enumerate(data_all):
        # save the mouse and date
        mouse = data_in.loc[0, 'mouse']
        date = data_in.loc[0, 'date']
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
        # reassign mouse and date
        binned_trials[-1]['mouse'] = mouse
        binned_trials[-1]['date'] = date
    # binned_trials = np.array(binned_trials)
    # concatenate into a dataframe
    binned_trials = pd.concat(binned_trials)

    return binned_trials


def aggregate_encounters(data_all):
    """Aggregate the traces in the queryset based on encounters"""
    # TODO: fix the constant distance to define an encounter
    # define the time window width, centered on the encounter (in seconds)
    encounter_window = 2
    # # allocate memory for the animal encounters
    # encounter_pertrial = []

    # set a list for concatenation
    encounter_list = []
    # initialize an encounter counter variable
    enc_counter = 0

    # define the thresholding function
    def thres_function(param, thres):
        return param < thres

    def thres_function_unity(param, thres):
        return param == thres

    # for all the trials
    for idx_in, data_in in enumerate(data_all):

        # # set a list for concatenation
        # encounter_list = []

        # if there are virtual crickets, iterate through them
        if 'vrcricket_0_x' in data_in.columns:
            # get the number of crickets
            vr_cricket_list = np.unique([el[:11] for el in data_in.columns if 'vrcricket' in el])
            # set the distance threshold
            vr_thresh = 0.07

            # for all the vr crickets
            for vrcricket in vr_cricket_list:

                # get encounters according to Unity
                encounters_unity = fp.timed_event_finder(data_in, vrcricket+'_encounter', 1, thres_function_unity,
                                                        window=encounter_window)

                # get the encounters
                encounters_temp = fp.timed_event_finder(data_in, vrcricket+'_mouse_distance', vr_thresh, thres_function,
                                                        window=encounter_window)

                # # For debugging
                # plt.plot(data_in['time_vector'], data_in[vrcricket+'_mouse_distance'])
                # plt.plot(data_in['time_vector'], vr_thresh * data_in[vrcricket + '_encounter'], "*")
                # plt.hlines(vr_thresh, 0, data_in['time_vector'].max())
                # plt.show()

                # if no encounters were found, skip
                if len(encounters_temp) == 0:
                    continue
                # drop all columns not having to do with this cricket
                for column in encounters_temp.columns:
                    if 'vrcricket' in column and vrcricket not in column:
                        encounters_temp.drop([column], axis=1, inplace=True)
                    elif vrcricket in column:
                        encounters_temp.rename(columns={column: column.replace(vrcricket, 'vrcricket_0')},
                                               inplace=True)

                encounter_list.append(encounters_temp)
                # if it's not the first one, concatenate to the previous ones
        # if a real cricket
        if 'cricket_0_x' in data_in.columns:
            # get the encounters
            # 100 was used for the poster
            # encounters_temp = fp.timed_event_finder(data_in, 'cricket_0_mouse_distance', 19.5, thres_function,
            #                                         window=encounter_window)
            # This was changed to real units with the DLC to Motive transformation
            encounters_temp = fp.timed_event_finder(data_in, 'cricket_0_mouse_distance', 2, thres_function,
                                                    window=encounter_window)

            # if no encounters were found, skip
            if len(encounters_temp) == 0:
                continue

            # before appending, eliminate the conflicting fields from unity
            # TODO: REMOVE THIS HACK
            for column in encounters_temp.columns:
                if column in ['head_direction', 'head_height', 'cricket_0_delta_head']:
                    encounters_temp.drop([column], axis=1, inplace=True)

            # add the trial column
            encounters_temp.loc[:, 'trial_id'] = idx_in
            # update the counter variable and column
            if len(encounter_list) > 0:
                enc_counter = np.max(encounter_list[-1]['event_id']) + 1

            encounters_temp.loc[:, 'event_id'] += enc_counter

            encounter_list.append(encounters_temp)

    # # if no encounters were found, skip
    # if len(encounter_list) == 0:
    #     continue

    # concatenate the list
    encounter_matrix = pd.concat(encounter_list, axis=0)

    # add the trial ID
    # encounters_local.loc[:, 'trial_id'] = idx_in
    # if encounters_local.shape[1] != 15:
    #     print('stop')

    # # append to the list
    # encounter_pertrial.append(encounter_list)

    # # if there were no encounters, return an empty
    # if len(encounter_pertrial) == 0:
    #     return []

    # encounter_matrix = pd.concat(encounter_pertrial)

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
    analysis_type = processing_parameters.analysis_type
    # define the search query
    search_query = processing_parameters.search_string

    # define the origin model
    ori_type = 'preprocessing'
    # get a dictionary with the search terms
    dict_path = fd.parse_search_string(search_query)

    # get the info and paths
    path_info, paths_all, parsed_query, date_list, animal_list = \
        fd.fetch_preprocessing(search_query + ', analysis_type:' + ori_type)
    # get the raw output_path
    dict_path['analysis_type'] = analysis_type
    basic_name = '_'.join(dict_path.values())
    raw_path = os.path.join(paths.analysis_path, '_'.join((ori_type, basic_name))+'.hdf5')

# get a list of the unique mice to split the loading
# unique_mice = np.unique(animal_list) if 'CA' in analysis_type else [0]
unique_mice = np.unique(animal_list)
# get the sub key
# sub_key = 'matched_calcium' if 'CA' in analysis_type else 'full_traces'
# allocate memory for the dataframe per animal
sub_data = []
# for all the mice
for mouse in unique_mice:
    # allocate memory for the data
    mouse_data = []
    mouse_paths = [el for idx, el in enumerate(paths_all) if mouse == animal_list[idx]]
    # read the data
    for idx, el in enumerate(mouse_paths[:10]):
        try:
            behavior = pd.read_hdf(el, 'matched_calcium')
            # get rid of the cells (for now at least)
            not_cells = [el for el in behavior.columns if 'cell' not in el]
            behavior = behavior.loc[:, not_cells]
            latents = pd.read_hdf(el, 'latents')
        except KeyError:
            behavior = pd.read_hdf(el, 'full_traces')
            # drop the sync frames column
            if 'sync_frames' in behavior.columns:
                behavior.drop(columns=['sync_frames'], inplace=True)
                latents = pd.read_hdf(el, 'latents')
            else:
                continue
        # add the date and mouse
        behavior.loc[:, 'mouse'] = animal_list[idx]
        behavior.loc[:, 'date'] = str(date_list[idx])

        # get the delta frames between latent and behavior
        delta_frames = behavior.shape[0] - latents.shape[0]
        padding = pd.DataFrame(np.zeros((int(delta_frames/2), len(latents.columns))), columns=latents.columns)
        # pad latents due to the VAME calculation window
        latents = pd.concat([padding, latents, padding], axis=0).reset_index(drop=True)
        mouse_data.append(pd.concat([behavior, latents], axis=1))

    # # get the unique animals and dates
    # unique_mice = np.unique(animal_list) if 'CA' in analysis_type else [0]
    # unique_dates = np.unique(date_list) if 'CA' in analysis_type else [0]
    #
    # # define a flag for the first save
    # first_save = True
    # # for all the mice
    # for idx, mouse in enumerate(unique_mice):
    #     # for all the dates
    #     for idx2, date in enumerate(unique_dates):

    # # if there is no calcium, concatenate across the entire queryset
    # if mouse == 0 and date == 0:
    # sub_data = data
    # else:
    #     # get the data from the corresponding mouse and date
    #     sub_data = [el for idx, el in enumerate(data)
    #                 if (date_list[idx] == date) & (animal_list[idx] == mouse)]
    #     # if there is no data for this mouse/date combination, skip it
    #     if len(sub_data) == 0:
    #         continue
    #     # assemble the group key for the hdf5 file (making sure the date is natural)
    #     group_key = '/'.join(('', mouse, 'd' + str(date)[:10].replace('-', '_'), analysis_type))

    # select the function to run
    # if 'aggFull' in analysis_type:
    #     sub_data = aggregate_full_traces(sub_data)
    if 'aggBin' in analysis_type:
        mouse_data = aggregate_bin_time(mouse_data)
    elif 'aggEnc' in analysis_type:
        mouse_data = aggregate_encounters(mouse_data)
    else:
        raise ValueError('Action not recognized')

    sub_data.append(mouse_data)

# # if the output is empty, print a message
# if len(sub_data) == 0:
#     print('No encounters were found in day %s and mouse %s' % (date, mouse))
#     continue
# if the flag is still on, create the file, otherwise append
# if first_save:
#     first_save = False
#     mode = 'w'
# else:
#     mode = 'a'

# concatenate before saving
sub_data = pd.concat(sub_data, axis=0)
# # convert date and mouse to pandas strings
# sub_data.loc[:, 'mouse'] = sub_data.loc[:, 'mouse'].astype(pd.StringDtype())
# sub_data.loc[:, 'date'] = sub_data.loc[:, 'date'].astype(pd.StringDtype())

# save to file
fd.save_create_snake(sub_data, paths_all, raw_path, analysis_type, dict_path, action='save', mode='w')

# create the entry
fd.save_create_snake([], paths_all, raw_path, analysis_type, dict_path, action='create')
