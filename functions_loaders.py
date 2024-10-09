import pandas as pd
import numpy as np
import os
import datetime
import h5py
import processing_parameters
import functions_bondjango as bd


def pad_latents(input_list, target_size):
    """Get the latents from a preprocessing file and add them to the main dataframe with the correct padding"""

    # allocate an output list
    output_list = []
    # for all the elements in the input list
    for el in input_list:
        # determine the delta size for padding
        delta_frames = target_size - el.shape[0]
        # pad latents due to the VAME calculation window
        latent_padding = pd.DataFrame(np.zeros((int(delta_frames / 2), len(el.columns))) * np.nan,
                                      columns=el.columns)
        # motif_padding = pd.DataFrame(np.zeros((int(delta_frames / 2), len(motifs.columns))) * np.nan,
        #                              columns=motifs.columns)
        # pad them with nan at the edges (due to VAME excluding the edges
        latents = pd.concat([latent_padding, el, latent_padding], axis=0).reset_index(drop=True)
        # add to the output list
        output_list.append(latents)
        # motifs = pd.concat([motif_padding, motifs, motif_padding], axis=0).reset_index(drop=True)

    return output_list


def query_search_list():
    """Query the database based on the search_list field from processing_parameters"""
    # get the search list
    search_list = processing_parameters.search_list

    # allocate a list for all paths (need to preload to get the dates)
    path_list = []
    query_list = []
    # for all the search strings
    for search_string in search_list:
        # query the database for data to plot
        data_all = bd.query_database('analyzed_data', search_string)
        data_path = [el['analysis_path'] for el in data_all if '_preproc' in el['slug']]
        path_list.append(data_path)
        query_list.append(data_all)
    return path_list, query_list


def load_preprocessing(input_path, data_all, latents_flag=True, matching_flag=True, behavior_flag=False):
    """Load data from a list of preprocessing paths and their database data"""
    # load the data
    data_list = []
    meta_list = []
    frame_list = []
    for idx, el in enumerate(input_path):
        # get the trial timestamp (for frame calculations)
        time_stamp = int(''.join(os.path.basename(el).split('_')[3:6]))
        try:
            if behavior_flag:
                temp_data = pd.read_hdf(el, 'full_traces')
            else:
                try:
                    temp_data = pd.read_hdf(el, 'matched_calcium')
                except KeyError:
                    continue
            temp_data['id'] = data_all[idx]['id']
            meta_list.append([data_all[idx][el1] for el1 in processing_parameters.meta_fields])
            # if the latents flag is on, load the latents too
            if latents_flag:
                # try to load the motifs and latents
                try:
                    latents = pd.read_hdf(el, 'latents')
                    motifs = pd.read_hdf(el, 'motifs')
                    egocentric_coords = pd.read_hdf(el, 'egocentric_coord')
                    egocentric_coords = egocentric_coords.loc[:, ['cricket_0_x', 'cricket_0_y']]
                    egocentric_coords = egocentric_coords.rename(columns={'cricket_0_x': 'ego_cricket_x',
                                                                          'cricket_0_y': 'ego_cricket_y'})

                    # pad them with nan at the edges (due to VAME excluding the edges
                    [latents, motifs] = pad_latents([latents, motifs], temp_data.shape[0])
                    # concatenate with the main data
                    temp_data = pd.concat([temp_data, egocentric_coords, latents, motifs], axis=1)
                except KeyError:
                    print(f'No latents in file {el}')

            # if the matching flag is on, also load the ROI matching

            data_list.append(temp_data)
            frame_list.append([time_stamp, 0, temp_data.shape[0]])
        except KeyError:
            # data_list.append([])
            frame_list.append([time_stamp, 0, 0])
    return data_list, frame_list, meta_list


def load_regression(all_paths, variable_list, time_shifts):
    """Load regression files from a list of paths and the databased info"""
    # allocate the outputs
    correlations = []
    weights = []

    # allocate internal lists (to not do double loading given weights and performance
    # are the same across trials on the same day)
    animal_list = []
    day_list = []
    joint_list = []
    # get the regression types
    regressors = processing_parameters.regressors

    # for all the list items
    for idx0, data_path in enumerate(all_paths):

        # for all the files
        for idx1, files in enumerate(data_path):

            # if a habi trial, skip
            if 'habi' in files:
                continue

            # get the animal and date from the slug
            name_parts = os.path.basename(files).split('_')
            animal = '_'.join(name_parts[7:10])
            day_s = '_'.join(name_parts[:3])
            time_s = '_'.join(name_parts[3:6])
            day = datetime.datetime.strptime(day_s, '%m_%d_%Y')

            # skip if the animal and day are already evaluated,
            # since the CC and weights are the same for the whole day (not for prediction, so leave here for now)
            if animal + '_' + day_s in joint_list:
                skip_flag = True
            else:
                skip_flag = False
                animal_list.append(animal)
                day_list.append(day)
                joint_list.append(animal + '_' + day_s)
            #         # get the cell matches (UNUSED FOR NOW)
            #         current_idx = match_cells(files)

            # load the data and the cell matches (wasteful, but cleaner I think)
            with h5py.File(files, 'r') as h:
                if 'regression' not in h.keys():
                    continue
                # get the keys present
                key_list = h['regression'].keys()
                # allocate the list to accumulate weights
                weights_across_features = []
                # for all the variables
                for idx_feature, feature in enumerate(variable_list):
                    cc_feature_list = []
                    weight_feature_list = []

                    # get the feature keys
                    current_feature = [el for el in key_list if feature in el]
                    # for all the regression types
                    for reg in regressors:
                        # get the relevant keys
                        current_regressor = [el for el in current_feature if reg in el]
                        # for real vs shuffle
                        for rvs in ['real', 'shuffle']:
                            # get the real/shuffle keys
                            current_rvs = [el for el in current_regressor if rvs in el]
                            # for the time shifts
                            for shift in time_shifts:
                                # get the current time keys
                                current_shift = [el for el in current_rvs if str(shift) in el]

                                # process the performances and weights
                                if not skip_flag:

                                    # performance
                                    current_correlation = [el for el in current_shift if
                                                           ('cc' in el) and ('_std' not in el)]
                                    assert len(current_correlation) == 1, 'more than one item in the cc list'
                                    cc_feature_list.append(
                                        [feature, np.array(h['/regression/' + current_correlation[0]]), reg, rvs,
                                         str(shift), animal, day])
                                    # weights
                                    current_weights = [el for el in current_shift if
                                                       ('coefficients' in el) and ('_std' not in el)]
                                    # create a dataframe
                                    weight_df = pd.DataFrame(np.array(h['/regression/' + current_weights[0]]),
                                                             columns=[feature])
                                    # if it's the first element, add the meta fields
                                    if idx_feature == 0:
                                        weight_df.insert(0, 'day', day)
                                        weight_df.insert(0, 'animal', animal)
                                        weight_df.insert(0, 'shift', str(shift))
                                        weight_df.insert(0, 'rvs', rvs)
                                        weight_df.insert(0, 'cell_id', np.arange(weight_df.shape[0]))
                                        weight_df.insert(0, 'regressor', reg)

                                    # store
                                    weight_feature_list.append(weight_df)
                    if not skip_flag:
                        # save the entries as a feature in the dict
                        correlations.extend(cc_feature_list)
                        weights_across_features.append(pd.concat(weight_feature_list, axis=0).reset_index(drop=True))
            if not skip_flag:
                # concatenate the features for the day
                weights.append(pd.concat(weights_across_features, axis=1).reset_index(drop=True))
    # convert to dataframe
    correlations = pd.DataFrame(correlations, columns=['feature', 'cc', 'regressor', 'rvs', 'shift', 'mouse', 'day'])
    correlations['cc'] = correlations['cc'].astype(float)
    # concatenate the weights across days
    weights = pd.concat(weights, axis=0).reset_index(drop=True)

    print(f'Shape of the performance dataframe: {correlations.shape}')
    print(f'Shape of the weights dataframe: {weights.shape}')
    return correlations, weights


def load_tc(all_paths, variable_list):
    """Load the tuning curves from a list of paths and the database info"""
    # define the relevant column types
    index_columns = ['Resp_index', 'Cons_index', 'Qual_index', 'Resp_test', 'Cons_test', 'Qual_test']
    # allocate the output
    tc_whole = []

    # for all the groups
    for group in all_paths:
        # for all the files
        for file in group:
            # allocate a per-file list
            per_file_list = []
            # read the meta
            try:
                meta_data = pd.read_hdf(file, 'meta_data')

                # skip the file if habi
                if meta_data.loc[0, 'result'] == 'habi':
                    continue
            except KeyError:
                print(f'file {file} skipped as there was no meta data')
                continue
            # for all the features
            for idx_feature, feature in enumerate(variable_list):

                # load the data
                data = pd.read_hdf(file, feature)
                # add the meta fields if it's the first
                if idx_feature == 0:
                    # add the cell_id column
                    data['cell_id'] = np.arange(data.shape[0])
                    # get the columns
                    target_columns = ['day', 'animal', 'cell_id'] + index_columns + [el for el in data.columns if
                                                                                     ('bin' in el) & ('half' not in el)]
                else:
                    target_columns = index_columns + [el for el in data.columns if ('bin' in el) & ('half' not in el)]
                # get only the relevant columns
                data = data[target_columns]
                # add the feature name to the columns
                new_names = {el: feature + '_' + el if ('bin' in el) | ('index' in el) | ('test' in el) else el for el in
                             target_columns}
                data = data.rename(columns=new_names)
                # store
                per_file_list.append(data)
            # concatenate across features and store
            tc_whole.append(pd.concat(per_file_list, axis=1).reset_index(drop=True))

    # concatenate across files
    tc_whole = pd.concat(tc_whole, axis=0).reset_index(drop=True)
    return tc_whole


