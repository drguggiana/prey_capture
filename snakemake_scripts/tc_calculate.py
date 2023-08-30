# imports
import os
import paths
import functions_bondjango as bd
import pandas as pd
import numpy as np
import scipy.signal as ss
import scipy.stats as stat
import processing_parameters
import functions_misc as fm
import yaml

np.seterr(divide='ignore', invalid='ignore')


def calculate_information(occupancy, tuning_curve, average):
    """Calculate the information on a tuning curve based on Stefanini et al. 2020"""
    information = np.nansum(occupancy*(tuning_curve/average) * np.log2(tuning_curve/average))
    return information


def clipping_function(trace_in, threshold=8):
    """Clip traces to their threshold-th percentile"""
    # skip if there are only zeros
    if np.sum(trace_in) == 0:
        return trace_in
    # get the baseline
    baseline = np.percentile(trace_in[trace_in > 0], threshold)

    # clip the trace
    trace_in[trace_in < baseline] = 0
    return trace_in


def clip_calcium(pre_data):
    """ Clip the calcium traces based on baseline """

    # allocate memory for the cleaned up data
    data = []
    # define the clipping threshold in percentile of baseline
    clip_threshold = 8
    # for all the trials
    for idx, el in enumerate(pre_data):

        # get the current df
        current_df = el[1]
        labels = list(current_df.columns)
        cells = [el for el in labels if 'cell' in el]
        not_cells = [el for el in labels if 'cell' not in el]
        # get the non-cell data
        non_cell_data = current_df[not_cells]
        # get the current calcium data
        cell_data = current_df[cells].fillna(0)

        # do the cell clipping
        cell_data.apply(clipping_function, axis=1, raw=True, threshold=clip_threshold)

        # assemble a new data frame with only the matched cells and the rest of the data
        data.append(pd.concat((non_cell_data, cell_data), axis=1))
    return data


def parse_features(data, feature_list, bin_number=10):
    """set up the feature and calcium matrices"""

    # allocate memory for a data frame without the encoding model features
    feature_raw_trials = []
    # allocate memory for the calcium
    calcium_trials = []

    # get the features
    for idx, el in enumerate(data):
        # get the intersection of the labels
        label_intersect = [feat for feat in feature_list if feat in el.columns]

        # # add the y coordinate of the variables with x
        # coordinate_variables = [column.replace('_x', '_y') for column in label_intersect if '_x' in column]
        # label_intersect += coordinate_variables

        # get the features of interest
        target_features = el.loc[:, label_intersect]
        # get the original columns
        original_columns = target_features.columns

        # for all the columns
        for label in original_columns:
            # skip if latent or motif
            if ('latent' in label) | (label == 'motifs'):
                target_features[label] = target_features[label]
                continue

            # smooth the feature
            target_features[label] = ss.medfilt(target_features[label], 21)

        # # allocate a copy of the target features for changes
        # temp_features = target_features.copy()
        # # for the coordinate variables, turn into a 2D grid
        # for variable in coordinate_variables:
        #     x_variable = target_features[variable.replace('_y', '_x')].to_numpy()
        #     y_variable = target_features[variable].to_numpy()
        #     bin_ranges = processing_parameters.tc_params[variable.replace('_y', '_x')]
        #     bins = np.linspace(bin_ranges[0], bin_ranges[1], num=bin_number + 1)
        #     # bin the variables in 2D
        #     current_tc = \
        #         stat.binned_statistic_2d(x_variable, y_variable, y_variable, statistic='count', bins=bins,
        #                                  expand_binnumbers=True)
        #
        #     binnumbers = current_tc[3]
        #     # current_tc = np.ravel_multi_index((current_tc[3][0, :], current_tc[3][1, :]), (bin_ranges[0], bin_ranges[1]), mode='clip')
        #     current_tc = np.ravel_multi_index(binnumbers, (11, 11), mode='raise')
            # replace the x column in the target features

            # eliminate the

        # store the features
        feature_raw_trials.append(target_features)

        # get the calcium data
        cells = [cell for cell in el.columns if 'cell' in cell]
        cells = el.loc[:, cells].to_numpy()

        # store
        calcium_trials.append(cells)

    return feature_raw_trials, calcium_trials


def extract_half_tc(current_feature_0, cell_number, calcium_trials, feature_counts, bins):
    """Get the the half tuning curves for consistency calculation"""
    tc_half_temp = []
    # for first and second half
    for half in np.arange(2):
        # get the half vector
        half_bound = int(np.floor(current_feature_0.shape[0] / 2))
        half_vector = np.arange(half_bound) + half_bound * half
        half_feature_0 = current_feature_0[half_vector]
        # exclude nan values
        keep_vector = ~np.isnan(half_feature_0)
        keep_feature_0 = half_feature_0[keep_vector]

        # allocate a list for the cells
        tc_cell = []
        # for all the cells
        for cell in np.arange(cell_number):
            # get the current cell
            half_cell = calcium_trials[half_vector, cell]
            keep_cell = half_cell[keep_vector]
            # get the tc
            current_tc = \
                stat.binned_statistic(keep_feature_0, keep_cell, statistic='sum', bins=bins)[0]

            # normalize the TC
            norm_tc = current_tc / feature_counts
            # remove nans and infs
            norm_tc[np.isnan(norm_tc)] = 0
            norm_tc[np.isinf(norm_tc)] = 0
            # store
            tc_cell.append(norm_tc)
        # store the cells
        tc_half_temp.append(tc_cell)
    return tc_half_temp


def extract_full_tc(counts_feature_0, feature_counts, cell_number,
                    calcium_trials, bins, keep_vector_full, shuffle_number, working_bin_number, percentile):
    """Get the full tc"""
    # allocate memory for the full tc per cell
    tc_cell_full = []
    tc_cell_resp = np.zeros((cell_number, 4))

    # calculate the full TC
    for cell in np.arange(cell_number):
        keep_cell = calcium_trials[keep_vector_full, cell]

        tc_cell, _, tc_idx = \
            stat.binned_statistic(counts_feature_0, keep_cell, statistic='sum', bins=bins)
        # get the information
        information_content = calculate_information(feature_counts, tc_cell, np.mean(keep_cell))
        # process the TC
        tc_cell = tc_cell / feature_counts
        tc_cell[np.isnan(tc_cell)] = 0
        tc_cell[np.isinf(tc_cell)] = 0

        # use the tc_idx to regenerate the activity
        predicted_calcium = tc_cell[tc_idx-2]
        # get the correlation with the real calcium
        tc_quality = stat.spearmanr(keep_cell, predicted_calcium, nan_policy='omit')[0]
        # allocate memory for the shuffles
        # shuffle_array = np.zeros((shuffle_number, working_bin_number))
        shuffle_array = np.zeros((shuffle_number, 1))
        shuffle_prediction = np.zeros((shuffle_number, 1))
        # generate the shuffles
        for shuffle in np.arange(shuffle_number):
            # randomize the calcium activity
            random_cell = keep_cell.copy()
            random_cell = np.random.choice(random_cell, keep_cell.shape[0])

            tc_random = \
                stat.binned_statistic(counts_feature_0, random_cell, statistic='sum', bins=bins)[0]

            # get the information
            shuffle_array[shuffle] = calculate_information(feature_counts, tc_random, np.mean(random_cell))
            # process the TC
            tc_random = tc_random / feature_counts
            tc_random[np.isnan(tc_random)] = 0
            tc_random[np.isinf(tc_random)] = 0
            # shuffle_array[shuffle, :] = tc_random
            # use the tc_idx to regenerate the activity
            predicted_calcium = tc_random[tc_idx-2]
            # get the correlation with the real calcium
            random_quality = stat.spearmanr(random_cell, predicted_calcium, nan_policy='omit')[0]
            shuffle_prediction[shuffle] = random_quality

        # get the threshold
        resp_threshold = np.percentile(np.abs(shuffle_array.flatten()), percentile)
        qual_threshold = np.percentile(np.abs(shuffle_prediction.flatten()), percentile)
        # fill up the responsivity matrix
        # tc_cell_resp[cell, 0] = np.mean(np.sort(np.abs(tc_cell), axis=None)[-3:]) / resp_threshold
        # tc_cell_resp[cell, 1] = np.sum(np.abs(tc_cell) > resp_threshold) > 3
        tc_cell_resp[cell, 0] = information_content
        tc_cell_resp[cell, 1] = np.abs(information_content) > resp_threshold
        tc_cell_resp[cell, 2] = tc_quality
        tc_cell_resp[cell, 3] = np.abs(tc_quality) > qual_threshold
        # store
        tc_cell_full.append(tc_cell)
    return tc_cell_full, tc_cell_resp


def extract_tcs_responsivity(feature_raw_trials, calcium_trials, target_variables, cell_number,
                             percentile=99, bin_number=10):
    """Extract the tuning curves (full and half) and their responsivity index"""

    # get the number of pairs
    var_number = len(target_variables)
    # define the number of calcium shuffles
    shuffle_number = 100

    # allocate memory for the trial TCs
    tc_half = {}
    tc_full = {}
    tc_resp = {}
    tc_counts = {}
    tc_edges = {}
    # initialize the template_idx
    template_idx = -1
    # for all the features
    for var_idx in np.arange(var_number):
        # get the current feature
        feature_name = target_variables[var_idx]
        # feature_names = feature_name.split('__')
        # skip the pair and save an empty if the feature is not present
        try:
            current_feature_0 = feature_raw_trials.loc[:, feature_name].to_numpy()
            # save the index of the feature
            template_idx = var_idx
            # current_feature_1 = feature_raw_trials.loc[:, feature_names[1]].to_numpy()
        except KeyError:
            tc_half[feature_name] = []
            tc_full[feature_name] = []
            tc_resp[feature_name] = []
            tc_counts[feature_name] = []
            tc_edges[feature_name] = []
            continue
        # get the bins from the parameters file
        try:
            bin_ranges = processing_parameters.tc_params[feature_name]
            # calculate the bin edges based on the ranges
            if len(bin_ranges) == 1:
                bins = np.arange(bin_ranges[0] + 1) - 0.5
                # bins = bin_ranges[0]
                working_bin_number = bins.shape[0] - 1
                # working_bin_number = bin_ranges[0]
            else:
                bins = np.linspace(bin_ranges[0], bin_ranges[1], num=bin_number + 1)
                working_bin_number = bin_number
        except KeyError:
            # if not in the parameters, go for default and report
            print(f'Feature {feature_name} not found, default to 10 bins ad hoc')
            bins = 10
            working_bin_number = 10

        # exclude nan values
        keep_vector_full = ~np.isnan(current_feature_0)
        counts_feature_0 = current_feature_0[keep_vector_full]
        # get the counts for each bin
        feature_counts_raw, tc_current_edges, _ = \
            stat.binned_statistic(counts_feature_0, counts_feature_0, statistic='count', bins=bins)
        feature_counts = feature_counts_raw.copy()

        # zero the positions with less than 3 counts
        feature_counts[feature_counts < 3] = 0
        # get the half tuning curves
        tc_half_temp = extract_half_tc(current_feature_0, cell_number, calcium_trials, feature_counts, bins)
        # get the full tuning curves
        tc_cell_full, tc_cell_resp = extract_full_tc(counts_feature_0, feature_counts, cell_number,
                                                     calcium_trials, bins, keep_vector_full, shuffle_number,
                                                     working_bin_number, percentile)
        # store the halves and fulls
        tc_half[feature_name] = tc_half_temp
        tc_full[feature_name] = tc_cell_full
        tc_resp[feature_name] = tc_cell_resp
        tc_counts[feature_name] = feature_counts_raw
        tc_edges[feature_name] = tc_current_edges
    # run through the features and fill up the non-populated ones with nan
    for feat in tc_half.keys():
        if len(tc_half[feat]) == 0:
            tc_half[feat] = tc_half[target_variables[template_idx]]
            tc_half[feat][0] = [el * np.nan for el in tc_half[feat][0]]
            tc_half[feat][1] = [el * np.nan for el in tc_half[feat][1]]
            tc_full[feat] = tc_full[target_variables[template_idx]]
            tc_full[feat] = [el * np.nan for el in tc_full[feat]]
            tc_resp[feat] = tc_resp[target_variables[template_idx]] * np.nan
            tc_counts[feat] = tc_counts[target_variables[template_idx]] * np.nan
            tc_edges[feat] = tc_edges[target_variables[template_idx]] * np.nan
    return tc_half, tc_full, tc_resp, tc_counts, tc_edges


def extract_consistency(tc_half, target_variables, cell_number, percentile=95):
    """Calculate TC consistency"""

    # define the number of shuffles
    shuffle_number = 100

    # get the number of pairs
    var_number = len(target_variables)

    # allocate memory for the trial TCs
    tc_cons = {}
    # for all the features
    for var_idx in np.arange(var_number):

        # get the name
        feature_name = target_variables[var_idx]
        # allocate an array for the correlations and tests
        tc_half_temp = np.zeros([cell_number, 2])
        # get the two halves
        halves = tc_half[feature_name]
        # if empty, skip
        if len(halves) == 0:
            tc_cons[feature_name] = []
            continue

        # calculate the real and shuffle correlation
        for cell in np.arange(cell_number):
            # get the current cell first and second half
            current_first = halves[0][cell].flatten()
            current_second = halves[1][cell].flatten()
            # real correlation
            real_correlation = np.corrcoef(current_first, current_second)[1][0]

            # shuffle array
            shuffle_array = np.zeros([shuffle_number, 1])
            # calculate the confidence interval
            for shuffle in np.arange(shuffle_number):
                random_second = current_second.copy().flatten()
                random_second = np.random.choice(random_second, random_second.shape[0])
                shuffle_array[shuffle] = np.corrcoef(current_first, random_second)[1][0]
            # turn nans into 0
            shuffle_array[np.isnan(shuffle_array)] = 0

            # get the confidence interval
            conf_interval = np.percentile(shuffle_array, percentile)
            # store the correlation and whether it passes the criterion
            tc_half_temp[cell, 0] = real_correlation
            tc_half_temp[cell, 1] = (real_correlation > conf_interval) & (real_correlation > 0) & \
                                    (conf_interval > 0)

        # store for the variable
        tc_cons[feature_name] = tc_half_temp
    return tc_cons


def convert_to_dataframe(half_in, full_in, counts_in, resp_in, cons_in, edges_in, date, mouse, setup):
    """Convert the TCs and their metrics into dataframe format"""
    # allocate an output dict
    out_dict = {}
    # also one for the counts and edges
    count_dict = {}
    edges_dict = {}
    # cycle through features
    for feat in half_in.keys():
        # get all the components
        c_half = half_in[feat]
        c_full = full_in[feat]
        c_count = counts_in[feat]
        c_resp = resp_in[feat]
        c_cons = cons_in[feat]
        c_edges = edges_in[feat]

        # if the feature is not present, skip
        if len(c_half) == 0:
            continue
        # flatten the tcs and generate labels
        flat_half = []
        labels_half = []
        for half in np.arange(2):
            flat_half.append(np.array([el.flatten() for el in c_half[half]]))
            labels_half.append(['half_'+str(half)+'_bin_'+str(el) for el in np.arange(flat_half[half].shape[1])])

        flat_full = np.array([el.flatten() for el in c_full])
        labels_full = ['bin_'+str(el) for el in np.arange(flat_full.shape[1])]

        flat_count = np.array([el.flatten() for el in c_count]).T
        labels_count = ['count_' + str(el) for el in np.arange(flat_count.shape[1])]

        flat_edges = np.array([el.flatten() for el in c_edges]).T
        labels_edges = ['edge_' + str(el) for el in np.arange(flat_edges.shape[1])]
        # turn everything into dataframes
        df_half = pd.DataFrame(np.hstack(flat_half), columns=np.hstack(labels_half), dtype=np.float32)
        df_full = pd.DataFrame(flat_full, columns=labels_full, dtype=np.float32)
        df_resp = pd.DataFrame(c_resp, columns=['Resp_index', 'Resp_test', 'Qual_index', 'Qual_test'], dtype=np.float32)
        df_cons = pd.DataFrame(c_cons, columns=['Cons_index', 'Cons_test'], dtype=np.float32)
        # concatenate
        df_concat = pd.concat((df_half, df_full, df_resp, df_cons), axis=1)
        # generate columns for date and animal
        df_concat['day'] = date
        df_concat['animal'] = mouse
        df_concat['rig'] = setup
        # store
        out_dict[feat] = df_concat
        # store the counts
        df_count = pd.DataFrame(flat_count, columns=labels_count, dtype=np.float32)
        df_count['day'] = date
        df_count['animal'] = mouse
        df_count['rig'] = setup
        count_dict[feat] = df_count

        # store the edges
        df_edges = pd.DataFrame(flat_edges, columns=labels_edges, dtype=np.float32)
        df_edges['day'] = date
        df_edges['animal'] = mouse
        df_edges['rig'] = setup
        edges_dict[feat] = df_edges

    return out_dict, count_dict, edges_dict


if __name__ == '__main__':
    # get the data paths
    try:
        input_path = snakemake.input
        # get the slugs
        slug_list = [os.path.basename(el).replace('_preproc.hdf5', '') for el in input_path]
        # read the output path and the input file urls
        out_path = snakemake.output[0]
        data_all = snakemake.params.file_info
        data_all = [yaml.load((data_all[el]), Loader=yaml.FullLoader) for el in slug_list]
        # get the parts for the file naming
        name_parts = os.path.basename(out_path).split('_')
        day = '_'.join(name_parts[:3])
        animal = '_'.join(name_parts[3:6])
        rig = name_parts[6]

    except NameError:
        # get the search query
        search_string = processing_parameters.search_string

        # get the paths from the database
        data_all = bd.query_database('analyzed_data', search_string)
        data_all = [el for el in data_all if '_preproc' in el['slug']]
        input_path = [el['analysis_path'] for el in data_all if '_preproc' in el['slug']]
        # get the day, animal and rig
        day = '_'.join(data_all[0]['slug'].split('_')[0:3])
        rig = data_all[0]['rig']
        animal = data_all[0]['slug'].split('_')[7:10]
        animal = '_'.join([animal[0].upper()] + animal[1:])

        # assemble the output path
        out_path = os.path.join(paths.analysis_path, '_'.join((day, animal, rig, 'tcday.hdf5')))

    # allocate memory for the data
    raw_data = []
    # allocate memory for the meta data
    meta_list = []
    # for all the files
    for idx0, files in enumerate(input_path):

        # get the metadata
        # meta_list.append([data_all[idx0][el1] for el1 in processing_parameters.meta_fields])

        # load the data
        with pd.HDFStore(files, mode='r') as h:
            if '/matched_calcium' in h.keys():

                # concatenate the latents
                dataframe = h['matched_calcium']

                # check for latents
                if '/latents' in h.keys():
                    # get the latents and motifs
                    latents = h['latents']
                    motifs = h['motifs']
                    egocentric_coords = h['egocentric_coord']
                    egocentric_coords = egocentric_coords.loc[:, ['cricket_0_x', 'cricket_0_y']]
                    egocentric_coords = egocentric_coords.rename(columns={'cricket_0_x': 'ego_cricket_x',
                                                                          'cricket_0_y': 'ego_cricket_y'})
                    # determine the delta size for padding
                    delta_frames = dataframe.shape[0] - latents.shape[0]
                    # pad latents due to the VAME calculation window
                    latent_padding = pd.DataFrame(np.zeros((int(delta_frames/2), len(latents.columns))) * np.nan,
                                                  columns=latents.columns)
                    motif_padding = pd.DataFrame(np.zeros((int(delta_frames/2), len(motifs.columns))) * np.nan,
                                                 columns=motifs.columns)
                    # pad them with nan at the edges (due to VAME excluding the edges
                    latents = pd.concat([latent_padding, latents, latent_padding], axis=0).reset_index(drop=True)
                    motifs = pd.concat([motif_padding, motifs, motif_padding], axis=0).reset_index(drop=True)
                    # concatenate with the main data
                    dataframe = pd.concat([dataframe, egocentric_coords, latents, motifs], axis=1)

                fluor_cols = [col for col in dataframe.columns if 'fluor' in col]
                dataframe.drop(columns=fluor_cols, inplace=True)
                # store
                raw_data.append((files, dataframe))

    # skip processing if the file is empty
    if len(raw_data) == 0:
        # save an empty file and end
        empty = pd.DataFrame([])
        empty.to_hdf(out_path, 'no_ROIs')
    else:
        # get the number of bins
        bin_num = processing_parameters.bin_number
        # define the pairs to quantify
        variable_names = processing_parameters.variable_list

        # clip the calcium traces
        clipped_data = clip_calcium(raw_data)

        # parse the features (bin number is for spatial bins in this one)
        features, calcium = parse_features(clipped_data, variable_names, bin_number=20)

        # concatenate all the trials
        features = pd.concat(features)
        calcium = np.concatenate(calcium)

        # get the number of cells
        cell_num = calcium.shape[1]

        # get the TCs and their responsivity
        tcs_half, tcs_full, tcs_resp, tc_count, tc_bins = extract_tcs_responsivity(features, calcium, variable_names,
                                                                                   cell_num, percentile=80,
                                                                                   bin_number=bin_num)
        # get the TC consistency
        tcs_cons = extract_consistency(tcs_half, variable_names, cell_num, percentile=80)
        # # get the tc quality
        # tcs_qual = extract_quality(tcs_full, features)
        # convert the outputs into a dataframe
        tcs_dict, tcs_counts_dict, _ = convert_to_dataframe(tcs_half, tcs_full, tc_count, tcs_resp,
                                                            tcs_cons, tc_bins, day, animal, rig)

        # for all the features
        for feature in tcs_dict.keys():
            tcs_dict[feature].to_hdf(out_path, feature)
            tcs_counts_dict[feature].to_hdf(out_path, feature+'_counts')
            # tcs_bins_dict[feature].to_hdf(out_path, feature + '_edges')
        # save the meta data
        # meta_data = pd.DataFrame(np.vstack(meta_list), columns=processing_parameters.meta_fields)
        # meta_data.to_hdf(out_path, 'meta_data')
        # save as a new entry to the data base
        # assemble the entry data
        entry_data = {
            'analysis_type': 'tc_analysis',
            'analysis_path': out_path,
            'date': '',
            'pic_path': '',
            'result': 'multi',
            'rig': rig,
            'lighting': 'multi',
            'imaging': 'multi',
            'slug': fm.slugify(os.path.basename(out_path)[:-5]),

        }

        # check if the entry already exists, if so, update it, otherwise, create it
        update_url = '/'.join((paths.bondjango_url, 'analyzed_data', entry_data['slug'], ''))
        output_entry = bd.update_entry(update_url, entry_data)
        if output_entry.status_code == 404:
            # build the url for creating an entry
            create_url = '/'.join((paths.bondjango_url, 'analyzed_data', ''))
            output_entry = bd.create_entry(create_url, entry_data)

        print('The output status was %i, reason %s' %
              (output_entry.status_code, output_entry.reason))
        if output_entry.status_code in [500, 400]:
            print(entry_data)
