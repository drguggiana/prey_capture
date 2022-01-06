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
import h5py


def flatten_tc():
    return


def clip_calcium(pre_data):
    """ Clip the calcium traces based on baseline """

    # allocate memory for the cleaned up data
    data = []

    # for all the trials
    for idx, el in enumerate(pre_data):
        # # get the date
        # current_date = os.path.basename(el[0])[:10]

        # get the current df
        current_df = el[1]
        labels = list(current_df.columns)
        cells = [el for el in labels if 'cell' in el]
        not_cells = [el for el in labels if 'cell' not in el]
        # get the non-cell data
        non_cell_data = current_df[not_cells]
        # get the current calcium data
        cell_data = current_df[cells]

        # calculate a baseline for all cells
        for name, single in cell_data.items():
            # skip if there are only zeros
            if np.sum(single) == 0:
                continue
            # get the baseline
            baseline = np.percentile(single[single > 0], 8)

            # clip the trace
            single[single < baseline] = 0
            # store
            cell_data[name] = single

        # # remove the nans after normalization
        # cell_data[np.isnan(cell_data)] = 0
        # assemble a new data frame with only the matched cells and the rest of the data
        data.append(pd.concat((non_cell_data, cell_data), axis=1))
        return data


def parse_features(data):
    """set up the feature and calcium matrices"""

    # define the design matrix
    feature_list = ['mouse_speed', 'cricket_0_speed', 'mouse_x', 'mouse_y', 'cricket_0_x', 'cricket_0_y',
                    'cricket_0_delta_heading', 'cricket_0_mouse_distance', 'cricket_0_visual_angle',
                    'mouse_heading', 'cricket_0_delta_head', 'cricket_0_heading', 'head_direction',
                    'latent_0', 'latent_1', 'latent_2', 'latent_3', 'latent_4',
                    'latent_5', 'latent_6', 'latent_7', 'latent_8', 'latent_9']

    # allocate memory for a data frame without the encoding model features
    feature_raw_trials = []
    # allocate memory for the calcium
    calcium_trials = []

    # get the features
    for idx, el in enumerate(data):
        # get the intersection of the labels
        label_intersect = [feat for feat in feature_list if feat in el.columns]

        if len(label_intersect) != len(feature_list):
            continue
        # get the features of interest
        target_features = el.loc[:, feature_list]
        # get the original columns
        original_columns = target_features.columns

        # turn the radial variables into linear ones
        # for all the columns
        for label in original_columns:
            # calculate head speed
            if label == 'head_direction':
                # get the head direction
                head = target_features[label].copy().to_numpy()
                # get the angular speed and acceleration of the head
                speed = np.concatenate(([0], np.diff(ss.medfilt(head, 21))), axis=0)
                acceleration = np.concatenate(([0], np.diff(head)), axis=0)
                # add to the features
                target_features['head_speed'] = speed
                target_features['head_acceleration'] = acceleration

            # check if the label is a speed and calculate acceleration
            if 'speed' in label:
                # get the speed
                speed = target_features[label].copy().to_numpy()
                # calculate the acceleration with the smoothed speed
                acceleration = np.concatenate(([0], np.diff(ss.medfilt(speed, 21))), axis=0)
                # add to the features
                target_features[label.replace('speed', 'acceleration')] = acceleration
            # smooth the feature
            target_features[label] = ss.medfilt(target_features[label], 21)

        # store the features
        feature_raw_trials.append(target_features)

        # get the calcium data
        cells = [cell for cell in el.columns if 'cell' in cell]
        cells = el.loc[:, cells].to_numpy()

        # store
        calcium_trials.append(cells)

        return feature_raw_trials, calcium_trials


def extract_tcs_responsivity(feature_raw_trials, calcium_trials, target_pairs):
    """Extract the tuning curves (full and half) and their responsivity index"""

    # get the number of pairs
    pair_number = len(target_pairs)
    # define the number of calcium shuffles
    shuffle_number = 100
    # define the confidence interval cutoff
    percentile = 95
    # define the number of bins for the TCs
    bin_number = 10

    # get the number of cells
    cell_number = calcium_trials.shape[1]
    # allocate memory for the trial TCs
    tc_half = {}
    tc_full = {}
    tc_resp = {}
    # for all the features
    for pair_idx in np.arange(pair_number):
        # get the current feature
        feature_name = target_pairs[pair_idx]
        feature_names = feature_name.split('|')
        current_feature_0 = feature_raw_trials.loc[:, feature_names[0]].to_numpy()
        current_feature_1 = feature_raw_trials.loc[:, feature_names[1]].to_numpy()
        # get the bins from the parameters file
        bin_ranges = processing_parameters.tc_params[feature_name]
        # calculate the bin edges based on the ranges
        bins = [np.linspace(el[0], el[1], num=bin_number + 1) for el in bin_ranges]
        # allocate a list for the 2 halves
        tc_half_temp = []
        # exclude nan values
        keep_vector_full = (~np.isnan(current_feature_0)) & (~np.isnan(current_feature_1))
        counts_feature_0 = current_feature_0[keep_vector_full]
        counts_feature_1 = current_feature_1[keep_vector_full]
        # get the counts
        feature_counts = stat.binned_statistic_2d(counts_feature_0, counts_feature_1, counts_feature_0,
                                                  statistic='count', bins=bins)[0]
        # zero the positions with less than 3 counts
        feature_counts[feature_counts < 3] = 0
        # for first and second half
        for half in np.arange(2):
            # get the half vector
            half_bound = int(np.floor(current_feature_0.shape[0] / 2))
            half_vector = np.arange(half_bound) + half_bound * half
            half_feature_0 = current_feature_0[half_vector]
            half_feature_1 = current_feature_1[half_vector]
            # exclude nan values
            keep_vector = (~np.isnan(half_feature_0)) & (~np.isnan(half_feature_1))
            keep_feature_0 = half_feature_0[keep_vector]
            keep_feature_1 = half_feature_1[keep_vector]

            # allocate a list for the cells
            tc_cell = []
            # for all the cells
            for cell in np.arange(cell_number):
                # get the current cell
                half_cell = calcium_trials[half_vector, cell]
                keep_cell = half_cell[keep_vector]

                # calculate the TC
                current_tc = stat.binned_statistic_2d(keep_feature_0, keep_feature_1, keep_cell,
                                                      statistic='sum', bins=bins)[0]
                # normalize the TC
                norm_tc = current_tc / feature_counts
                # remove nans and infs
                norm_tc[np.isnan(norm_tc)] = 0
                norm_tc[np.isinf(norm_tc)] = 0
                # store
                tc_cell.append(norm_tc)
            # store the cells
            tc_half_temp.append(tc_cell)
        # allocate memory for the full tc per cell
        tc_cell_full = []
        tc_cell_resp = np.zeros((cell_number, 2))
        # calculate the full TC
        for cell in np.arange(cell_number):
            keep_cell = calcium_trials[keep_vector_full, cell]
            tc_cell = stat.binned_statistic_2d(counts_feature_0, counts_feature_1,
                                               keep_cell, statistic='sum', bins=bins)[0]
            tc_cell = tc_cell / feature_counts
            tc_cell[np.isnan(tc_cell)] = 0
            tc_cell[np.isinf(tc_cell)] = 0
            # allocate memory for the shuffles
            shuffle_array = np.zeros((shuffle_number, bin_number, bin_number))
            # generate the shuffles
            for shuffle in np.arange(shuffle_number):
                # randomize the calcium activity
                random_cell = keep_cell.copy()
                np.random.shuffle(random_cell)
                tc_random = stat.binned_statistic_2d(counts_feature_0, counts_feature_1,
                                                     random_cell, statistic='sum', bins=bins)[0]
                tc_random = tc_random / feature_counts
                tc_random[np.isnan(tc_random)] = 0
                tc_random[np.isinf(tc_random)] = 0
                shuffle_array[shuffle, :, :] = tc_random
            # get the threshold
            resp_threshold = np.percentile(np.abs(shuffle_array.flatten()), percentile)
            # fill up the responsivity matrix
            tc_cell_resp[cell, 0] = np.mean(np.sort(np.abs(tc_cell), axis=None)[-3:]) / resp_threshold
            tc_cell_resp[cell, 1] = np.sum(np.abs(tc_cell) > resp_threshold) > 3
            # store
            tc_cell_full.append(tc_cell)
        # store the halves and fulls
        tc_half[feature_name] = tc_half_temp
        tc_full[feature_name] = tc_cell_full
        tc_resp[feature_name] = tc_cell_resp

        return tc_half, tc_full, tc_resp


def extract_consistency(tc_half, target_pairs):
    """Calculate TC consistency"""

    # define the number of shuffles
    shuffle_number = 100
    # define the percentile
    percentile = 95

    # get the number of cells
    cell_number = len(tc_half)
    # get the number of pairs
    pair_number = len(target_pairs)

    # allocate memory for the trial TCs
    tc_cons = {}
    # for all the features
    for pair_idx in np.arange(pair_number):

        # get the name
        feature_name = target_pairs[pair_idx]
        # allocate an array for the correlations and tests
        tc_half_temp = np.zeros([cell_number, 2])
        # get the two halves
        halves = tc_half[feature_name]

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
                np.random.shuffle(random_second)
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


# get the data paths
try:
    input_path = snakemake.input
    # read the output path and the input file urls
    out_path = snakemake.output[0]
    # data_all = snakemake.params.file_info
    # get the parts for the file naming
    name_parts = out_path.split('_')
    day = name_parts[0]
    animal = name_parts[1]
    rig = name_parts[2]
except NameError:
    # get the search query
    search_string = processing_parameters.search_string

    # get the paths from the database
    all_path = bd.query_database('analyzed_data', search_string)
    input_path = [el['analysis_path'] for el in all_path if '_preproc' in el['slug']]
    # get the day, animal and rig
    day = '_'.join(all_path[0]['slug'].split('_')[0:3])
    rig = all_path[0]['rig']
    animal = '_'.join(all_path[0]['slug'].split('_')[3:6])

    # assemble the output path
    out_path = os.path.join(paths.analysis_path, '_'.join((day, animal, rig, 'tcday.hdf5')))

# allocate memory for the data
raw_data = []
# for all the files
for files in input_path:
    # load the data
    with pd.HDFStore(files) as h:
        if '/matched_calcium' in h.keys():

            # concatenate the latents
            dataframe = pd.concat([h['matched_calcium'], h['latents']], axis=1)
            # store
            raw_data.append((files, dataframe))

# skip processing if the file is empty
if len(raw_data) == 0:
    # save an empty file and end
    with h5py.File(out_path, 'w') as f:
        f.create_dataset('no_ROIs', data=[])
else:

    # clip the calcium traces
    clipped_data = clip_calcium(raw_data)

    # parse the features
    features, calcium = parse_features(clipped_data)

    # concatenate all the trials
    features = pd.concat(features)
    calcium = np.concatenate(calcium)

    # define the pairs to quantify
    variable_pairs = list(processing_parameters.tc_params.keys())

    # get the TCs and their responsivity
    tcs_half, tcs_full, tcs_resp = extract_tcs_responsivity(features, calcium, variable_pairs)
    # get the TC consistency
    tcs_cons = extract_consistency(tcs_half, variable_pairs)

    # save the data to an h5py file
    with h5py.File(out_path, 'w') as f:

        # for all the features
        for feature in tcs_half.keys():
            group = f.create_group(feature)
            group['half_tcs'] = np.array(tcs_half[feature]).astype(np.float32)
            group['full_tcs'] = np.array(tcs_full[feature]).astype(np.float32)
            group['responsivity'] = np.array(tcs_resp[feature]).astype(np.float32)
            group['consistency'] = np.array(tcs_cons[feature]).astype(np.float32)

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