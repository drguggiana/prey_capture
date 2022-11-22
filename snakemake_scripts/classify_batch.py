# imports

import paths
import functions_bondjango as bd
import os
import yaml
import processing_parameters

import numpy as np
import pandas as pd
from sklearn import preprocessing
import sklearn.linear_model as lin
import sklearn.svm as svm
import sklearn.model_selection as mod
import h5py
import functions_misc as fm
import random
import scipy.stats as stat
import functions_loaders as fl

# set the random seed
random.seed(1)


def reverse_roll_shuffle(feature, **kwargs):
    """Shuffle the data by inverting and randomly circshifting"""
    # flip the array
    output_array = np.flip(feature)
    # get the random amount of shifting
    random_shift = np.random.choice(feature.shape[0], 1)
    # shift the array
    output_array = np.roll(output_array, random_shift)
    return output_array


def chunk_shuffle(feature, chunk_size_shuffle=0.05, **kwargs):
    """Shuffle the input array in chunks of a given size"""
    # get the size of the chunks
    chunk_length = int(np.floor(chunk_size_shuffle * feature.shape[0]))
    # get the list of indexes
    chunk_list = np.arange(0, feature.shape[0], chunk_length)

    # if there's a short chunk, make it part of the last one
    if (chunk_list[-1] + chunk_length) > feature.shape[0]:
        chunk_list = chunk_list[:-1]
    # get the test chunks
    chunk_list = np.random.choice(chunk_list, len(chunk_list), replace=False)
    # allocate memory for the output
    shuffled_feature = []
    # for all the chunks
    for chunk_start in chunk_list:
        if chunk_start == np.max(chunk_list):
            chunk_stop = feature.shape[0]
        else:
            # get the stop
            chunk_stop = chunk_start + chunk_length
        # select the corresponding stretches
        shuffled_feature.append(feature[chunk_start:chunk_stop])
    # concatenate
    shuffled_feature = np.hstack(shuffled_feature)

    # output the concatenated shuffled feature
    return shuffled_feature


def chunk_split(calcium, feature, test_size=0.2, chunk_size=0.05, **kwargs):
    """Split the data into train and test sets based on shuffled chunks of a given size in fraction of dataset"""
    # determine the number of and size of the chunks
    chunk_number = int(np.round(test_size/chunk_size))
    chunk_length = int(np.floor(chunk_size * feature.shape[0]))

    # get the list of indexes
    chunk_list = np.arange(0, feature.shape[0], chunk_length)
    # if there's a short chunk, make it part of the last one
    if (chunk_list[-1] + chunk_length) > feature.shape[0]:
        chunk_list = chunk_list[:-1]
    # get the test chunks
    test_chunks = np.random.choice(chunk_list, chunk_number, replace=False)

    # allocate memory for the outputs
    c_train = []
    c_test = []
    p_train = []
    p_test = []
    # for all the chunks
    for chunk_start in chunk_list:
        # get the stop
        if chunk_start == np.max(chunk_list):
            chunk_stop = feature.shape[0]
        else:
            chunk_stop = chunk_start + chunk_length
        # select the corresponding stretches
        if chunk_start in test_chunks:
            c_test.append(calcium[chunk_start:chunk_stop, :])
            p_test.append(feature[chunk_start:chunk_stop])
        else:
            c_train.append(calcium[chunk_start:chunk_stop, :])
            p_train.append(feature[chunk_start:chunk_stop])

    # convert the lists to arrays
    c_train = np.vstack(c_train)
    c_test = np.vstack(c_test)
    p_train = np.hstack(p_train)
    p_test = np.hstack(p_test)

    return c_train, c_test, p_train, p_test


def train_test_regressor(parameter_w, calcium_data_w, scaler_fun, regressor_fun, performance_fun,
                         time_s=0, shuffle_f=False, empty=False, chunk=True, **kwargs):
    """Train a regressor using the given preprocessing and regression functions"""
    # trim the parameter and calcium traces according to the time shift
    if time_s > 0:
        parameter_w = parameter_w[time_s:]
        calcium_data_w = calcium_data_w[:-time_s, :]
    elif time_s < 0:
        parameter_w = parameter_w[:time_s]
        calcium_data_w = calcium_data_w[-time_s:, :]

    # if the empty flag is active, output nan-filled arrays of the correct dimensions
    if empty:
        coefficients_out = np.zeros(calcium_data_w.shape[1]) * np.nan
        linear_pred_out = np.zeros(calcium_data_w.shape[0]) * np.nan
        cc_score_out = np.nan
        return linear_pred_out, coefficients_out, cc_score_out

    # leave only points without nans
    nan_vector = np.isnan(parameter_w) == 0
    parameter_w = parameter_w[nan_vector]
    calcium_data_w = calcium_data_w[nan_vector, :]

    # if shuffling is on, shuffle the parameter in time
    if shuffle_f:
        # shuffle using chunks like for partitioning
        # parameter_w = chunk_shuffle(parameter_w, **kwargs)
        parameter_w = reverse_roll_shuffle(parameter_w, **kwargs)
    # if the chunk flag it on, perform chunk splitting
    if chunk:
        calcium_train, calcium_test, parameter_train, parameter_test = \
            chunk_split(calcium_data_w, parameter_w, **kwargs)
    # else, perform sequential splitting (not fully random cause calcium)
    else:
        # remove potential parameters that shouldn't be there
        remove_params = ['chunk_size', 'chunk_size_shuffle']
        for el in remove_params:
            if el in kwargs.keys():
                del kwargs[el]
        # separate train and test set (without shuffling)
        calcium_train, calcium_test, parameter_train, parameter_test = \
            mod.train_test_split(calcium_data_w, parameter_w, **kwargs)

    # scale the calcium data
    if scaler_fun is not None:
        calcium_scaler = scaler_fun().fit(calcium_train)
        calcium_train = calcium_scaler.transform(calcium_train)
        calcium_test = calcium_scaler.transform(calcium_test)
        calcium_data_w = calcium_scaler.transform(calcium_data_w)
    # train the regressor
    # if type(regressor_fun) == type:
    regressor_fun.fit(calcium_train, parameter_train)
    # predict the test and the whole data
    test_pred = regressor_fun.predict(calcium_test)
    linear_pred_out = regressor_fun.predict(calcium_data_w)
    # else:
    #     regressor = regressor_fun(parameter_train, exog=calcium_train, lags=5, missing='drop').fit()

    # get the cc score and coefficients
    cc_score_out = performance_fun(parameter_test, test_pred)
    # check if tuple, to use also scipy functions
    if isinstance(cc_score_out, tuple):
        cc_score_out = cc_score_out[0]
    # get the weights or output nan if nonlinear
    try:
        coefficients_out = regressor_fun.coef_
    except AttributeError:
        coefficients_out = np.zeros(calcium_data_w.shape[1]) * np.nan

    return linear_pred_out, coefficients_out, cc_score_out


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
        name_parts = out_path.split('_')
        day = name_parts[0]
        animal = name_parts[1]
        rig = name_parts[2]

    except NameError:
        # get the search string
        animal = processing_parameters.animal
        day = processing_parameters.day
        rig = processing_parameters.rig
        search_string = 'imaging:doric, slug:%s' % day

        # query the database for data to plot
        data_all = bd.query_database('analyzed_data', search_string)

        input_path = [el['analysis_path'] for el in data_all if ('preproc' in el['slug'] and animal.lower() in el['slug'])]

        # assemble the output path
        out_path = os.path.join(paths.analysis_path, '_'.join((day, animal, rig, 'regressionday.hdf5')))

    # load the data
    data_list = []
    meta_list = []
    frame_list = []
    for idx, el in enumerate(input_path):
        # get the trial timestamp (for frame calculations)
        time_stamp = int(''.join(os.path.basename(el).split('_')[3:6]))

        try:
            temp_data = pd.read_hdf(el, 'matched_calcium')
            # temp_data['trial_id'] = time_signature
            temp_data['id'] = data_all[idx]['id']

            meta_list.append([data_all[idx][el1] for el1 in processing_parameters.meta_fields])
            # try to load the motifs and latents
            try:
                latents = pd.read_hdf(el, 'latents')
                motifs = pd.read_hdf(el, 'motifs')
                egocentric_coords = pd.read_hdf(el, 'egocentric_coord')
                egocentric_coords = egocentric_coords.loc[:, ['cricket_0_x', 'cricket_0_y']]
                egocentric_coords = egocentric_coords.rename(columns={'cricket_0_x': 'ego_cricket_x',
                                                                      'cricket_0_y': 'ego_cricket_y'})

                # pad them with nan at the edges (due to VAME excluding the edges
                [latents, motifs] = fl.pad_latents([latents, motifs], temp_data.shape[0])
                # concatenate with the main data
                temp_data = pd.concat([temp_data, egocentric_coords, latents, motifs], axis=1)
            except KeyError:
                print(f'No latents in file {el}')
            data_list.append(temp_data)
            frame_list.append([time_stamp, 0, temp_data.shape[0]])
        except KeyError:
            # data_list.append([])
            frame_list.append([time_stamp, 0, 0])

    # if none of the files has calcium, exclude
    if len(data_list) == 0:
        # save in the target file
        with h5py.File(out_path, 'w') as f:
            f.create_dataset('no_ROIs', data=[])
    else:

        # turn the frame list into an np array
        frame_list = np.array(frame_list)
        # make the frame numbers cumulative to read them as limits
        frame_list[:, 2] = np.cumsum(frame_list[:, 2])
        frame_list[1:, 1] = frame_list[:-1, 2]
        # save in the target file
        with h5py.File(out_path, 'w') as f:
            f.create_dataset('frame_list', data=frame_list)

        # Regression of behavioral variables

        # define the target variables
        variable_list = processing_parameters.variable_list
        time_shifts = processing_parameters.time_shifts
        # get the cells
        labels = list(np.unique(np.array([el.columns for el in data_list]).flatten()))
        cells = [el for el in labels if 'cell' in el]
        # for all the entries
        for idx, target_behavior in enumerate(variable_list):

            print(f'Current variable: {target_behavior}')

            # check if the current variable is in the file, otherwise, fill with nan and go to the next iteration
            if target_behavior not in data_list[0].columns:
                sub_data = [el[cells] for el in data_list]
                sub_data = pd.concat(sub_data)
                parameter = np.zeros((sub_data.shape[0], 1)) * np.nan
                empty_flag = True

            else:
                # get the relevant columns
                sub_data = [el[[target_behavior]+cells] for el in data_list if (target_behavior in el.columns)]
                sub_data = pd.concat(sub_data)
                # get the parameter of interest
                parameter = sub_data.loc[:, target_behavior].to_numpy()
                empty_flag = False

            # get the cell data
            calcium_data = np.array(sub_data[cells].copy())

            # for all the regressors
            for reg in processing_parameters.regressors:

                # for all the time shifts
                for time_shift in time_shifts:
                    # copy the calcium and parameter for working
                    parameter_working = parameter.copy()
                    calcium_data_working = calcium_data.copy()

                    # run the real and shuffle regressions
                    for realvshuffle in np.arange(2):
                        # initialize the suffix (so I don't get the warning below)
                        suffix = reg
                        if realvshuffle == 1:
                            suffix += '_shuffle'
                            shuffle_flag = True
                        else:
                            suffix += '_real'
                            shuffle_flag = False
                        # allocate lists for each rep
                        pred_list = []
                        coeff_list = []
                        cc_list = []
                        # run the reps
                        for shuffler in np.arange(processing_parameters.regression_repeats):

                            # create the linear regressor
                            if reg == 'linear':
                                regressor = lin.TweedieRegressor(alpha=0.01, max_iter=5000, fit_intercept=False, power=0)
                            elif reg == 'SVR':
                                regressor = svm.SVR(max_iter=10000, kernel='rbf', C=100)
                            else:
                                raise ValueError('Unrecognized regressor option')

                            # run the training function
                            linear_pred, coefficients, cc_score = train_test_regressor(parameter_working,
                                                                                       calcium_data_working,
                                                                                       preprocessing.StandardScaler,
                                                                                       regressor,
                                                                                       stat.spearmanr,
                                                                                       time_s=time_shift,
                                                                                       shuffle_f=shuffle_flag,
                                                                                       empty=empty_flag,
                                                                                       chunk=True,
                                                                                       test_size=0.3,
                                                                                       chunk_size=0.05,
                                                                                       chunk_size_shuffle=0.05)
                            # store the reps
                            pred_list.append(linear_pred)
                            coeff_list.append(coefficients/np.sum(coefficients))
                            cc_list.append(cc_score)
                        # add the shift to the suffix
                        suffix += '_shift'+str(time_shift)

                        # average/std the lists and store
                        mean_coefficients = np.sum(coeff_list, axis=0)
                        mean_linear_pred = np.mean(pred_list, axis=0)
                        mean_cc_score = np.mean(cc_list, axis=0)
                        std_coefficients = np.std(coeff_list, axis=0)
                        std_linear_pred = np.std(pred_list, axis=0)
                        std_cc_score = np.std(cc_list, axis=0)
                        # save the file
                        with h5py.File(out_path, 'a') as f:
                            # save the results
                            f.create_dataset('_'.join(['coefficients', target_behavior, suffix]),
                                             data=mean_coefficients)
                            f.create_dataset('_'.join(['prediction', target_behavior, suffix]),
                                             data=mean_linear_pred)
                            f.create_dataset('_'.join(['cc', target_behavior, suffix]),
                                             data=mean_cc_score)
                            f.create_dataset('_'.join(['coefficients', target_behavior, suffix+'_std']),
                                             data=std_coefficients)
                            f.create_dataset('_'.join(['prediction', target_behavior, suffix+'_std']),
                                             data=std_linear_pred)
                            f.create_dataset('_'.join(['cc', target_behavior, suffix+'_std']),
                                             data=std_cc_score)

        # assemble the metadata frame
        dt = h5py.special_dtype(vlen=str)
        meta_data = np.vstack((processing_parameters.meta_fields, np.vstack(meta_list))).astype(dt)
        # get the trial ID data
        trial_id = np.hstack([el['id'].to_numpy() for el in data_list])
        # save the meta data and trial id
        with h5py.File(out_path, 'a') as f:
            f.create_dataset('meta_data', data=meta_data)
            f.create_dataset('id', data=trial_id)

        # save as a new entry to the data base
        # assemble the entry data
        entry_data = {
            'analysis_type': 'regression_day',
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
