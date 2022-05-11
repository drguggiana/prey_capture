# imports

import paths
import functions_bondjango as bd
import os
import yaml
import processing_parameters
import datetime

import numpy as np
import pandas as pd
from sklearn import preprocessing
import sklearn.metrics as smet
import sklearn.linear_model as lin
import sklearn.model_selection as mod
import h5py
import functions_misc as fm
import random
import scipy.stats as stat


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
    # also get the trial time signature, used as unique ID
    # time_signature = int(''.join(os.path.basename(el).split('_')[0:6]))

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
            # determine the delta size for padding
            delta_frames = temp_data.shape[0] - latents.shape[0]
            # pad latents due to the VAME calculation window
            latent_padding = pd.DataFrame(np.zeros((int(delta_frames / 2), len(latents.columns))) * np.nan,
                                          columns=latents.columns)
            motif_padding = pd.DataFrame(np.zeros((int(delta_frames / 2), len(motifs.columns))) * np.nan,
                                         columns=motifs.columns)
            # pad them with nan at the edges (due to VAME excluding the edges
            latents = pd.concat([latent_padding, latents, latent_padding], axis=0).reset_index(drop=True)
            motifs = pd.concat([motif_padding, motifs, motif_padding], axis=0).reset_index(drop=True)
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

            # for all the time shifts
            for time_shift in time_shifts:
                # get the "time shifted" length so it's compatible with the real data
                calcium_data_working = sub_data.to_numpy().copy()

                # trim the parameter and calcium traces according to the time shift
                if time_shift > 0:
                    calcium_data_working = calcium_data_working[:-time_shift, :]
                elif time_shift < 0:
                    calcium_data_working = calcium_data_working[-time_shift:, :]
                # for a real and a shuffle version
                for shuffler in np.arange(processing_parameters.regression_shuffles + 1):

                    if shuffler > 0:
                        suffix = 'shuffle' + str(shuffler)
                    else:
                        suffix = 'real'
                    # add the shift to the suffix
                    suffix += '_shift' + str(time_shift)

                    # create the nan versions of the output
                    coefficients = np.zeros(sub_data.shape[1]) * np.nan
                    linear_pred = np.zeros(sub_data.shape[0]) * np.nan
                    cc_score = np.nan
                    # save the file
                    with h5py.File(out_path, 'a') as f:
                        # save an empty
                        f.create_dataset('_'.join(['coefficients', target_behavior, suffix]),
                                         data=coefficients)
                        f.create_dataset('_'.join(['prediction', target_behavior, suffix]),
                                         data=linear_pred)
                        f.create_dataset('_'.join(['cc', target_behavior, suffix]),
                                         data=cc_score)
            continue

        # get the relevant columns
        sub_data = [el[[target_behavior]+cells] for el in data_list if (target_behavior in el.columns)]
        sub_data = pd.concat(sub_data)

        # get the cell data
        calcium_data = np.array(sub_data[cells].copy())
        # get the parameter of interest
        parameter = sub_data.loc[:, target_behavior].to_numpy()

        # for all the time shifts
        for time_shift in time_shifts:
            # copy the calcium and parameter for working
            parameter_working = parameter.copy()
            calcium_data_working = calcium_data.copy()

            # trim the parameter and calcium traces according to the time shift
            if time_shift > 0:
                parameter_working = parameter_working[time_shift:]
                calcium_data_working = calcium_data_working[:-time_shift, :]
            elif time_shift < 0:
                parameter_working = parameter_working[:time_shift]
                calcium_data_working = calcium_data_working[-time_shift:, :]

            # exclude points without nans
            nan_vector = np.isnan(parameter_working) == 0
            parameter_working = parameter_working[nan_vector]
            calcium_data_working = calcium_data_working[nan_vector, :]

            # for a real and a shuffle version
            for shuffler in np.arange(processing_parameters.regression_shuffles+1):

                if shuffler > 0:
                    suffix = 'shuffle'+str(shuffler)
                    random.shuffle(parameter_working)
                else:
                    suffix = 'real'

                # add the shift to the suffix
                suffix += '_shift'+str(time_shift)
                # separate train and test set (without shuffling)
                calcium_train, calcium_test, parameter_train, parameter_test = \
                    mod.train_test_split(calcium_data_working, parameter_working, test_size=0.2, shuffle=False)
                # scale the calcium data
                calcium_scaler = preprocessing.StandardScaler().fit(calcium_train)
                calcium_train = calcium_scaler.transform(calcium_train)
                calcium_test = calcium_scaler.transform(calcium_test)
                calcium_data_working = calcium_scaler.transform(calcium_data_working)
                # create the linear regressor
                linear = lin.TweedieRegressor(alpha=0.01, max_iter=5000, fit_intercept=False, power=0)
                # train the regressor
                linear.fit(calcium_train, parameter_train)
                # predict the test and the whole data
                test_pred = linear.predict(calcium_test)
                linear_pred = linear.predict(calcium_data_working)

                # get the cc score and coefficients
                cc_score = stat.spearmanr(parameter_test, test_pred)[0]
                coefficients = linear.coef_

                # save the file
                with h5py.File(out_path, 'a') as f:
                    # save an empty
                    f.create_dataset('_'.join(['coefficients', target_behavior, suffix]),
                                     data=coefficients)
                    f.create_dataset('_'.join(['prediction', target_behavior, suffix]),
                                     data=linear_pred)
                    f.create_dataset('_'.join(['cc', target_behavior, suffix]),
                                     data=cc_score)

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
