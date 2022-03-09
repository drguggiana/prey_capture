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
    # get the search string
    search_string = processing_parameters.search_string_calcium
    animal = processing_parameters.animal
    day = processing_parameters.day
    rig = processing_parameters.rig
    # query the database for data to plot
    data_all = bd.query_database('analyzed_data', search_string)
    # video_data = data_all[0]
    # video_path = video_data['tif_path']
    input_path = [el['analysis_path'] for el in data_all]
    # # overwrite data_all with just the urls
    # video_urls = {os.path.basename(el['analysis_path'])[:-4]: el['video_analysis'] for el in data_all}
    # vr_urls = {os.path.basename(el['analysis_path'])[:-4]: el['vr_analysis'] for el in data_all}
    # data_all = {os.path.basename(el['analysis_path'])[:-4]: el['url'] for el in data_all}
    # assemble the output path
    out_path = os.path.join(paths.analysis_path, '_'.join((day, animal, rig, 'regressionday.hdf5')))

# # get the urls of the linked files
# for el in data_all:
#
#     video_urls = {os.path.basename(el['analysis_path'])[:-4]: el['video_analysis'] for el in data_all}
#     vr_urls = {os.path.basename(el['analysis_path'])[:-4]: el['vr_analysis'] for el in data_all}
# load the data
data_list = []
frame_list = []
for el in input_path:
    # get the trial timestamp
    time_stamp = int(''.join(os.path.basename(el).split('_')[3:6]))

    try:
        temp_data = pd.read_hdf(el, 'matched_calcium')
        data_list.append(temp_data)
        frame_list.append([time_stamp, 0, temp_data.shape[0]])
    except KeyError:
        # data_list.append([])
        frame_list.append([time_stamp, 0, 0])


try:
    # # concatenate the trials
    # sub_data = pd.concat(data_list)
    # turn the frame list into an np array
    frame_list = np.array(frame_list)
    # make the frame numbers cumulative to read them as limits
    frame_list[:, 2] = np.cumsum(frame_list[:, 2])
    frame_list[1:, 1] = frame_list[:-1, 2]
    # save in the target file
    with h5py.File(out_path, 'w') as f:
        f.create_dataset('frame_list', data=frame_list)

    # Multivariate regression

    # define the target variables
    # variable_list = [['cricket_0_x', 'cricket_0_y'], ['mouse_x', 'mouse_y'], ['cricket_0_mouse_distance']]
    # variable_title = ['cricketPosition', 'mousePosition', 'preyDistance']
    variable_list = processing_parameters.variable_list
    variable_title = processing_parameters.variable_title
    # # allocate the output dictionary
    # results_dict = {}
    # for all the entries
    for idx, target_behavior in enumerate(variable_list):

        # get the cells
        labels = list(data_list[0].columns)
        cells = [el for el in labels if 'cell' in el] + ['cricket_0_mouse_distance']

        # get the relevant columns
        sub_data = [el[target_behavior+cells] for el in data_list if (target_behavior[0] in el.columns)]
        sub_data = pd.concat(sub_data)

        # define the target variable
        # target_behavior = ['cricket_0_x', 'cricket_0_y']
        # target_behavior = ['mouse_x', 'mouse_y']
        # target_behavior = ['cricket_0_mouse_distance']

        # sub_data = pd.read_hdf(input_path, 'calcium_data')
        # get the available columns
        # labels = list(sub_data.columns)
        # cells = [el for el in labels if 'cell' in el]
        # not_cells = [el for el in labels if 'cell' not in el]
        # get the cell data
        calcium_data = np.array(sub_data[cells].copy())
        # get rid of the super small values
        calcium_data[np.isnan(calcium_data)] = 0

        # scale (convert to float to avoid warning, potentially from using too small a dtype)
        calcium_data = preprocessing.StandardScaler().fit_transform(calcium_data)

        # get the distance to cricket
        # distance = ss.medfilt(sub_data.loc[:, target_behavior].to_numpy(), 21)
        parameter = sub_data.loc[:, target_behavior].to_numpy()
        distance_to_prey = sub_data.loc[:, 'cricket_0_mouse_distance'].to_numpy()

        keep_vector = np.pad(np.diff(parameter[:, 0]) != 0, (1, 0), mode='constant', constant_values=0)
        distance_vector = distance_to_prey > 3
        # random.shuffle(distance_vector)

        # keep_vector = (keep_vector) & (distance_vector)

        # for a real and a shuffle version
        for shuffler in np.arange(2):
            parameter_working = parameter[keep_vector, :]
            calcium_data_working = calcium_data[keep_vector, :]

            if shuffler == 1:
                suffix = 'shuffle'
                random.shuffle(parameter_working)
            else:
                suffix = ''

            # train the classifier

            calcium_train, calcium_test, parameter_train, parameter_test = \
                mod.train_test_split(calcium_data_working, parameter_working, test_size=0.2)

            linear = lin.MultiTaskElasticNetCV(max_iter=5000, l1_ratio=[.1, .5, .7, .9, .95, .99, 1], n_jobs=7)
            linear.fit(calcium_train, parameter_train)
            linear_pred = linear.predict(calcium_data_working)

            # get the r2 score
            r2_score = smet.r2_score(parameter_working, linear_pred)

            # # save on the dict
            # # collect the relevant components of the regression
            # results_dict['_'.join(['coefficients', variable_list[idx], suffix])] = linear.coef_
            # results_dict['_'.join(['prediction', variable_list[idx], suffix])] = linear_pred
            # results_dict['_'.join(['r2', variable_list[idx], suffix])] = r2_score

            # save the file
            with h5py.File(out_path, 'a') as f:
                # save an empty
                f.create_dataset('_'.join(['coefficients', variable_title[idx], suffix]),
                                 data=linear.coef_)
                f.create_dataset('_'.join(['prediction', variable_title[idx], suffix]),
                                 data=linear_pred)
                f.create_dataset('_'.join(['r2', variable_title[idx], suffix]),
                                 data=r2_score)

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
        # 'video_analysis': [el for el in video_urls.values()],
        # 'vr_analysis': [el for el in vr_urls.values()],
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
except (ValueError, IndexError):
    # save in the target file
    with h5py.File(out_path, 'w') as f:
        f.create_dataset('no_ROIs', data=[])


# linear = lin.PoissonRegressor(alpha=1000, max_iter=5000)
# linear_distance = parameter/np.max(parameter)
# print(calcium_data.shape)
#
#
# print('Exp variance:'+str(smet.r2_score(parameter_test, linear_pred)))
# print(linear.alpha_)
# print(linear.l1_ratio_)
# print(linear.mse_path_)
# print(linear.alphas_)
