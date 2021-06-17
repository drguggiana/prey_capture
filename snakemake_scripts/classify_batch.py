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


# get the data paths
try:
    # get the target video path
    video_path = sys.argv[1]
    # find the occurrences of .tif terminators
    ends = [el.start() for el in re.finditer('.tif', video_path)]
    # allocate the list of videos
    video_list = []
    count = 0
    # read the paths
    for el in ends:
        video_list.append(video_path[count:el + 4])
        count = el + 5

    video_path = video_list
    # read the output path and the input file urls
    out_path = sys.argv[2]
    data_all = json.loads(sys.argv[3])
    # get the parts for the file naming
    name_parts = out_path.split('_')
    day = name_parts[0]
    animal = name_parts[1]
    rig = name_parts[2]

except IndexError:
    # get the search string
    search_string = processing_parameters.search_string_calcium
    animal = processing_parameters.animal
    day = processing_parameters.day
    rig = processing_parameters.rig
    # query the database for data to plot
    data_all = bd.query_database('video_experiment', search_string)
    # video_data = data_all[0]
    # video_path = video_data['tif_path']
    video_path = [el['tif_path'] for el in data_all]
    # overwrite data_all with just the urls
    data_all = {os.path.basename(el['bonsai_path'])[:-4]: el['url'] for el in data_all}
    # assemble the output path
    out_path = os.path.join(paths.analysis_path, '_'.join((day, animal, rig, 'calciumday.hdf5')))

# Multivariate regression
# for all the entries
for current_path in paths_all:
    # define the target variable
    # target_behavior = ['cricket_0_x', 'cricket_0_y']
    # target_behavior = ['mouse_x', 'mouse_y']
    target_behavior = ['cricket_0_mouse_distance']

    sub_data = pd.read_hdf(current_path, 'calcium_data')
    # get the available columns
    labels = list(sub_data.columns)
    cells = [el for el in labels if 'cell' in el]
    not_cells = [el for el in labels if 'cell' not in el]
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
    parameter = parameter[keep_vector, :]
    calcium_data = calcium_data[keep_vector, :]

    # random.shuffle(parameter)

    # # shufle the label vector
    # random.shuffle(label_vector)
    # random.shuffle(distance)

    # train the classifier

    calcium_train, calcium_test, parameter_train, parameter_test = \
        mod.train_test_split(calcium_data, parameter, test_size=0.2)

    linear = lin.MultiTaskElasticNetCV(max_iter=5000, l1_ratio=[.1, .5, .7, .9, .95, .99, 1], n_jobs=7)
    linear.fit(calcium_train, parameter_train)
    linear_pred = linear.predict(calcium_data)

    # get the r2 score
    r2_score = smet.r2_score(parameter, linear_pred)
    # collect the relevant components of the regression
    results_list = [linear.coef_, linear_pred, r2_score]
    # save as a new entry to the



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
