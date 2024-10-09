import os
import sys

# turn off GPU for cuda
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import paths
import functions_bondjango as bd
import functions_misc as fm
import functions_loaders as fl
import numpy as np
import processing_parameters
import scipy.signal as ss
import functions_glmnn as fg
import joblib as jb
from classify_batch import reverse_roll_shuffle
from sklearn import preprocessing as preproc


# load the data
# get the paths from the database using search_list
all_paths, all_queries = fl.query_search_list()
# print(all_paths)

data_list = []
# load the data
for path, queries in zip(all_paths, all_queries):
    data, _, _ = fl.load_preprocessing(path, queries)
    data_list.append(data)

# list the radial features in the dataset
radial_features = ['cricket_0_delta_heading', 'cricket_0_visual_angle', 'mouse_heading',
                   'cricket_0_delta_head', 'cricket_0_heading', 'head_direction']

# define the scalers to use
scalers = {
    'standard': preproc.StandardScaler
}

delete_list = ['mouse', 'datetime', 'hunt_trace', 'motifs']
# define the design matrix
feature_list = processing_parameters.variable_list

# original 1s sigma, 9 basis src, 0.2 s spacing
# ideal so far 20 src, 0.1 for sigma and spacing

# define the frame rate (fps)
frame_rate = 10

# get the kernel basic parameters and calculate the efective ones
kernel_parameters = processing_parameters.feature_parameters
# define the width of the kernel (s), multiplied to convert to frames
sigma = kernel_parameters['sigma_width'] * frame_rate
# calculate the kernel
kernel = ss.gaussian(sigma * kernel_parameters['sigma_factor'], sigma)
# define the number of basis src per regressor
basis_number = kernel_parameters['basis_number']
# define the kernel spacing (in s)
kernel_spacing = kernel_parameters['spacing'] * frame_rate
# get the total length of the kernel
total_length = kernel_spacing * (basis_number - 1) + kernel.shape[0]

# create the feature matrix parameter dict
feature_dict = {
    'sigma': sigma,
    'kernel': kernel,
    'basis_number': basis_number,
    'kernel_spacing': kernel_spacing,
    'total_length': total_length,
}
# # get the start positions of the basis src (assume sigma defines the interval)
# basis_starts = [int(el) for el in np.arange(-sigma*((basis_number-1)/2),
#                                        sigma*((basis_number-1)/2)+1, sigma)]
# allocate memory for the output
feature_trials = []
# allocate memory for a data frame without the encoding model features
feature_raw_trials = []
# allocate memory for the calcium
calcium_trials = []
# allocate a list for the mouse/day pairs
pairs_list = []
# get the number of trials
trial_number = len(data_list[0])
# get the features
for idx, el in enumerate(data_list[0]):

    # generate the features
    target_features, cells = fg.generate_features(el, feature_list, radial_features, **feature_dict)
    if type(target_features) == list:
        continue
    # store the features
    feature_trials.append(target_features)
    # store the calcium
    calcium_trials.append(cells)
    # store the mouse and date
    pairs_list.append([el.loc[0, 'mouse'], el.loc[0, 'datetime'][:10]])

# print(f'Time by features: {feature_trials[0].shape}')
# print(f'Time by ROIs: {calcium_trials[0].shape}')

# calculate the unique pairs for mouse and date
unique_pairs = np.unique(pairs_list, axis=0)

# allocate memory for the performances
glm_performances = []
# allocate memory for the weights
glm_weights = []
# allocate memory for the losses
glm_loss = []

# get the fitting parameters
fit_parameters = processing_parameters.fit_parameters

# replace the scaler key by its item
fit_parameters['scaler'] = scalers[fit_parameters['scaler']]
# get the number of iterations
iteration_number = processing_parameters.iteration_number

# for all the pairs
for mouse, day in unique_pairs[:1]:
    print(mouse, day)
    # find the corresponding trials
    trial_idx = [el for el in np.arange(len(feature_trials)) if
                 (mouse == pairs_list[el][0]) & (day == pairs_list[el][1])]

    # for all the trials
    current_features = [feature_trials[el] for el in trial_idx]
    current_calcium = [calcium_trials[el] for el in trial_idx]

    # allocate the output
    performance_list = []
    # with fut.ProcessPoolExecutor() as executor:
    with jb.Parallel(n_jobs=-1) as parallel:

        # train the real net
        fit_parameters['sample_shuffle'] = None
        nn_outputs = parallel(jb.delayed(fg.train_test_glm_nn)(current_features, current_calcium, **fit_parameters)
                              for el in np.arange(iteration_number))
        for idx, el in enumerate(nn_outputs):
            performance_list.append([el[0][1], idx, False])

        # train the shuffle net
        fit_parameters['sample_shuffle'] = reverse_roll_shuffle
        nn_outputs = parallel(jb.delayed(fg.train_test_glm_nn)(current_features, current_calcium, **fit_parameters)
                              for el in np.arange(iteration_number))
        for idx, el in enumerate(nn_outputs):
            performance_list.append([el[0][1], idx, True])

    # save the output
    glm_performances.append(performance)
    glm_weights.append(weights)
    glm_loss.append(loss)
