import numpy as np
import pandas as pd
import scipy.signal as ss
import scipy.stats as stat
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam


def maxmin(array_in):
    """Normalize an array by its maximum and minimum"""
    return (array_in - np.nanmin(array_in)) / (np.nanmax(array_in) - np.nanmin(array_in))


def basis_predictors(variable, basis_number=None, kernel=None, kernel_spacing=None, total_length=None, label=None):
    """Generate basis functions based on the time course of a give variable"""
    # initialize the output dataframe
    out_frame = pd.DataFrame()
    # generate the displaced basis functions
    for idx2 in np.arange(basis_number):
        # generate the sizes of the before and after padding of the kernel
        back = int(kernel_spacing * idx2)
        front = int(total_length - kernel.shape[0] - back)
        # generate the full kernel
        if back == 0:
            current_kernel = np.concatenate((kernel, np.zeros(front)))
        elif idx2 == basis_number - 1:
            current_kernel = np.concatenate((np.zeros(back), kernel))
        else:
            current_kernel = np.concatenate((np.zeros(back), kernel, np.zeros(front)))

        #         # if the whole kernel is longer than the trial, trim the kernel at the end
        #         if current_kernel.shape[0] > variable.shape[0]:
        #             current_kernel = current_kernel[:variable.shape[0]]

        # convolve with the data
        vector = np.convolve(variable, current_kernel, 'full')

        # normalize to 0-1
        vector = maxmin(vector)
        # if the vector was all zeros, it'll turn into nans so remove
        vector[np.isnan(vector)] = 0

        # generate the field in the new data frame
        out_frame[label + '_' + str(idx2)] = vector

    return out_frame


# set up the feature and calcium matrices
def generate_features(data_trial, feature_list, radial_features, **params):
    """Take a single trial worth of data and output the converted features and calcium"""
    # get the intersection of the labels
    label_intersect = [feat for feat in feature_list if feat in data_trial.columns]

    if len(label_intersect) != len(feature_list):
        return [], []
    # get the features of interest
    target_features = data_trial.loc[:, feature_list]
    # # save the original features for simpler calculations
    # feature_raw_trials.append(target_features.copy())
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
        # check if the feature is radial
        if label in radial_features:
            # get the feature
            rad_feature = target_features[label].copy().to_numpy()
            # convert to radians
            rad_feature = np.deg2rad(rad_feature)
            # perform angular decomposition (assume unit circle)
            x = np.cos(rad_feature)
            y = np.sin(rad_feature)
            # replace the original column by the extracted ones
            target_features[label + '_x'] = x
            target_features[label + '_y'] = y
            # drop the original column
            target_features.drop(labels=label, axis=1, inplace=True)
        # check if the label is a speed and calculate acceleration
        if 'speed' in label:
            # get the speed
            speed = target_features[label].copy().to_numpy()
            # calculate the acceleration with the smoothed speed
            acceleration = np.concatenate(([0], np.diff(ss.medfilt(speed, 21))), axis=0)
            # add to the features
            target_features[label.replace('speed', 'acceleration')] = acceleration

    # # save the group names for later use
    # group_names = target_features.columns
    # Generate the gaussian convolved and displaced regressors
    # allocate an empty dataframe for the outputs
    new_dataframe = pd.DataFrame()
    # for all the regressors
    for label in target_features:
        # get the variable
        variable = target_features[label].to_numpy().copy()
        # Remove nans
        variable[np.isnan(variable)] = 0

        # get the basis function-based predictors
        # out_frame = basis_predictors(variable, basis_number, kernel, kernel_spacing, total_length, label)
        out_frame = basis_predictors(variable, **params)

        # add to the dataframe
        new_dataframe = pd.concat((new_dataframe, out_frame), axis=1)

    #     # add a constant factor
    #     constant = np.ones(new_dataframe.shape[0])
    #     new_dataframe['constant'] = constant
    # add a trial factor
    #     new_dataframe['trial'] = idx*np.ones(vector.shape[0])
    #     # for all the trials
    #     for trial in np.arange(trial_number):
    #         new_dataframe['trial_'+str(trial)] = np.zeros(vector.shape[0])
    #         if trial == idx:
    #             new_dataframe['trial_'+str(trial)] += 1

    # replace the old dataframe with the new one
    target_features = new_dataframe

    # store the columns
    resulting_columns = target_features.columns
    # turn the dataframe into an array
    target_features = target_features.to_numpy()

    # get the calcium data
    cells = [cell for cell in data_trial.columns if 'cell' in cell]
    cells = data_trial.loc[:, cells].to_numpy()

    # get the difference in length between features and calcium
    pad_length = int((target_features.shape[0] - data_trial.shape[0]) / 2)
    # pad to match the features (to get past prediction as well)
    cells = np.concatenate((np.zeros((pad_length, cells.shape[1])), cells, np.zeros((pad_length, cells.shape[1]))))
    # check that the dimensions match
    assert cells.shape[0] == target_features.shape[0]

    return target_features, cells


def train_test_glm_nn(current_features, current_calcium, scaler=None, activation='relu', l1=0.01, l2=0.01, loss='mse',
                      learning_rate=0.001, validation_split=0.3,
                      batch_size=100, epochs=200, test_train_shuffle=True, verbose=0, sample_shuffle=None,
                      scale_calcium=False, cell_subset=False, return_model=False,
                      rotate_data=False, noise=2):
    """Train a GLM-NN for the given data"""

    #     # copy the data
    #     current_features = current_features_in.copy()
    #     current_calcium = current_calcium_in.copy()

    # scale the features
    if scaler is not None:
        current_features = [scaler().fit_transform(el) for el in current_features]
    # scale the calcium per trial
    if scale_calcium:
        current_calcium = [el / np.max(el) for el in current_calcium]

    # concatenate them
    current_features = np.vstack(current_features)
    current_calcium = np.vstack(current_calcium)

    # if the rotate flag is on, change the order so that the validation set is different
    if rotate_data:
        # get the random number of elements to rotate the data by
        roll_factor = int(np.round(np.random.sample() * current_features.shape[0])) - current_features.shape[0]
        # rotate calcium and behavior according to the given index
        current_features = np.roll(current_features, roll_factor, axis=0)
        current_calcium = np.roll(current_calcium, roll_factor, axis=0)

    # take a subset of cells for prototyping
    if cell_subset:
        current_calcium = current_calcium[:, :10]

    # get the trial data
    X = current_features.copy()

    # get the calcium
    y = current_calcium.copy()
    # shuffle if needed
    if sample_shuffle is not None:
        y = sample_shuffle(y)

    # get the number of output features
    output_features = y.shape[1]
    # define the optimizer and network parameters
    model = tf.keras.Sequential()
    initializer = tf.keras.initializers.GlorotNormal(seed=1)
    model.add(layers.Dense(output_features, activation=activation, kernel_initializer=initializer))
    model.add(layers.ActivityRegularization(l1=l1, l2=l2))
    model.add(layers.GaussianNoise(noise))

    # compile the model with the Adam optimizer
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss=loss)

    # train the model
    history = model.fit(X, y, validation_split=validation_split, batch_size=batch_size, epochs=epochs,
                        shuffle=test_train_shuffle, verbose=verbose)

    # calculate performance
    predictions = model.predict(X)
    correlations_per_cell = [stat.spearmanr(predictions[:, el], y[:, el])[0] for el in np.arange(predictions.shape[1])]
    average_correlation = np.nanmean(correlations_per_cell)
    performance = [correlations_per_cell, average_correlation]
    # extract the weights
    weights = model.layers[0].get_weights()[0]
    # extract the history
    history = history.history

    if return_model:
        model_out = model
    else:
        model_out = []

    return performance, weights, history, X, y, model_out
