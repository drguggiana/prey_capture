import pandas as pd
import numpy as np
import sklearn.preprocessing as preproc
from mine_pub.mine import Mine
import processing_parameters
import paths


def run_mine(target_trials, mine_params, predictor_columns):
    """Run MINE"""

    # concatenate
    all_trials = pd.concat(target_trials, axis=0)

    # get the calcium and predictors
    cell_columns = [el for el in all_trials.columns if 'cell' in el]

    calcium = all_trials[cell_columns].to_numpy()
    predictors = all_trials[predictor_columns].fillna(0).to_numpy()

    # split into train and test for scaling and augmentation
    train_frames = int(mine_params['tt_split'] * calcium.shape[0])
    calcium_train = calcium[:train_frames, :]
    calcium_test = calcium[train_frames:, :]
    calcium_scaler = preproc.StandardScaler().fit(calcium_train)
    calcium_train = calcium_scaler.transform(calcium_train)
    calcium_test = calcium_scaler.transform(calcium_test)
    # duplicate calcium for the augmented predictors
    #     if augmentation_factor == -1:
    #         MINE_params['tt_split'] = 2/3
    calcium = np.concatenate([calcium_train, calcium_test], axis=0)
    #     else:
    #         MINE_params['tt_split'] = 4/5
    #         calcium = np.concatenate([calcium_train, calcium_train, calcium_test], axis=0)

    predictors_train = predictors[:train_frames, :]
    predictors_test = predictors[train_frames:, :]
    predictors_scaler = preproc.StandardScaler().fit(predictors_train)
    predictors_train = predictors_scaler.transform(predictors_train)
    predictors_test = predictors_scaler.transform(predictors_test)
    #     if augmentation_factor == -1:
    predictors = np.concatenate([predictors_train, predictors_test], axis=0)
    #     else:
    #         predictors_aug = predictors_train.copy() + np.random.randn(*predictors_train.shape)*augmentation_factor
    #         # add the augmented data
    #         predictors = np.concatenate([predictors_train, predictors_aug, predictors_test], axis=0)
    #     print(predictors_train[:5, 0])
    #     print(predictors_aug[:5, 0])
    #     raise ValueError
    # create the MINE element
    miner = Mine(*mine_params.values())
    # run Mine
    mine_data = miner.analyze_data(predictors.T, calcium.T)

    return mine_data
