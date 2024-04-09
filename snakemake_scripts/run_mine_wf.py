import sys
import os
import yaml
import h5py
import numpy as np
import pandas as pd
import sklearn.preprocessing as preproc

import paths
import processing_parameters
import functions_kinematic as fk
import functions_bondjango as bd
from functions_data_handling import parse_search_string
from functions_misc import slugify
from functions_tuning import calculate_dff, normalize_responses

sys.path.insert(0, os.path.abspath(r'F:\Repositories\mine_pub\prey_capture'))
sys.path.insert(0, os.path.abspath(r'F:\Repositories\mine_pub'))
from mine import Mine, MineData


def calculate_extra_angles(ds, exp_type):
    # Apply wrapping for directions to get range [0, 360]
    ds['direction_wrapped'] = ds['direction'].copy()
    mask = ds['direction_wrapped'] > -1000
    ds.loc[mask, 'direction_wrapped'] = ds.loc[mask, 'direction_wrapped'].apply(fk.wrap)

    # Now find the direction relative to the ground plane
    if exp_type == 'free':
        ds['direction_rel_ground'] = ds['direction_wrapped'].copy()
        ds.loc[mask, 'direction_rel_ground'] = ds.loc[mask, 'direction_rel_ground'] + ds.loc[mask, 'head_roll']
    else:
        ds['direction_rel_ground'] = ds['direction_wrapped'].copy()

    # Calculate orientation explicitly
    if 'orientation' not in ds.columns:
        ds['orientation'] = ds['direction_wrapped'].copy()
        ds['orientation_rel_ground'] = ds['direction_rel_ground'].copy()
        mask = ds['orientation'] > -1000
        ds.loc[mask, 'orientation'] = ds.loc[mask, 'orientation'].apply(fk.wrap, bound=180.1)
        ds.loc[mask, 'orientation_rel_ground'] = ds.loc[mask, 'orientation_rel_ground'].apply(fk.wrap, bound=180.1)

    return ds


def run_mine(target_trials, mine_params, predictor_columns):
    """Run MINE"""

    # concatenate
    # all_trials = pd.concat(target_trials, axis=0)
    all_trials = target_trials
    all_trials.dropna(inplace=True)

    # get the calcium and predictors
    cell_columns = [el for el in all_trials.columns if 'cell' in el]

    calcium = all_trials[cell_columns].to_numpy()
    predictors = all_trials[predictor_columns].fillna(0).to_numpy()

    # split into train and test for scaling and augmentation
    train_frames = int(mine_params['train_fraction'] * calcium.shape[0])

    calcium_train = calcium[:train_frames, :]
    calcium_test = calcium[train_frames:, :]

    calcium_scaler = preproc.StandardScaler().fit(calcium_train)
    calcium_train = calcium_scaler.transform(calcium_train)
    calcium_test = calcium_scaler.transform(calcium_test)

    # duplicate calcium for the augmented predictors
    #     if augmentation_factor == -1:
    #         MINE_params['train_fraction'] = 2/3
    calcium = np.concatenate([calcium_train, calcium_test], axis=0)
    #     else:
    #         MINE_params['train_fraction'] = 4/5
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



if __name__ == '__main__':
    # get the data paths
    try:
        input_path = snakemake.input[0]
        out_path = snakemake.output[0]
        file_info = yaml.load(snakemake.params.file_info, Loader=yaml.FullLoader)
        # get the slugs
        slug = file_info['slug']
        day = '_'.join(file_info['slug'].split('_')[0:3])
        rig = file_info['rig']
        animal = file_info['slug'].split('_')[7:10]
        animal = '_'.join([animal[0].upper()] + animal[1:])

    except NameError:
        # get the paths from the database
        search_string = processing_parameters.search_string
        parsed_search = parse_search_string(search_string)
        data_all = bd.query_database('analyzed_data', processing_parameters.search_string)
        file_info = [entry for entry in data_all if parsed_search['mouse'] in entry['analysis_path']][0]
        input_path = file_info['analysis_path']
        out_path = input_path.replace('preproc', 'mine')
        # get the day, animal and rig
        day = '_'.join(file_info['slug'].split('_')[0:3])
        rig = file_info['rig']
        animal = file_info['slug'].split('_')[7:10]
        animal = '_'.join([animal[0].upper()] + animal[1:])

    # define ca activity type
    ca_type = 'fluor'    #'spikes' or 'fluor'
    # define if dropping ITI
    drop_ITI = False
    # get frame rate
    frame_rate = processing_parameters.wf_frame_rate

    variable_list = processing_parameters.variable_list_visual

    if rig in ['VWheelWF', 'VWheel']:
        variable_list += processing_parameters.variable_list_fixed
        exp_type = 'fixed'
    elif rig in ['VTuningWF', 'VTuning']:
        variable_list += processing_parameters.variable_list_free
        exp_type = 'free'
    else:
        ValueError('Unrecognized rig')

    # define the columns to exclude
    exclude_columns = ['mouse', 'datetime', 'motifs']

    # Load the data
    with pd.HDFStore(input_path, mode='r') as h:
        data = h['matched_calcium']

    # drop activity not of correct type
    cols_to_drop = [el for el in data.columns if ('cell' in el) and (ca_type not in el)]
    data.drop(cols_to_drop, axis='columns', inplace=True)

    # If using fluorescence data, calulate normalized dF/F
    if ca_type == 'fluor':
        data = calculate_dff(data, baseline_type='quantile', inplace=False)
        data = normalize_responses(data)

    # Do a quick calculation of orientation & dir/ori relative to ground
    if ('direction_wrapped' in variable_list) and ('direction_wrapped' not in data.columns):
        data = calculate_extra_angles(data, exp_type)

    if ('wheel_speed_abs' in variable_list) and ('wheel_speed_abs' not in data.columns):
        data['wheel_speed_abs'] = np.abs(data['wheel_speed']).copy()

    # Drop the ITI
    if drop_ITI:
        data.drop(data[data['trial_num'] == 0].index, inplace=True)
        data.reset_index(drop=True, inplace=True)

    # define the MINE parameters
    MINE_params = {
        'train_fraction': 2.0/3.0,
        'model_history': frame_rate*10,
        'corr_cut': 0.2,
        'compute_taylor': True,
        'return_jacobians': True,
        'taylor_look_ahead': frame_rate*5,
        'taylor_pred_every': frame_rate*5,
    }

    # Run MINE
    mine_data = run_mine(data, MINE_params, variable_list)

    # Save data
    with h5py.File(out_path, 'w') as f:
        mine_data.save_to_hdf5(f, overwrite=True)
        # f.create_dataset('cell_ids', data=cell_ids.astype('S'))

    # save as a new entry to the data base
    # assemble the entry data
    entry_data = {
        'analysis_type': 'mine_analysis',
        'analysis_path': out_path,
        'date': '',
        'pic_path': '',
        'result': str(file_info['result']),
        'rig': str(file_info['rig']),
        'lighting': str(file_info['lighting']),
        'imaging': 'wirefree',
        'slug': slugify(os.path.basename(out_path)[:-5]),

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

    print("Done calculating tuning curves!")