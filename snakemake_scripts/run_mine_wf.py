import sys
import os
import yaml
import h5py
import numpy as np
import pandas as pd
import sklearn.preprocessing as preproc

import paths
import processing_parameters
import functions_bondjango as bd
from functions_data_handling import parse_search_string
from functions_misc import slugify
from functions_tuning import normalize_responses
from wf_tc_calculate import parse_kinematic_data

sys.path.insert(0, os.path.abspath(r'C:\Users\setup\Repositories\mine_pub'))
from mine import Mine, MineData


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
        file_info = [entry for entry in data_all if (parsed_search['mouse'] in entry['analysis_path']) and
                     ('preproc' in entry['slug'])][0]
        input_path = file_info['analysis_path']
        out_path = input_path.replace('preproc', 'mine')
        # get the day, animal and rig
        day = '_'.join(file_info['slug'].split('_')[0:3])
        rig = file_info['rig']
        animal = file_info['slug'].split('_')[7:10]
        animal = '_'.join([animal[0].upper()] + animal[1:])

    # define ca activity type
    ca_type = 'dff'    # 'dff', 'spikes' or 'deconv_fluor'

    # define if dropping ITI
    drop_ITI = False

    # get some vars from the processing parameters
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
        full_dataset = h['/matched_calcium']

    # Parse the data into calcium and kinematic data
    kinematics, raw_fluor, dff, inferred_spikes, deconvolved_fluor = parse_kinematic_data(full_dataset, rig)

    if ca_type == 'dff':
        data = dff
    elif ca_type == 'spikes':
        data = inferred_spikes
    elif ca_type == 'deconv_fluor':
        data = deconvolved_fluor
    else:
        ValueError('Unrecognized calcium activity type')

    # If using fluorescence data, calculate normalized dF/F
    if ca_type in ['dff', 'deconv_fluor']:
        data = normalize_responses(data)

    # Concatenate the kinematic data to the selected calcium data
    cell_cols = [col for col in data.columns if 'cell' in col]
    used_data = pd.concat([kinematics, data], axis=1)

    # Drop the ITI
    if drop_ITI:
        used_data.drop(used_data[used_data['trial_num'] == 0].index, inplace=True)
        used_data.reset_index(drop=True, inplace=True)

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
    mine_data = run_mine(used_data, MINE_params, variable_list)

    # Save data
    with h5py.File(out_path, 'w') as f:
        mine_data.save_to_hdf5(f, overwrite=True)
        # f.create_dataset('cell_ids', data=cell_ids.astype('S'))

    # Define the result for correct saving (not necessarily consistent with proproc data)
    if 'control' in input_path:
        result = 'control'
    elif 'repeat' in input_path:
        result = 'repeat'
    elif 'fullfield' in input_path:
        result = 'fullfield'
    else:
        result = 'multi'

    # save as a new entry to the data base
    # assemble the entry data
    entry_data = {
        'analysis_type': 'mine_analysis',
        'analysis_path': out_path,
        'date': '',
        'pic_path': '',
        'result': result,
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
