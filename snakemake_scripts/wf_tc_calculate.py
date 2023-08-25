import os
import paths
import numpy as np
import pandas as pd
import pycircstat as circ
from scipy.stats import percentileofscore, sem

import processing_parameters
import functions_bondjango as bd
import functions_kinematic as fk
import functions_tuning as tuning
from wirefree_experiment import WirefreeExperiment, DataContainer


def calculate_visual_tuning(cell, tuning_kind, dataset='norm_spikes', tuning_fit='von_mises'):
    # Get the mean reponse per trial and drop the inter-trial interval from df
    activity = getattr(cell, dataset).copy()
    trial_activity = activity.groupby([tuning_kind, 'trial_num'])[cell.id].agg(np.nanmean)
    trial_activity = trial_activity.droplevel(['trial_num'])
    trial_activity = trial_activity.drop(trial_activity[trial_activity.index == -1000].index)
    trial_responses = trial_activity.reset_index()

    #-- Create the response vectors --#
    mean_resp, angles = tuning.generate_response_vector(trial_responses, np.nanmean)
    sem_resp, _ = tuning.generate_response_vector(trial_responses, sem, nan_policy='omit')
    std_resp, _ = tuning.generate_response_vector(trial_responses, np.std)

    # Normalize if cell is responsive. Here we normalize the responses of each cell to the maximum response of the cell on any given trial
    if np.max(mean_resp) > 0:
        norm_trial_resp = trial_responses.copy()
        norm_trial_resp[cell.id] = tuning.normalize(trial_responses[cell.id])
        
        norm_mean_resp, _ = tuning.generate_response_vector(norm_trial_resp, np.nanmean)
        norm_sem_resp, _ = tuning.generate_response_vector(norm_trial_resp, sem, nan_policy='omit')
        norm_std_resp, _ = tuning.generate_response_vector(norm_trial_resp, np.std)
        
    else:
        norm_trial_resp = trial_responses
        norm_mean_resp = mean_resp
        norm_sem_resp = sem_resp
        norm_std_resp = std_resp
    
    # -- Fit tuning curves to get preference-- #
    if 'direction' in tuning_kind:
        if tuning_fit == 'von_mises':
            fit_function = tuning.calculate_pref_direction_vm
        else:
            fit_function = tuning.calculate_pref_direction
    else:
        fit_function = tuning.calculate_pref_orientation

    # Calculate fit on whole dataset and get R2
    fit, fit_curve, pref_angle, real_pref_angle = fit_function(norm_trial_resp[tuning_kind], norm_trial_resp[cell.id], 
                                                               mean=angles[np.argmax(norm_mean_resp)])
    fit_r2 = tuning.fit_r2(norm_trial_resp[tuning_kind], norm_trial_resp[cell.id], fit_curve[:,0], fit_curve[:,1])

    # -- Get resultant vector and respose variance-- #
    thetas = np.deg2rad(norm_trial_resp[tuning_kind])
    magnitudes = norm_trial_resp[cell.id]
    angle_sep = np.mean(np.diff(thetas))
    
    resultant_length = circ.resultant_vector_length(thetas, w=magnitudes, d=angle_sep)
    resultant_angle = circ.mean(thetas, w=magnitudes, d=angle_sep)
    resultant_angle = np.rad2deg(resultant_angle)

    circ_var = circ.var(thetas, w=magnitudes, d=angle_sep)
    responsivity = 1 - circ_var

    # -- Run permutation test -- #
    # Here we shuffle the trial IDs and compare the real selectivity index to the bootstrapped distribution
    _, shuffled_responsivity = tuning.bootstrap_responsivity(thetas, magnitudes, num_shuffles=500)
    p = percentileofscore(shuffled_responsivity, responsivity, kind='mean') / 100.

    # Try leave one out

    # -- Assign variables to the cell class -- #
    vars = ['cell_id', 'trial_resp', 'trial_resp_norm', 'mean', 'mean_norm', 'std', 'std_norm', 'sem', 'sem_norm', \
            'tuning_curve', 'tuning_curve_norm', 'resultant', 'circ_var', 'responsivity', \
            'fit', 'fit_curve', 'fit_r2', 'pref', 'shown_pref']
    
    data = [cell.id, trial_responses.to_numpy(), norm_trial_resp.to_numpy(), mean_resp, norm_mean_resp, std_resp, norm_std_resp, sem_resp, norm_sem_resp, \
            np.vstack([angles, mean_resp]).T, np.vstack([angles, norm_mean_resp]).T, (resultant_length, resultant_angle), circ_var, responsivity, \
            fit, fit_curve, fit_r2, pref_angle, real_pref_angle]

    df = pd.DataFrame(columns=vars, dtype='object')
    df = df.append(dict(zip(vars, data)), ignore_index=True)
    return df

    # setattr(cell, f'{data_type}_{tuning_label}_props', df)


def tuning_loop(experiment, tuning_kind, dataset, **kwargs):
    df_list = []
    for _, cell in list(experiment.cell_props.items()):
        df = calculate_visual_tuning(cell, tuning_kind, dataset=dataset, **kwargs)
        df_list.append(df)

    props = pd.concat(df_list)
    tuning_label = tuning_kind.split('_')[0]
    setattr(experiment, f'{dataset}_{tuning_label}_props', props)


def filter_speed(exp, dataset_name, speed_column, percentile):
    speed_cutoff = np.percentile(np.abs(exp.kinematics[speed_column]), percentile)
    exp.kinematics['is_running'] = np.abs(exp.kinematics[speed_column]) >= speed_cutoff
    exp.kinematics[f'{speed_column}_abs'] = np.abs(exp.kinematics[speed_column])

    still_trials = exp.kinematics.groupby('trial_num').filter(lambda x: x[f'{speed_column}_abs'].mean() < speed_cutoff).trial_num.unique()
    still_trials = viewed_trials[np.in1d(viewed_trials, still_trials)]
    still_spikes = getattr(exp, dataset_name).loc[exp.norm_spikes_viewed.trial_num.isin(still_trials)]

    attr_name = f'{dataset_name}_still'
    setattr(exp, attr_name, still_spikes.copy())
    exp.add_attributes_to_cells(['attr_name'])


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
        name_parts = os.path.basename(out_path).split('_')
        day = '_'.join(name_parts[:3])
        animal = '_'.join(name_parts[3:6])
        rig = name_parts[6]

    except NameError:
        # get the search query
        search_string = processing_parameters.search_string

        # get the paths from the database
        data_all = bd.query_database('analyzed_data', search_string)
        data_all = [el for el in data_all if '_preproc' in el['slug']]
        input_path = [el['analysis_path'] for el in data_all if '_preproc' in el['slug']]
        # get the day, animal and rig
        day = '_'.join(data_all[0]['slug'].split('_')[0:3])
        rig = data_all[0]['rig']
        animal = data_all[0]['slug'].split('_')[7:10]
        animal = '_'.join([animal[0].upper()] + animal[1:])

        # assemble the output path
        out_path = os.path.join(paths.analysis_path, '_'.join((day, animal, rig, 'tcday.hdf5')))

    for idx, file in enumerate(input_path):

        # Load experiment
        exp = WirefreeExperiment(file, use_xarray=False)

        # Calculate dFF and normalize other neural data
        exp.dff = tuning.calculate_dff(exp.raw_fluor)
        exp.norm_spikes = tuning.normalize_responses(exp.raw_spikes)
        exp.norm_fluor = tuning.normalize_responses(exp.raw_fluor)
        exp.norm_dff = tuning.normalize_responses(exp.dff)
        exp.add_attributes_to_cells(['raw_spikes', 'norm_spikes', 'raw_fluor', 'norm_fluor', 'dff', 'norm_dff'])

        # Tidy up pitch, yaw, roll
        pitch = -fk.wrap_negative(exp.kinematics.mouse_xrot_m.values)
        exp.kinematics['pitch'] = tuning.smooth_trace(pitch, range=(-180, 180), kernel_size=10, discont=2*np.pi)

        yaw = fk.wrap_negative(exp.kinematics.mouse_zrot_m.values)
        exp.kinematics['yaw'] = tuning.smooth_trace(yaw, range=(-180, 180), kernel_size=10, discont=2*np.pi)

        roll = fk.wrap_negative(exp.kinematics.mouse_yrot_m.values)
        exp.kinematics['roll'] = tuning.smooth_trace(roll, range=(-180, 180), kernel_size=10, discont=2*np.pi)

        # Filter trials by head pitch
        pitch_upper_cutoff = 20
        pitch_lower_cutoff = -90
        view_fraction = 0.7
        exp.kinematics['viewed'] = np.logical_and(exp.kinematics.pitch >= pitch_lower_cutoff, exp.kinematics.pitch <= pitch_upper_cutoff)
        viewed_trials = exp.kinematics.groupby('trial_num').filter(lambda x: (x['viewed'].sum() / len(x['viewed'])) > view_fraction).trial_num.unique()

        norm_spikes_viewed = exp.norm_spikes.loc[exp.norm_spikes.trial_num.isin(viewed_trials)]
        exp.norm_spikes_viewed = norm_spikes_viewed.copy()
        exp.add_attributes_to_cells(['norm_spikes_viewed'])

        # Filter trials by running speed
        if rig == 'VTuningWF':
            filter_speed(exp, 'mouse_speed', 80)
        else:
            filter_speed(exp, 'wheel_speed', 80)

        # Run the tuning loop
        for tuning_type in ['direction_wrapped', 'orientation']:
            for dataset in ['norm_spikes', 'norm_spikes_viewed', 'norm_spikes_viewed_still']:
                tuning_loop(exp, tuning_type, dataset)



