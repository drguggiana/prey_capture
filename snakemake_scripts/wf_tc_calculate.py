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
from tc_calculate import *

def calculate_visual_tuning(activity_df, tuning_kind, tuning_fit='von_mises', bootstrap_shuffles=500):
    # Get the mean response per trial and drop the inter-trial interval from df
    cells = [col for col in activity_df.columns if 'cell' in col]
    mean_trial_activity = activity_df.groupby([tuning_kind, 'trial_num'])[cells].agg(np.nanmean).reset_index()
    mean_trial_activity = mean_trial_activity.drop(mean_trial_activity[mean_trial_activity.trial_num == 0].index)

    # -- Create the response vectors --#
    mean_resp, unique_angles = tuning.generate_response_vector(mean_trial_activity, np.nanmean)
    sem_resp, _ = tuning.generate_response_vector(mean_trial_activity, sem, nan_policy='omit')
    std_resp, _ = tuning.generate_response_vector(mean_trial_activity, np.std)

    # Normalize the responses of each cell to the maximum response of the cell on any given trial
    norm_trial_activity = mean_trial_activity.copy()
    norm_trial_activity[cells] = norm_trial_activity[cells].apply(tuning.normalize)
    norm_mean_resp, _ = tuning.generate_response_vector(norm_trial_activity, np.nanmean)
    norm_sem_resp, _ = tuning.generate_response_vector(norm_trial_activity, sem, nan_policy='omit')
    norm_std_resp, _ = tuning.generate_response_vector(norm_trial_activity, np.std)

    # -- Fit tuning curves to get preference-- #
    if 'direction' in tuning_kind:
        if tuning_fit == 'von_mises':
            fit_function = tuning.calculate_pref_direction_vm
        else:
            fit_function = tuning.calculate_pref_direction
    else:
        fit_function = tuning.calculate_pref_orientation

    cell_data_list = []
    for cell in cells:
        # -- Calculate fit and responsivity using all trials -- #
        mean_guess = unique_angles[np.argmax(norm_mean_resp[cell], axis=0)]
        fit, fit_curve, pref_angle, real_pref_angle = fit_function(norm_trial_activity[tuning_kind].to_numpy(),
                                                                   norm_trial_activity[cell].to_numpy(),
                                                                   mean=mean_guess)
        fit_r2 = tuning.fit_r2(norm_trial_activity[tuning_kind].to_numpy(), norm_trial_activity[cell].to_numpy(),
                               fit_curve[:, 0], fit_curve[:, 1])

        # -- Get resultant vector and response variance-- #
        thetas = np.deg2rad(norm_trial_activity[tuning_kind].to_numpy())
        theta_sep = np.mean(np.diff(thetas))
        magnitudes = norm_trial_activity[cell].to_numpy()

        resultant_length = circ.resultant_vector_length(thetas, w=magnitudes, d=theta_sep)
        resultant_angle = circ.mean(thetas, w=magnitudes, d=theta_sep)
        resultant_angle = np.rad2deg(resultant_angle)

        circ_var = circ.var(thetas, w=magnitudes, d=theta_sep)
        responsivity = 1 - circ_var

        # -- Run permutation tests -- #
        # Calculate fit on subset of data
        bootstrap_r2 = tuning.bootstrap_tuning_curve(norm_trial_activity[[tuning_kind, 'trial_num', cell]],
                                                     fit_function,
                                                     num_shuffles=bootstrap_shuffles, mean=mean_guess)
        p_r2 = percentileofscore(bootstrap_r2, fit_r2, kind='mean') / 100.

        # Shuffle the trial IDs and compare the real selectivity index to the bootstrapped distribution
        bootstrap_responsivity = tuning.bootstrap_responsivity(thetas, magnitudes, num_shuffles=bootstrap_shuffles)
        p_res = percentileofscore(bootstrap_responsivity, responsivity, kind='mean') / 100.

        cell_data = [mean_trial_activity[[tuning_kind, cell]].to_numpy(),
                     norm_trial_activity[[tuning_kind, cell]].to_numpy(),
                     np.vstack([unique_angles, mean_resp[cell].to_numpy()]).T,
                     np.vstack([unique_angles, norm_mean_resp[cell].to_numpy()]).T,
                     std_resp[cell].to_numpy(), norm_std_resp[cell].to_numpy(),
                     sem_resp[cell].to_numpy(), norm_sem_resp[cell].to_numpy(),
                     fit, fit_curve, fit_r2, pref_angle, real_pref_angle,
                     (resultant_length, resultant_angle), circ_var, responsivity,
                     bootstrap_r2, p_r2, bootstrap_responsivity, p_res]

        cell_data_list.append(cell_data)

    # -- Assemble large dataframe -- #
    data_cols = ['trial_resp', 'trial_resp_norm', 'mean', 'mean_norm', 'std', 'std_norm', 'sem', 'sem_norm',
                 'fit', 'fit_curve', 'fit_r2', 'pref', 'shown_pref', 'resultant', 'circ_var', 'responsivity',
                 'bootstrap_fit_r2', 'p_r2', 'bootstrap_responsivity', 'p_responsivity']

    data_df = pd.DataFrame(index=cells, columns=data_cols, data=cell_data_list)
    return data_df


def parse_kinematic_data(matched_calcium):
    # Calculate orientation explicitly
    if 'orientation' not in matched_calcium.columns:
        matched_calcium['orientation'] = matched_calcium['direction']
        matched_calcium['orientation'][
            (matched_calcium['orientation'] > -180) & (matched_calcium['orientation'] < 0)] += 180

    # Apply wrapping for directions to get range [0, 360]
    matched_calcium['direction_wrapped'] = matched_calcium['direction']
    mask = matched_calcium['direction_wrapped'] > -1000
    matched_calcium.loc[mask, 'direction_wrapped'] = matched_calcium.loc[mask, 'direction_wrapped'].apply(wrap)

    # For all data
    if rig in ['VWheel', 'VWheelWF']:
        exp_type = 'fixed'
    else:
        exp_type = 'free'

    spikes_cols = [key for key in matched_calcium.keys() if 'spikes' in key]
    fluor_cols = [key for key in matched_calcium.keys() if 'fluor' in key]
    motive_tracking_cols = ['mouse_y_m', 'mouse_z_m', ' mouse_x_m', 'mouse_yrot_m', 'mouse_zrot_m', 'mouse_xrot_m']

    # If there is more than one spatial or temporal frequency, include it, othewise don't
    stimulus_cols = ['trial_num', 'time_vector', 'direction', 'direction_wrapped', 'orientation', 'grating_phase']

    # For headfixed data
    eye_cols = ['eye_horizontal_vector_x', 'eye_horizontal_vector_y', 'eye_midpoint_x', 'eye_midpoint_y',
                'pupil_center_ref_x', 'pupil_center_ref_y', 'fit_pupil_center_x', 'fit_pupil_center_y',
                'pupil_diameter', 'minor_axis', 'pupil_rotation', 'eyelid_angle']
    wheel_cols = ['wheel_speed', 'wheel_acceleration']

    # For free data
    mouse_kinem_cols = ['mouse_heading', 'mouse_angular_speed', 'mouse_speed', 'mouse_acceleration', 'head_direction',
                        'head_height']
    mouse_dlc_cols = ['mouse_snout_x', 'mouse_snout_y', 'mouse_barl_x', 'mouse_barl_y', 'mouse_barr_x', 'mouse_barr_y',
                      'mouse_x', 'mouse_y', 'mouse_body2_x', 'mouse_body2_y',
                      'mouse_body3_x', 'mouse_body3_y', 'mouse_base_x', 'mouse_base_y', 'mouse_head_x', 'mouse_head_y',
                      'miniscope_top_x', 'miniscope_top_y']

    if exp_type == 'fixed':
        kinematics = matched_calcium.loc[:, stimulus_cols + motive_tracking_cols + eye_cols + wheel_cols]
    else:
        kinematics = matched_calcium.loc[:, stimulus_cols + motive_tracking_cols + mouse_kinem_cols]

    raw_spikes = matched_calcium.loc[:, stimulus_cols + spikes_cols]
    raw_spikes.columns = [key.rsplit('_', 1)[0] if 'spikes' in key else key for key in self.raw_spikes.columns]
    raw_fluor = matched_calcium.loc[:, stimulus_cols + fluor_cols]
    raw_fluor.columns = [key.rsplit('_', 1)[0] if 'fluor' in key else key for key in self.raw_fluor.columns]

    return kinematics, raw_spikes, raw_fluor


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

        # Load the data
        with pd.HDFStore(file, mode='r') as h:
            if '/matched_calcium' in h.keys():
                # concatenate the latents
                dataframe = h['matched_calcium']
                dataframe = dataframe.fillna(0)

        dataframe = dataframe.fillna(0)

        ### --- Process visual tuning --- ###

        kinematics, raw_spikes, raw_fluor = parse_kinematic_data(dataframe)

        # Calculate dFF and normalize other neural data
        dff = tuning.calculate_dff(raw_fluor)
        norm_spikes = tuning.normalize_responses(raw_spikes)
        norm_fluor = tuning.normalize_responses(raw_fluor)
        norm_dff = tuning.normalize_responses(dff)

        # Tidy up pitch, yaw, roll
        pitch = -fk.wrap_negative(kinematics.mouse_xrot_m.values)
        kinematics['pitch'] = tuning.smooth_trace(pitch, range=(-180, 180), kernel_size=10, discont=2 * np.pi)

        yaw = fk.wrap_negative(kinematics.mouse_zrot_m.values)
        kinematics['yaw'] = tuning.smooth_trace(yaw, range=(-180, 180), kernel_size=10, discont=2 * np.pi)

        roll = fk.wrap_negative(kinematics.mouse_yrot_m.values)
        kinematics['roll'] = tuning.smooth_trace(roll, range=(-180, 180), kernel_size=10, discont=2 * np.pi)

        # Filter trials by head pitch
        pitch_upper_cutoff = 20
        pitch_lower_cutoff = -90
        view_fraction = 0.7
        kinematics['viewed'] = np.logical_and(kinematics.pitch >= pitch_lower_cutoff,
                                              kinematics.pitch <= pitch_upper_cutoff)
        viewed_trials = kinematics.groupby('trial_num').filter(
            lambda x: (x['viewed'].sum() / len(x['viewed'])) > view_fraction).trial_num.unique()

        raw_spikes_viewed = raw_spikes.loc[raw_spikes.trial_num.isin(viewed_trials)].copy()
        norm_spikes_viewed = norm_spikes.loc[norm_spikes.trial_num.isin(viewed_trials)].copy()
        norm_dff_viewed = norm_dff.loc[norm_dff.trial_num.isin(viewed_trials)].copy()

        # Filter trials by running speed
        if rig == 'VTuningWF':
            speed_column = 'mouse_speed'
        else:
            speed_column = 'wheel_speed'

        speed_cutoff = np.percentile(np.abs(kinematics[speed_column]), 80)
        kinematics['is_running'] = np.abs(kinematics[speed_column]) >= speed_cutoff
        kinematics[f'{speed_column}_abs'] = np.abs(kinematics[speed_column])

        still_trials = kinematics.groupby('trial_num').filter(
            lambda x: x[f'{speed_column}_abs'].mean() < speed_cutoff).trial_num.unique()
        still_trials = viewed_trials[np.in1d(viewed_trials, still_trials)]

        raw_spikes_viewed_still = raw_spikes_viewed.loc[raw_spikes_viewed.trial_num.isin(still_trials)]
        norm_spikes_viewed_still = norm_spikes_viewed.loc[norm_spikes_viewed.trial_num.isin(still_trials)]
        norm_dff_viewed_still = norm_dff_viewed.loc[norm_dff_viewed.trial_num.isin(still_trials)]

        # Run the visual tuning loop
        vis_prop_dict = {}
        for dataset in ['raw_spikes', 'norm_spikes_viewed', 'norm_spikes_viewed_still']:
            for tuning_type in ['direction_wrapped', 'orientation']:
                props = calculate_visual_tuning(dataset, tuning_type, bootstrap_shuffles=500)
                label = tuning_type.split('_')[0]
                vis_prop_dict[label] = props

