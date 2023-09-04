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
        ori = matched_calcium['direction'].to_numpy()
        ori[(ori > -180) & (ori < 0)] += 180
        matched_calcium['orientation'] = ori

    # Apply wrapping for directions to get range [0, 360]
    matched_calcium['direction_wrapped'] = matched_calcium['direction']
    mask = matched_calcium['direction_wrapped'] > -1000
    matched_calcium.loc[mask, 'direction_wrapped'] = matched_calcium.loc[mask, 'direction_wrapped'].apply(fk.wrap)

    # For all data
    if rig in ['VWheel', 'VWheelWF']:
        exp_type = 'fixed'
    else:
        exp_type = 'free'

    spikes_cols = [key for key in matched_calcium.keys() if 'spikes' in key]
    fluor_cols = [key for key in matched_calcium.keys() if 'fluor' in key]
    motive_tracking_cols = ['mouse_y_m', 'mouse_z_m', 'mouse_x_m', 'mouse_yrot_m', 'mouse_zrot_m', 'mouse_xrot_m']

    # If there is more than one spatial or temporal frequency, include it, othewise don't
    stimulus_cols = ['trial_num', 'time_vector', 'direction', 'direction_wrapped', 'orientation', 'grating_phase']

    # For headfixed data
    eye_cols = ['eye_horizontal_vector_x', 'eye_horizontal_vector_y', 'eye_midpoint_x', 'eye_midpoint_y',
                'pupil_center_ref_x', 'pupil_center_ref_y', 'fit_pupil_center_x', 'fit_pupil_center_y',
                'pupil_diameter', 'minor_axis', 'pupil_rotation', 'eyelid_angle']
    wheel_cols = ['wheel_speed', 'wheel_acceleration']

    # For free data
    mouse_kinem_cols = ['mouse_heading', 'mouse_angular_speed', 'mouse_speed', 'mouse_acceleration', 'head_direction',
                        'head_height', 'head_pitch', 'head_yaw', 'head_roll']
    mouse_dlc_cols = ['mouse_snout_x', 'mouse_snout_y', 'mouse_barl_x', 'mouse_barl_y', 'mouse_barr_x', 'mouse_barr_y',
                      'mouse_x', 'mouse_y', 'mouse_body2_x', 'mouse_body2_y', 'mouse_body3_x', 'mouse_body3_y',
                      'mouse_base_x', 'mouse_base_y', 'mouse_head_x', 'mouse_head_y', 'miniscope_top_x',
                      'miniscope_top_y']

    if exp_type == 'fixed':
        kinematics = matched_calcium.loc[:, stimulus_cols + motive_tracking_cols + eye_cols + wheel_cols]
    else:
        kinematics = matched_calcium.loc[:, stimulus_cols + motive_tracking_cols + mouse_kinem_cols]

    raw_spikes = matched_calcium.loc[:, stimulus_cols + spikes_cols]
    raw_spikes.columns = [key.rsplit('_', 1)[0] if 'spikes' in key else key for key in raw_spikes.columns]
    raw_fluor = matched_calcium.loc[:, stimulus_cols + fluor_cols]
    raw_fluor.columns = [key.rsplit('_', 1)[0] if 'fluor' in key else key for key in raw_fluor.columns]

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

    # allocate memory for the data
    raw_data = []

    for idx, file in enumerate(input_path):

        # Load the data
        with pd.HDFStore(file, mode='r') as h:

            if '/matched_calcium' in h.keys():
                # concatenate the latents
                dataframe = h['matched_calcium']

            if '/cell_matches' in h.keys():
                # concatenate the latents
                cell_matches = h['cell_matches']

            # store
            raw_data.append((file, dataframe))

        # skip processing if the file is empty
        if len(raw_data) == 0:
            # save an empty file and end
            empty = pd.DataFrame([])
            empty.to_hdf(out_path, 'no_ROIs')

        else:

            dataframe = raw_data[0][-1]
            dataframe = dataframe.fillna(0)

            # --- Process visual tuning --- #
            kinematics, raw_spikes, raw_fluor = parse_kinematic_data(dataframe)

            # Calculate dFF and normalize other neural data
            actvity_ds_dict = {}
            dff = tuning.calculate_dff(raw_fluor)
            norm_spikes = tuning.normalize_responses(raw_spikes)
            norm_fluor = tuning.normalize_responses(raw_fluor)
            norm_dff = tuning.normalize_responses(dff)
            actvity_ds_dict['dff'] = dff
            actvity_ds_dict['norm_spikes'] = norm_spikes
            actvity_ds_dict['norm_fluor'] = norm_fluor
            actvity_ds_dict['norm_dff'] = norm_dff

            # Filter trials by head pitch
            pitch_lower_cutoff = processing_parameters.head_pitch_cutoff[0]
            pitch_upper_cutoff = processing_parameters.head_pitch_cutoff[1]
            view_fraction = processing_parameters.view_fraction
            kinematics['viewed'] = np.logical_and(kinematics['head_pitch'].to_numpy() >= pitch_lower_cutoff,
                                                  kinematics['head_pitch'].to_numpy() <= pitch_upper_cutoff)
            viewed_trials = kinematics.groupby('trial_num').filter(
                lambda x: (x['viewed'].sum() / len(x['viewed'])) > view_fraction).trial_num.unique()

            raw_spikes_viewed = raw_spikes.loc[raw_spikes.trial_num.isin(viewed_trials)].copy()
            norm_spikes_viewed = norm_spikes.loc[norm_spikes.trial_num.isin(viewed_trials)].copy()
            norm_dff_viewed = norm_dff.loc[norm_dff.trial_num.isin(viewed_trials)].copy()

            actvity_ds_dict['raw_spikes_viewed'] = raw_spikes_viewed
            actvity_ds_dict['norm_spikes_viewed'] = norm_spikes_viewed
            actvity_ds_dict['norm_dff_viewed'] = norm_dff_viewed

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

            actvity_ds_dict['raw_spikes_viewed_still'] = raw_spikes_viewed_still
            actvity_ds_dict['norm_spikes_viewed_still'] = norm_spikes_viewed_still
            actvity_ds_dict['norm_dff_viewed_still'] = norm_dff_viewed_still

            # Run the visual tuning loop
            print('Calculating visual tuning curves...')
            vis_prop_dict = {}
            for ds_name in processing_parameters.activity_datasets:
                activity_ds = actvity_ds_dict[ds_name]
                for tuning_type in ['direction_wrapped', 'orientation']:
                    props = calculate_visual_tuning(activity_ds, tuning_type,
                                                    bootstrap_shuffles=processing_parameters.bootstrap_repeats)
                    label = tuning_type.split('_')[0]
                    vis_prop_dict[f'{ds_name}_{label}_props'] = props

            # Save visual features to hdf5 file
            for key in vis_prop_dict.keys():
                vis_prop_dict[key].to_hdf(out_path, key)

            # --- Process kinematic tuning --- #
            # This is lifted directly from tc_calculate.py
            print('Calculating kinematic tuning curves...')

            # Drop fluorescence columns since not used for this analysis
            fluor_cols = [col for col in dataframe.columns if 'fluor' in col]
            dataframe.drop(columns=fluor_cols, inplace=True)

            # get the number of bins
            bin_num = processing_parameters.bin_number
            # define the pairs to quantify
            if rig in ['VWheel', 'VWheelWF']:
                variable_names = processing_parameters.variable_list_fixed
            else:
                variable_names = processing_parameters.variable_list_free

            # clip the calcium traces
            clipped_data = clip_calcium([('', dataframe)])

            # parse the features (bin number is for spatial bins in this one)
            features, calcium = parse_features(clipped_data, variable_names, bin_number=20)

            # concatenate all the trials
            features = pd.concat(features)
            calcium = np.concatenate(calcium)

            # get the number of cells
            cell_num = calcium.shape[1]

            # get the TCs and their responsivity
            tcs_half, tcs_full, tcs_resp, tc_count, tc_bins = \
                extract_tcs_responsivity(features, calcium, variable_names, cell_num, percentile=80, bin_number=bin_num)
            # get the TC consistency
            tcs_cons = extract_consistency(tcs_half, variable_names, cell_num, percentile=80)
            # # get the tc quality
            # tcs_qual = extract_quality(tcs_full, features)
            # convert the outputs into a dataframe
            tcs_dict, tcs_counts_dict, _ = convert_to_dataframe(tcs_half, tcs_full, tc_count, tcs_resp,
                                                                tcs_cons, tc_bins, day, animal, rig)

            # for all the features
            for feature in tcs_dict.keys():
                tcs_dict[feature].to_hdf(out_path, feature)
                tcs_counts_dict[feature].to_hdf(out_path, feature + '_counts')
                # tcs_bins_dict[feature].to_hdf(out_path, feature + '_edges')

            # --- save the metadata --- #
            cell_matches.to_hdf(out_path, 'cell_matches')
            # meta_data = pd.DataFrame(np.vstack(meta_list), columns=processing_parameters.meta_fields)
            # meta_data.to_hdf(out_path, 'meta_data')

        # save as a new entry to the data base
        # assemble the entry data

        entry_data = {
            'analysis_type': 'tc_analysis',
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

