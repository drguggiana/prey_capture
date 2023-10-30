import os.path
import pycircstat as circ
from scipy.stats import percentileofscore, sem
import warnings

import functions_kinematic as fk
import functions_tuning as tuning
from tc_calculate import *
from processing_parameters import wf_frame_rate

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)


def parse_trial_frames(df):
    trial_idx_frames = df[df.direction > -1000].groupby(['trial_num']).apply(lambda x: [x.index[0], x.index[0] + 5*wf_frame_rate])
    trial_idx_frames = np.array(trial_idx_frames.to_list())

    if trial_idx_frames[-1, 1] > df.index[-1]:
        trial_idx_frames[-1, 1] = df.index[-1]
    if trial_idx_frames[0, 0] < 0:
        trial_idx_frames[0, 0] = 0

    traces = []
    for i, frame in enumerate(trial_idx_frames):
        df_slice = df.iloc[frame[0]:frame[1], :].copy()
        df_slice['trial_num'] = i + 1
        traces.append(df_slice)
    
    traces = pd.concat(traces, axis=0).reset_index(drop=True)
    return traces



def calculate_visual_tuning(activity_df, tuning_kind, tuning_fit='von_mises', bootstrap_shuffles=500):

    # --- Threshold data and parse trials --- #
    # define the clipping threshold in percentile of baseline
    clip_threshold = 8
    # do the cell clipping
    cells = [col for col in activity_df.columns if 'cell' in col]
    activity_df[cells] = activity_df.loc[:, cells].apply(clipping_function, axis=1, raw=True,
                                                         threshold=clip_threshold)

    # Correctly parse trials since the data has some errors
    activity_df = parse_trial_frames(activity_df)

    # Get the mean response per trial and drop the inter-trial interval from df
    mean_trial_activity = activity_df.groupby([tuning_kind, 'trial_num'])[cells].agg(np.nanmean).reset_index()
    mean_trial_activity = mean_trial_activity.drop(mean_trial_activity[mean_trial_activity[tuning_kind] == -1000].index).sort_values('trial_num')

    # Make sure to explicitly represent 360 degrees in the data by duplicating the 0 degrees value
    if 'direction' in tuning_kind:
        dup_angle = 360.
        dup = mean_trial_activity.loc[mean_trial_activity[tuning_kind] == 0].copy()
        dup.loc[:, tuning_kind] = dup_angle
        mean_trial_activity = pd.concat([mean_trial_activity, dup], ignore_index=True)

    # -- Create the response vectors --#
    mean_resp, unique_angles = tuning.generate_response_vector(mean_trial_activity, np.nanmean)
    sem_resp, _ = tuning.generate_response_vector(mean_trial_activity, sem, nan_policy='omit')
    std_resp, _ = tuning.generate_response_vector(mean_trial_activity, np.nanstd)

    # Normalize the responses of each cell to the maximum mean response of the cell on any given trial
    norm_trial_activity = mean_trial_activity.copy()
    norm_trial_activity[cells] = norm_trial_activity[cells].apply(tuning.normalize)
    norm_mean_resp, _ = tuning.generate_response_vector(norm_trial_activity, np.nanmean)
    norm_sem_resp, _ = tuning.generate_response_vector(norm_trial_activity, sem, nan_policy='omit')
    norm_std_resp, _ = tuning.generate_response_vector(norm_trial_activity, np.nanstd)

    # -- Fit tuning curves to get preference-- #
    if tuning_fit == 'von_mises':
        fit_function = tuning.calculate_pref_von_mises
    else:
        fit_function = tuning.calculate_pref_gaussian

    # For all cells
    cell_data_list = []
    for cell in cells:
        # -- Calculate fit and responsivity using all trials -- #
        try:
            mean_guess = unique_angles[np.argmax(norm_mean_resp[cell].fillna(0), axis=0)]
        except:
            # Sometimes this throws an error if there are no responses
            mean_guess = unique_angles[len(unique_angles) // 2]

        fit, fit_curve, pref_angle, real_pref_angle = \
            fit_function(unique_angles, norm_mean_resp[cell].to_numpy(), tuning_kind, mean=mean_guess)

        fit_gof = tuning.goodness_of_fit(norm_trial_activity[tuning_kind].to_numpy(),
                                         norm_trial_activity[cell].to_numpy(),
                                         fit_curve[:, 0], fit_curve[:, 1],
                                         type=processing_parameters.gof_type)

        # -- Get resultant vector and response variance using the tuning curve (trial response means) -- #
        if 'direction' in tuning_kind:
            multiplier = 1.
        else:
            multiplier = 2.

        thetas = np.deg2rad(unique_angles)
        theta_sep = np.mean(np.diff(thetas))
        magnitudes = norm_mean_resp[cell].copy().to_numpy()

        if 'direction' in tuning_kind:
            dsi_nasal_temporal, dsi_abs, osi, resultant_length, resultant_angle, null_angle = \
                tuning.calculate_dsi_osi_resultant(thetas, magnitudes)
            resultant_angle = fk.wrap(np.rad2deg(resultant_angle), bound=360/multiplier)
            null_angle = fk.wrap(np.rad2deg(null_angle), bound=360/multiplier)

        else:
            resultant_length = circ.resultant_vector_length(thetas, w=magnitudes, d=theta_sep, axial_correction=multiplier)
            resultant_angle = circ.mean(thetas, w=magnitudes, d=theta_sep, axial_correction=multiplier)
            resultant_angle = fk.wrap(np.rad2deg(resultant_angle), bound=360/multiplier)

            dsi_nasal_temporal = np.nan
            dsi_abs = np.nan
            osi = np.nan
            null_angle = np.nan

        circ_var = circ.var(thetas, w=magnitudes, d=theta_sep)
        responsivity = 1 - circ_var

        # -- Run permutation tests using single trial mean responses-- #
        # 1. Shuffle the trial IDs and compare the real selectivity indices to the bootstrapped distribution
        # 2. Bootstrap resultant length and angle while guaranteeing the same number of presentations per angle
        #    This is what Joel does for significant shifts in tuning curves 
        # 3. Calculate regression fit on a subset of data and compare to bootstrapped distribution

        if 'direction' in tuning_kind:
            # 1. Shuffle trial IDs
            bootstrap_dsi_nasal_temporal, bootstrap_dsi_abs, bootstrap_osi, bootstrap_responsivity, bootstrap_null_angle= \
                tuning.boostrap_dsi_osi_resultant(norm_trial_activity[[tuning_kind, 'trial_num', cell]],
                                                  sampling_method='shuffle_trials', num_shuffles=bootstrap_shuffles)

            bootstrap_responsivity = bootstrap_responsivity[:, 0]
            p_dsi_nasal_temporal = percentileofscore(bootstrap_dsi_nasal_temporal, dsi_nasal_temporal, kind='mean') / 100.
            p_dsi_abs = percentileofscore(bootstrap_dsi_abs, dsi_abs, kind='mean') / 100.
            p_osi = percentileofscore(bootstrap_osi, osi, kind='mean') / 100.
            p_res = percentileofscore(bootstrap_responsivity, responsivity, kind='mean') / 100.

            # 2. Bootstrap resultant vector
            _, _, _, bootstrap_resultant, _ = \
                tuning.boostrap_dsi_osi_resultant(norm_trial_activity[[tuning_kind, 'trial_num', cell]],
                                                  sampling_method='equal_trial_nums', num_shuffles=bootstrap_shuffles)

        else:
            # 1. Shuffle trial IDs
            bootstrap_responsivity = tuning.bootstrap_responsivity(thetas, magnitudes, multiplier, 
                                                                   num_shuffles=bootstrap_shuffles)
            p_res = percentileofscore(bootstrap_responsivity, responsivity, kind='mean') / 100.

            # 2. Bootstrap resultant vector
            bootstrap_resultant = tuning.bootstrap_resultant(norm_trial_activity[[tuning_kind, 'trial_num', cell]],
                                                            multiplier=multiplier, num_shuffles=bootstrap_shuffles)

            bootstrap_dsi_nasal_temporal = np.nan
            bootstrap_dsi_abs = np.nan
            bootstrap_osi = np.nan
            p_dsi_nasal_temporal = np.nan
            p_dsi_abs = np.nan
            p_osi = np.nan

        # 3. Calculate fit on subset of data
        bootstrap_gof, bootstrap_pref_angle, bootstrap_real_pref = \
            tuning.bootstrap_tuning_curve(norm_trial_activity[[tuning_kind, 'trial_num', cell]], fit_function,
                                          gof_type=processing_parameters.gof_type,
                                          num_shuffles=bootstrap_shuffles, mean=mean_guess)
        p_gof = percentileofscore(bootstrap_gof[~np.isnan(bootstrap_gof)], fit_gof, kind='mean') / 100.

        # -- Assemble data for saving -- #
        cell_data = [mean_trial_activity[[tuning_kind, cell]].to_numpy(),
                     norm_trial_activity[[tuning_kind, cell]].to_numpy(),
                     np.vstack([unique_angles, mean_resp[cell].to_numpy()]).T,
                     np.vstack([unique_angles, norm_mean_resp[cell].to_numpy()]).T,
                     std_resp[cell].to_numpy(), norm_std_resp[cell].to_numpy(),
                     sem_resp[cell].to_numpy(), norm_sem_resp[cell].to_numpy(),
                     fit, fit_curve, fit_gof, bootstrap_gof, p_gof,
                     pref_angle, bootstrap_pref_angle, real_pref_angle, bootstrap_real_pref,
                     (resultant_length, resultant_angle), bootstrap_resultant,
                     circ_var, responsivity, bootstrap_responsivity, p_res,
                     dsi_nasal_temporal, bootstrap_dsi_nasal_temporal, p_dsi_nasal_temporal,
                     dsi_abs, bootstrap_dsi_abs, p_dsi_abs,
                     osi, bootstrap_osi, p_osi]

        cell_data_list.append(cell_data)

    # -- Assemble large dataframe -- #
    data_cols = ['trial_resp', 'trial_resp_norm',
                 'mean', 'mean_norm', 'std', 'std_norm', 'sem', 'sem_norm',
                 'fit', 'fit_curve', 'gof', 'bootstrap_gof', 'p_gof',
                 'pref', 'bootstrap_pref', 'shown_pref', 'bootstrap_shown_pref',
                 'resultant', 'bootstrap_resultant',
                 'circ_var', 'responsivity', 'bootstrap_responsivity', 'p_responsivity',
                 'dsi_nasal_temporal', 'bootstrap_dsi_nasal_temporal', 'p_dsi_nasal_temporal',
                 'dsi_abs', 'bootstrap_dsi_abs', 'p_dsi_abs',
                 'osi', 'bootstrap_osi', 'p_osi']

    data_df = pd.DataFrame(index=cells, columns=data_cols, data=cell_data_list)
    return data_df


def parse_kinematic_data(matched_calcium, rig):
    # For all data
    if rig in ['VWheel', 'VWheelWF']:
        exp_type = 'fixed'
    else:
        exp_type = 'free'

    # Apply wrapping for directions to get range [0, 360]
    matched_calcium['direction_wrapped'] = matched_calcium['direction'].copy()
    mask = matched_calcium['direction_wrapped'] > -1000
    matched_calcium.loc[mask, 'direction_wrapped'] = matched_calcium.loc[mask, 'direction_wrapped'].apply(fk.wrap)

    # Now find the direction relative to the ground plane
    if exp_type == 'free':
        matched_calcium['direction_rel_ground'] = matched_calcium['direction_wrapped'].copy()
        matched_calcium.loc[mask, 'direction_rel_ground'] = matched_calcium.loc[mask, 'direction_rel_ground'] + \
                                                            matched_calcium.loc[mask, 'head_roll']
    else:
        matched_calcium['direction_rel_ground'] = matched_calcium['direction_wrapped'].copy()

    # Calculate orientation explicitly
    if 'orientation' not in matched_calcium.columns:
        matched_calcium['orientation'] = matched_calcium['direction_wrapped'].copy()
        matched_calcium['orientation_rel_ground'] = matched_calcium['direction_rel_ground'].copy()
        mask = matched_calcium['orientation'] > -1000
        matched_calcium.loc[mask, 'orientation'] = matched_calcium.loc[mask, 'orientation'].apply(fk.wrap, bound=180)
        matched_calcium.loc[mask, 'orientation_rel_ground'] = matched_calcium.loc[mask, 'orientation_rel_ground'].apply(fk.wrap, bound=180)

    spikes_cols = [key for key in matched_calcium.keys() if 'spikes' in key]
    fluor_cols = [key for key in matched_calcium.keys() if 'fluor' in key]
    motive_tracking_cols = ['mouse_y_m', 'mouse_z_m', 'mouse_x_m', 'mouse_yrot_m', 'mouse_zrot_m', 'mouse_xrot_m']

    # If there is more than one spatial or temporal frequency, include it, othewise don't
    stimulus_cols = ['trial_num', 'time_vector', 'direction', 'direction_wrapped', 'direction_rel_ground',
                     'orientation', 'orientation_rel_ground', 'grating_phase']

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

    # Convert to cm
    for col in ['wheel_speed', 'wheel_acceleration', 'mouse_y_m', 'mouse_z_m', 'mouse_x_m', 'head_height',
                'mouse_speed', 'mouse_acceleration']:
        if col in kinematics.columns:
            kinematics[col] = kinematics[col] * 100.
        else:
            pass

    raw_spikes = matched_calcium.loc[:, stimulus_cols + spikes_cols]
    raw_spikes.columns = [key.rsplit('_', 1)[0] if 'spikes' in key else key for key in raw_spikes.columns]
    raw_fluor = matched_calcium.loc[:, stimulus_cols + fluor_cols]
    raw_fluor.columns = [key.rsplit('_', 1)[0] if 'fluor' in key else key for key in raw_fluor.columns]

    return kinematics, raw_spikes, raw_fluor


if __name__ == '__main__':
    # get the data paths
    try:
        input_path = list(snakemake.input)
        # get the slugs
        slug_list = [os.path.basename(el).replace('_preproc.hdf5', '') for el in input_path]
        # read the output path and the input file urls
        out_path = [snakemake.output[0]]
        data_all = [yaml.load(snakemake.params.file_info, Loader=yaml.FullLoader)]
        # get the parts for the file naming
        rigs = [d['rig'] for d in data_all]
        animals = [slug.split('_')[7:10] for slug in slug_list]
        animals = ['_'.join([animal[0].upper()] + animal[1:]) for animal in animals]
        days = [slug[:10] for slug in slug_list]

    except NameError:

        # get the paths from the database
        data_all = bd.query_database('analyzed_data', processing_parameters.search_string)
        data_all = [el for el in data_all if '_preproc' in el['slug']]
        input_path = [el['analysis_path'] for el in data_all]
        out_path = [os.path.join(paths.analysis_path, os.path.basename(path).replace('preproc', 'tcday')) for
                    path in input_path]
        # get the day, animal and rig
        days = ['_'.join(d['slug'].split('_')[0:3]) for d in data_all]
        rigs = [el['rig'] for el in data_all]
        animals = [d['slug'].split('_')[7:10] for d in data_all]
        animals = ['_'.join([animal[0].upper()] + animal[1:]) for animal in animals]

    for idx, (in_file, out_file) in enumerate(zip(input_path, out_path)):
        # allocate memory for the data
        raw_data = []
        rig = rigs[idx]
        animal = animals[idx]
        day = days[idx]

        # Load the data
        with pd.HDFStore(in_file, mode='r') as h:

            if '/cell_matches' in h.keys():
                # concatenate the latents
                cell_matches = h['cell_matches']

            if '/matched_calcium' in h.keys():
                # concatenate the latents
                dataframe = h['matched_calcium']

                # store
                raw_data.append((in_file, dataframe.fillna(0)))

        # skip processing if the file is empty
        if len(raw_data) == 0:
            # save an empty file and end
            empty = pd.DataFrame([])
            empty.to_hdf(out_file, 'no_ROIs')

        else:
            # --- Process visual tuning --- #
            kinematics, raw_spikes, raw_fluor = parse_kinematic_data(raw_data[0][-1], rig)

            # Calculate dFF and normalize other neural data
            activity_ds_dict = {}
            dff = tuning.calculate_dff(raw_fluor)
            norm_spikes = tuning.normalize_responses(raw_spikes)
            norm_fluor = tuning.normalize_responses(raw_fluor)
            norm_dff = tuning.normalize_responses(dff)
            activity_ds_dict['dff'] = dff
            activity_ds_dict['norm_spikes'] = norm_spikes
            activity_ds_dict['norm_fluor'] = norm_fluor
            activity_ds_dict['norm_dff'] = norm_dff

            # Filter trials by head pitch if freely moving
            if rig in ['VTuningWF', 'VTuning']:
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
            else:
                viewed_trials = raw_spikes.trial_num.unique()
                raw_spikes_viewed = raw_spikes.copy()
                norm_spikes_viewed = norm_spikes.copy()
                norm_dff_viewed = norm_dff.copy()

            activity_ds_dict['raw_spikes_viewed'] = raw_spikes_viewed
            activity_ds_dict['norm_spikes_viewed'] = norm_spikes_viewed
            activity_ds_dict['norm_dff_viewed'] = norm_dff_viewed

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

            activity_ds_dict['raw_spikes_viewed_still'] = raw_spikes_viewed_still
            activity_ds_dict['norm_spikes_viewed_still'] = norm_spikes_viewed_still
            activity_ds_dict['norm_dff_viewed_still'] = norm_dff_viewed_still

            # Run the visual tuning loop
            print('Calculating visual tuning curves...')
            vis_prop_dict = {}
            for ds_name in processing_parameters.activity_datasets:
                activity_ds = activity_ds_dict[ds_name]
                for tuning_type in ['direction_wrapped', 'orientation']:
                    props = calculate_visual_tuning(activity_ds, tuning_type,
                                                    bootstrap_shuffles=processing_parameters.bootstrap_repeats)
                    label = tuning_type.split('_')[0]
                    vis_prop_dict[f'{ds_name}_{label}_props'] = props

            # Save visual features to hdf5 file
            for key in vis_prop_dict.keys():
                vis_prop_dict[key].to_hdf(out_file, key)

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
                dataframe['wheel_speed_abs'] = np.abs(dataframe['wheel_speed'])
            else:
                variable_names = processing_parameters.variable_list_free

            # Convert to cm
            for col in ['wheel_speed', 'wheel_speed_abs', 'wheel_acceleration', 'mouse_y_m', 'mouse_z_m', 'mouse_x_m',
                        'head_height', 'mouse_speed', 'mouse_acceleration']:
                if col in dataframe.columns:
                    dataframe[col] = dataframe[col] * 100.
                else:
                    pass

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
            tcs_dict, tcs_counts_dict, tcs_bins_dict = convert_to_dataframe(tcs_half, tcs_full, tc_count, tcs_resp,
                                                                tcs_cons, tc_bins, day, animal, rig)

            # for all the features
            for feature in tcs_dict.keys():
                tcs_dict[feature].to_hdf(out_file, feature)
                tcs_counts_dict[feature].to_hdf(out_file, feature + '_counts')
                tcs_bins_dict[feature].to_hdf(out_file, feature + '_edges')

            # --- save the metadata --- #
            cell_matches.to_hdf(out_file, 'cell_matches')
            # meta_data = pd.DataFrame(np.vstack(meta_list), columns=processing_parameters.meta_fields)
            # meta_data.to_hdf(out_path, 'meta_data')

        # save as a new entry to the data base
        # assemble the entry data

        entry_data = {
            'analysis_type': 'tc_analysis',
            'analysis_path': out_file,
            'date': '',
            'pic_path': '',
            'result': str(data_all[idx]['result']),
            'rig': str(data_all[idx]['rig']),
            'lighting': str(data_all[idx]['lighting']),
            'imaging': 'wirefree',
            'slug': fm.slugify(os.path.basename(out_file)[:-5]),

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

