import os.path
import pycircstat as circ
from scipy.stats import percentileofscore, sem
import warnings

import functions_kinematic as fk
import functions_tuning as tuning
from snakemake_scripts.tc_calculate import *
from processing_parameters import wf_frame_rate

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)


def parse_trial_frames(df):
    trial_idx_frames = df[df.trial_num > 0].groupby(['trial_num']).apply(lambda x: [x.index[0], x.index[0] +
                                                                                    5*wf_frame_rate])
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


def drop_partial_or_long_trials(df, min_trial_length=4.5, max_trial_length=5.5):
    trial_lengths = df[df.trial_num > 0].groupby('trial_num').apply(lambda x: x.shape[0] / wf_frame_rate)

    # Drop trials that are shorter than min_trial_length (partial trials)
    short_trials = trial_lengths[trial_lengths < min_trial_length].index
    df = df.drop(df[df.trial_num.isin(short_trials)].index)

    # Drop trials that are longer than max_trial_length (errors in trial number indexing)
    long_trials = trial_lengths[trial_lengths > max_trial_length].index
    df = df.drop(df[df.trial_num.isin(long_trials)].index)

    return df


def generate_tcs_and_stats(df):
    _mean, unique_angles = tuning.generate_response_vector(df, np.mean)
    _sem, _ = tuning.generate_response_vector(df, sem, nan_policy='omit')
    _std, _ = tuning.generate_response_vector(df, np.std)
    return _mean, _sem, _std, unique_angles


def calculate_visual_tuning(activity_df, direction_label='direction_wrapped', tuning_fit='von_mises',
                            bootstrap_shuffles=500):
    # --- Parse trials and cells--- #
    # Fill any NaNs in the data with 0
    activity_df = activity_df.fillna(0)
    # Drop trials that are poorly indexed or incomplete
    activity_df = drop_partial_or_long_trials(activity_df)
    # Get the column names for the cells
    cells = [col for col in activity_df.columns if 'cell' in col]

    # --- Threshold data --- #
    # define the clipping threshold in percentile of baseline and do the cell clipping
    clip_threshold = 8
    activity_df[cells] = activity_df.loc[:, cells].apply(clipping_function, axis=1, raw=True, threshold=clip_threshold)

    # -- Mean activity per trial -- #
    # Get the mean direction response per trial and drop the inter-trial interval from df
    mean_direction_activity = activity_df.groupby([direction_label, 'trial_num'])[cells].agg(np.mean).copy().reset_index()
    mean_direction_activity = mean_direction_activity.drop(mean_direction_activity[mean_direction_activity.trial_num == 0].index).sort_values('trial_num')

    # Make sure to explicitly represent 360 degrees in the direction data by duplicating the 0 degrees value
    dup = mean_direction_activity.loc[mean_direction_activity[direction_label] == 0].copy()
    dup.loc[:, direction_label] = 360.
    mean_direction_activity = pd.concat([mean_direction_activity, dup], ignore_index=True)

    mean_orientation_activity = activity_df.groupby(['orientation', 'trial_num'])[cells].agg(np.mean).copy().reset_index()
    mean_orientation_activity = mean_orientation_activity.drop(mean_orientation_activity[mean_orientation_activity.trial_num == 0].index).sort_values('trial_num')

    # Make sure to explicitly represent 180 degrees in the orientation data by duplicating the 0 degrees value
    # dup = mean_orientation_activity.loc[mean_orientation_activity['orientation'] == 0].copy()
    # dup.loc[:, 'orientation'] = 180.
    # mean_orientation_activity = pd.concat([mean_orientation_activity, dup], ignore_index=True)

    # -- Create the response vectors and normalized response vectors --#
    # Normalize the responses of each cell to the maximum mean response of the cell on any given trial
    mean_dir, sem_dir, std_dir, unique_dirs = generate_tcs_and_stats(mean_direction_activity)
    norm_direction_activity = mean_direction_activity.copy()
    norm_direction_activity[cells] = norm_direction_activity[cells].apply(tuning.normalize)
    norm_mean_dir, norm_sem_dir, norm_std_dir, _ = generate_tcs_and_stats(norm_direction_activity)

    mean_ori, sem_ori, std_ori, unique_oris = generate_tcs_and_stats(mean_orientation_activity)
    norm_orientation_activity = mean_orientation_activity.copy()
    norm_orientation_activity[cells] = norm_orientation_activity[cells].apply(tuning.normalize)
    norm_mean_ori, norm_sem_ori, norm_std_ori, _ = generate_tcs_and_stats(norm_orientation_activity)

    # -- Fit tuning curves to get preference-- #
    if tuning_fit == 'von_mises':
        fit_function = tuning.calculate_pref_von_mises
    else:
        fit_function = tuning.calculate_pref_gaussian

    # For all cells
    cell_data_list = []
    for cell in cells:

        # -- 1. Calculate fit and responsivity using all trials -- #
        try:
            mean_guess_dir = np.deg2rad(unique_dirs[np.argmax(norm_mean_dir[cell].fillna(0), axis=0)])
            mean_guess_ori = np.deg2rad(unique_oris[np.argmax(norm_mean_ori[cell].fillna(0), axis=0)])
        except:
            # Sometimes this throws an error if there are no responses
            mean_guess_dir = np.pi
            mean_guess_ori = np.pi/2

        dir_fit, dir_fit_curve, pref_dir, real_pref_dir = \
            fit_function(unique_dirs, norm_mean_dir[cell].to_numpy(), direction_label, mean=mean_guess_dir)

        ori_fit, ori_fit_curve, pref_ori, real_pref_ori = \
            fit_function(unique_oris, norm_mean_ori[cell].to_numpy(), 'orientation', mean=mean_guess_ori)

        dir_gof = tuning.goodness_of_fit(norm_direction_activity[direction_label].to_numpy(),
                                         norm_direction_activity[cell].to_numpy(),
                                         dir_fit_curve[:, 0], dir_fit_curve[:, 1],
                                         type=processing_parameters.gof_type)

        ori_gof = tuning.goodness_of_fit(norm_orientation_activity['orientation'].to_numpy(),
                                         norm_orientation_activity[cell].to_numpy(),
                                         ori_fit_curve[:, 0], ori_fit_curve[:, 1],
                                         type=processing_parameters.gof_type)

        # --- Bootstrap fit and responsivity using all trials -- #
        # Split the data with an 80-20 train-test split. Fit the tuning curve on the training data and calculate
        # goodness of fit on the test data.
        bootstrap_dir_gof, bootstrap_pref_dir, bootstrap_real_pref_dir = \
            tuning.bootstrap_tuning_curve(norm_direction_activity[[direction_label, 'trial_num', cell]], fit_function,
                                          gof_type=processing_parameters.gof_type,
                                          num_shuffles=bootstrap_shuffles, mean=mean_guess_dir)
        p_dir_gof = percentileofscore(bootstrap_dir_gof[~np.isnan(bootstrap_dir_gof)], dir_gof, kind='mean') / 100.

        bootstrap_ori_gof, bootstrap_pref_ori, bootstrap_real_pref_ori = \
            tuning.bootstrap_tuning_curve(norm_orientation_activity[['orientation', 'trial_num', cell]], fit_function,
                                          gof_type=processing_parameters.gof_type,
                                          num_shuffles=bootstrap_shuffles, mean=mean_guess_ori)
        p_ori_gof = percentileofscore(bootstrap_ori_gof[~np.isnan(bootstrap_ori_gof)], ori_gof, kind='mean') / 100.

        # -- 2. Get resultant vector, variance, DSI and OSI using the tuning curves (normalized means) -- #

        # Use the direction dataset first
        theta_dirs = np.deg2rad(unique_dirs)
        dir_magnitudes = norm_mean_dir[cell].copy().to_numpy()

        dsi_nasal_temporal, dsi_abs, osi, resultant_dir_length, resultant_dir, null_dir = \
            tuning.calculate_dsi_osi_resultant(theta_dirs, dir_magnitudes)
        resultant_dir = fk.wrap(np.rad2deg(resultant_dir), bound=360.)
        null_dir = fk.wrap(np.rad2deg(null_dir), bound=360.)
        responsivity_dir = resultant_dir_length

        # For the orientation dataset
        theta_oris = np.deg2rad(unique_oris)
        ori_sep = np.mean(np.diff(theta_oris))
        ori_magnitudes = norm_mean_ori[cell].copy().to_numpy()

        resultant_ori_length, resultant_ori = tuning.resultant_vector(theta_oris, ori_magnitudes, 2)
        resultant_ori = fk.wrap(np.rad2deg(resultant_ori), bound=180.)
        null_ori = fk.wrap(resultant_ori + 90, bound=180.)

        circ_var_ori = circ.var(theta_oris, w=ori_magnitudes, d=ori_sep)
        responsivity_ori = 1 - circ_var_ori

        # -- Run permutation tests using single trial mean responses-- #
        # A. Bootstrap resultant length and angle while guaranteeing the same number of presentations per angle
        #    This is what Joel does for significant shifts in tuning curves
        # B. Shuffle the trial IDs and compare the real selectivity indices to the bootstrapped distribution

        # A. Bootstrap resultant vector while guaranteeing the same number of presentations per angle
        # For direction data
        bootstrap_dsi_nasal_temporal, bootstrap_dsi_abs, bootstrap_osi, bootstrap_resultant_dir, bootstrap_null_dir \
            = tuning.boostrap_dsi_osi_resultant(norm_direction_activity[[direction_label, 'trial_num', cell]],
                                                sampling_method='equal_trial_nums', num_shuffles=bootstrap_shuffles)
        # TODO check nonetype output here
        bootstrap_responsivity_dir = bootstrap_resultant_dir[:, 0]
        p_dsi_nasal_temporal_bootstrap = percentileofscore(bootstrap_dsi_nasal_temporal, dsi_nasal_temporal, kind='mean') / 100.
        p_dsi_abs_bootstrap = percentileofscore(bootstrap_dsi_abs, dsi_abs, kind='mean') / 100.
        p_osi_bootstrap = percentileofscore(bootstrap_osi, osi, kind='mean') / 100.
        p_responsivity_dir_bootstrap = percentileofscore(bootstrap_responsivity_dir, responsivity_dir, kind='mean') / 100.

        # For orientation data
        bootstrap_resultant_ori = \
            tuning.bootstrap_resultant(norm_orientation_activity[['orientation', 'trial_num',  cell]],
                                       sampling_method='equal_trial_nums', multiplier=2, num_shuffles=bootstrap_shuffles)
        bootstrap_responsivity_ori = bootstrap_resultant_ori[:, 0]
        p_responsivity_ori_bootstrap = percentileofscore(bootstrap_responsivity_ori, responsivity_ori, kind='mean') / 100.

        # B. Shuffle the trial IDs and compare the real selectivity indices to the bootstrapped distribution
        # For direction data
        shuffle_dsi_nasal_temporal, shuffle_dsi_abs, shuffle_osi, shuffle_resultant_dir, shuffle_null_dir \
            = tuning.boostrap_dsi_osi_resultant(norm_direction_activity[[direction_label, 'trial_num', cell]],
                                                sampling_method='shuffle_trials', num_shuffles=bootstrap_shuffles)

        p_dsi_nasal_temporal_shuffle = percentileofscore(shuffle_dsi_nasal_temporal, dsi_nasal_temporal, kind='mean') / 100.
        shuffle_responsivity_dir = shuffle_resultant_dir[:, 0]
        p_dsi_abs_shuffle = percentileofscore(shuffle_dsi_abs, dsi_abs, kind='mean') / 100.
        p_osi_shuffle = percentileofscore(shuffle_osi, osi, kind='mean') / 100.
        p_responsivity_dir_shuffle = percentileofscore(shuffle_responsivity_dir, responsivity_dir, kind='mean') / 100.

        # For orientation data
        shuffle_resultant_ori = \
            tuning.bootstrap_resultant(norm_orientation_activity[['orientation', 'trial_num', cell]],
                                       sampling_method='shuffle_trials', multiplier=2,
                                       num_shuffles=bootstrap_shuffles)
        shuffle_responsivity_ori = shuffle_resultant_ori[:, 0]
        p_responsivity_ori_shuffle = percentileofscore(shuffle_responsivity_ori, responsivity_ori, kind='mean') / 100.

        # -- Assemble data for saving -- #
        direction_data = [mean_direction_activity[[direction_label, cell]].to_numpy(),
                          norm_direction_activity[[direction_label, cell]].to_numpy(),
                          np.vstack([unique_dirs, mean_dir[cell].to_numpy()]).T,
                          np.vstack([unique_dirs, norm_mean_dir[cell].to_numpy()]).T,
                          std_dir[cell].to_numpy(), norm_std_dir[cell].to_numpy(),
                          sem_dir[cell].to_numpy(), norm_sem_dir[cell].to_numpy(),
                          dir_fit, dir_fit_curve, dir_gof, bootstrap_dir_gof, p_dir_gof,
                          pref_dir, bootstrap_pref_dir, real_pref_dir, bootstrap_real_pref_dir,
                          (resultant_dir_length, resultant_dir), bootstrap_resultant_dir, shuffle_resultant_dir,

                          responsivity_dir, bootstrap_responsivity_dir, p_responsivity_dir_bootstrap,
                          shuffle_responsivity_dir, p_responsivity_dir_shuffle,

                          null_dir, bootstrap_null_dir, shuffle_null_dir,
                          dsi_nasal_temporal, bootstrap_dsi_nasal_temporal, p_dsi_nasal_temporal_bootstrap,
                          shuffle_dsi_nasal_temporal, p_dsi_nasal_temporal_shuffle,
                          dsi_abs, bootstrap_dsi_abs, p_dsi_abs_bootstrap,
                          shuffle_dsi_abs, p_dsi_abs_shuffle]

        orientation_data = [mean_orientation_activity[['orientation', cell]].to_numpy(),
                            norm_orientation_activity[['orientation', cell]].to_numpy(),
                            np.vstack([unique_oris, mean_ori[cell].to_numpy()]).T,
                            np.vstack([unique_oris, norm_mean_ori[cell].to_numpy()]).T,
                            std_ori[cell].to_numpy(), norm_std_ori[cell].to_numpy(),
                            sem_ori[cell].to_numpy(), norm_sem_ori[cell].to_numpy(),
                            ori_fit, ori_fit_curve, ori_gof, bootstrap_ori_gof, p_ori_gof,
                            pref_ori, null_ori, bootstrap_pref_ori, real_pref_ori, bootstrap_real_pref_ori,
                            (resultant_ori_length, resultant_ori), bootstrap_resultant_ori, shuffle_resultant_ori,

                            responsivity_ori, bootstrap_responsivity_ori, p_responsivity_ori_bootstrap,
                            shuffle_responsivity_ori, p_responsivity_ori_shuffle,

                            osi, bootstrap_osi, p_osi_bootstrap, shuffle_osi, p_osi_shuffle]

        cell_data = direction_data + orientation_data
        cell_data_list.append(cell_data)

    # -- Assemble large dataframe -- #
    direction_columns = ['resp_dir', 'resp_norm_dir',
                         'mean_dir', 'mean_norm_dir',
                         'std_dir', 'std_norm_dir',
                         'sem_dir', 'sem_norm_dir',
                         'fit_dir', 'fit_curve_dir', 'gof_dir', 'bootstrap_gof_dir', 'p_gof_dir',
                         'pref_dir', 'bootstrap_pref_dir', 'real_pref_dir', 'bootstrap_real_pref_dir',
                         'resultant_dir', 'bootstrap_resultant_dir', 'shuffle_resultant_dir',
                         'responsivity_dir', 'bootstrap_responsivity_dir', 'bootstrap_p_responsivity_dir',
                         'shuffle_responsivity_dir', 'shuffle_p_responsivity_dir',
                         'null_dir', 'bootstrap_null_dir', 'shuffle_null_dir',
                         'dsi_nasal_temporal', 'bootstrap_dsi_nasal_temporal', 'bootstrap_p_dsi_nasal_temporal',
                         'shuffle_dsi_nasal_temporal', 'shuffle_p_dsi_nasal_temporal',
                         'dsi_abs', 'bootstrap_dsi_abs', 'bootstrap_p_dsi_abs',
                         'shuffle_dsi_abs', 'shuffle_p_dsi_abs']

    orientation_columns = ['resp_ori', 'resp_norm_ori',
                           'mean_ori', 'mean_norm_ori',
                           'std_ori', 'std_norm_ori',
                           'sem_ori', 'sem_norm_ori',
                           'fit_ori', 'fit_curve_ori', 'gof_ori', 'bootstrap_gof_ori', 'p_gof_ori',
                           'pref_ori', 'null_ori', 'bootstrap_pref_ori', 'real_pref_ori', 'bootstrap_real_pref_ori',
                           'resultant_ori', 'bootstrap_resultant_ori', 'shuffle_resultant_ori',
                           'responsivity_ori', 'bootstrap_responsivity_ori', 'bootstrap_p_responsivity_ori',
                           'shuffle_responsivity_ori', 'shuffle_p_responsivity_ori',
                           'osi', 'bootstrap_osi', 'bootstrap_p_osi', 'shuffle_osi', 'shuffle_p_osi']

    data_cols = direction_columns + orientation_columns
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
        matched_calcium.loc[mask, 'orientation'] = matched_calcium.loc[mask, 'orientation'].apply(fk.wrap, bound=180.1)
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

    # Convert to cm, cm/s or cm/s^2
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


def calculate_kinematic_tuning(df, day, animal, rig):
    # --- Process kinematic tuning --- #
    # This is lifted directly from tc_calculate.py
    # Note here that the kinematic data that was used for the visual tuning is not fed to the kinematic tuning
    # curve calculation below since the formatting is different.

    print('Calculating kinematic tuning curves...')

    # Drop fluorescence columns since not used for this analysis
    fluor_cols = [col for col in df.columns if 'fluor' in col]
    df.drop(columns=fluor_cols, inplace=True)

    # get the number of bins
    bin_num = processing_parameters.bin_number
    shuffle_kind = processing_parameters.tc_shuffle_kind
    percentile = processing_parameters.tc_percentile_cutoff

    # define the pairs to quantify
    if rig in ['VWheel', 'VWheelWF']:
        variable_names = processing_parameters.variable_list_fixed
        df['wheel_speed_abs'] = np.abs(df['wheel_speed'])
    else:
        variable_names = processing_parameters.variable_list_free

    # Convert to cm
    for col in ['wheel_speed', 'wheel_speed_abs', 'wheel_acceleration',
                'mouse_y_m', 'mouse_z_m', 'mouse_x_m',
                'head_height', 'mouse_speed', 'mouse_acceleration']:
        if col in df.columns:
            df[col] = df[col] * 100.
        else:
            pass

    # clip the calcium traces
    clipped_data = clip_calcium([('', df)])

    # parse the features (bin number is for spatial bins in this one)
    features, calcium = parse_features(clipped_data, variable_names, bin_number=processing_parameters.spatial_bins)

    # concatenate all the trials
    features = pd.concat(features)
    calcium = np.concatenate(calcium)

    # get the number of cells
    cell_num = calcium.shape[1]

    # get the TCs and their responsivity
    tcs_half, tcs_full, tcs_resp, tc_count, tc_bins = \
        extract_tcs_responsivity(features, calcium, variable_names, cell_num,
                                 percentile=percentile, bin_number=bin_num, shuffle_kind=shuffle_kind)

    # get the TC consistency
    tcs_cons = extract_consistency(tcs_half, variable_names, cell_num, percentile=80)

    # convert the outputs into a dataframe
    tcs_dict, tcs_counts_dict, tcs_bins_dict = convert_to_dataframe(tcs_half, tcs_full, tc_count, tcs_resp,
                                                        tcs_cons, tc_bins, day, animal, rig)

    return tcs_dict, tcs_counts_dict, tcs_bins_dict


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

            # Run the visual tuning loop and save to file
            print('Calculating visual tuning curves...')
            vis_prop_dict = {}
            for ds_name in processing_parameters.activity_datasets:
                activity_ds = activity_ds_dict[ds_name]
                props = calculate_visual_tuning(activity_ds, bootstrap_shuffles=processing_parameters.bootstrap_repeats)
                # Save visual features to hdf5 file
                props.to_hdf(out_file, f'{ds_name}_props')

            # --- Process kinematic tuning --- #
            tcs_dict, tcs_counts_dict, tcs_bins_dict = calculate_kinematic_tuning(dataframe, day, animal, rig)

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

