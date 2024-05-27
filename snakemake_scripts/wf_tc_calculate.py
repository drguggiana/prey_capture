import warnings

import pandas as pd
import pycircstat as circ
from hmmlearn import hmm
from scipy.stats import percentileofscore, sem, t, mannwhitneyu

import processing_parameters
import functions_kinematic as fk
import functions_tuning as tuning
from snakemake_scripts.tc_calculate import *

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)


def drop_partial_or_long_trials(df, min_trial_length=4.5, max_trial_length=5.5):
    """
    This function drops trials that are shorter than min_trial_length (partial trials) and
    trials that are longer than max_trial_length (errors in trial number indexing) from the dataframe.

    Parameters:
    df (DataFrame): The dataframe containing the trials.
    min_trial_length (float): The minimum length for a trial. Defaults to 4.5.
    max_trial_length (float): The maximum length for a trial. Defaults to 5.5.

    Returns:
    DataFrame: The dataframe after dropping the partial and long trials.
    """

    trial_lengths = df[df.trial_num > 0].groupby('trial_num').apply(lambda x: x.shape[0] / processing_parameters.wf_frame_rate)

    # Drop trials that are shorter than min_trial_length (partial trials)
    short_trials = trial_lengths[trial_lengths < min_trial_length].index
    df = df.drop(df[df.trial_num.isin(short_trials)].index)

    # Drop trials that are longer than max_trial_length (errors in trial number indexing)
    long_trials = trial_lengths[trial_lengths > max_trial_length].index
    df = df.drop(df[df.trial_num.isin(long_trials)].index).reset_index(drop=True)

    return df


def generate_tcs_and_stats(df):
    """
    This function generates tuning curves and statistics for the given dataframe.

    Parameters:
    df (DataFrame): The dataframe for which to generate tuning curves and statistics.

    Returns:
    Tuple: A tuple containing the mean, standard error of the mean, standard deviation, and unique angles.
    """

    _mean, unique_angles = tuning.generate_response_vector(df, np.mean)
    _sem, _ = tuning.generate_response_vector(df, sem, nan_policy='omit')
    _std, _ = tuning.generate_response_vector(df, np.std)
    return _mean, _sem, _std, unique_angles


def calculate_response_ratios(trial_activity, iti_activity, cells):
    """
    This function calculates the ratio of activity during the trial to the activity during the ITI.

    Parameters:
    trial_activity (DataFrame): The dataframe containing the trial activity.
    iti_activity (DataFrame): The dataframe containing the ITI activity.
    cells (List[str]): The list of cells.

    Returns:
    DataFrame: The dataframe containing the ratio of activity during the trial to the activity during the ITI.
    """

    ratio_activity = trial_activity[cells] / iti_activity[cells]
    ratio_activity.fillna(0, inplace=True)
    mask = np.isinf(ratio_activity)
    ratio_activity[mask] = trial_activity[mask]
    return ratio_activity


def calculate_response_diff(trial_activity, iti_activity, cells):
    """
    This function calculates the difference of activity during the trial to the activity during the ITI.

    Parameters:
    trial_activity (DataFrame): The dataframe containing the trial activity.
    iti_activity (DataFrame): The dataframe containing the ITI activity.
    cells (List[str]): The list of cells.

    Returns:
    DataFrame: The dataframe containing the difference of activity during the trial to the activity during the ITI.
    """

    diff_activity = trial_activity[cells] - iti_activity[cells]
    diff_activity.fillna(0, inplace=True)
    mask = np.isinf(diff_activity)
    diff_activity[mask] = trial_activity[mask]
    return diff_activity


def calculate_vis_ori_dir_responsivity(trial_activity, iti_mean_activity, iti_std_activity, num_std=6):
    
    """
    Calculate the visual, direction, and orientation responsivity of cells based on trial activity.

    Args:
        trial_activity (DataFrame): DataFrame containing trial activity data.
        iti_mean_activity (DataFrame): DataFrame containing ITI mean activity data.
        iti_std_activity (DataFrame): DataFrame containing ITI standard deviation activity data.
        num_std (int, optional): Number of standard deviations above the ITI mean to consider a cell visually responsive. Defaults to 6.

    Returns:
        tuple: A tuple containing three boolean arrays indicating the visual, direction, and orientation responsivity of cells.
    """

    cells = [col for col in trial_activity.columns if 'cell' in col]

    # Determine if cell is visually responsive
    #    A cell is responsive if during a trial its max response during that trial is greater than 6 stds above the
    #    ITI mean. To be considered visually responsive, a cell must respond during at least 50% of the trials

    vis_trial_resps = ((trial_activity[cells] - iti_mean_activity[cells]) >=
                       (iti_mean_activity[cells] + num_std * iti_std_activity[cells]))
    vis_trial_resps.insert(0, 'trial_num', trial_activity['trial_num'])
    vis_trial_resps.insert(1, 'direction_wrapped', trial_activity['direction_wrapped'])
    vis_trial_resps.insert(2, 'orientation', trial_activity['orientation'])
    is_vis_responsive = vis_trial_resps[cells].apply(lambda x: x.sum() >= np.ceil(x.count() / 2))

    # Determine if a cell is direction responsive
    #    To be considered direction responsive, a cell must respond to at least 50% of the trials for at least one
    #    direction stimulus
    is_direction_responsive = (vis_trial_resps.groupby(['direction_wrapped'])[cells].agg(list)
                               .applymap(lambda x: np.sum(x) >= np.ceil(len(x) / 2)))
    is_direction_responsive = is_direction_responsive[cells].apply(lambda x: x.sum() > 0)

    # Determine if a cell is orientation responsive
    #    To be considered direction responsive, a cell must respond to at least 50% of the trials for at least one
    #    orientation stimulus
    is_orientation_responsive = (vis_trial_resps.groupby(['orientation'])[cells].agg(list)
                                 .applymap(lambda x: np.sum(x) >= np.ceil(len(x) / 2)))
    is_orientation_responsive = is_orientation_responsive[cells].apply(lambda x: x.sum() > 0)

    return is_vis_responsive, is_direction_responsive, is_orientation_responsive


def calculate_visual_tuning(activity_df, activity_type, direction_label='direction_wrapped',
                            tuning_fit='von_mises', metric_for_analysis='diff_auc',
                            bootstrap_shuffles=500, min_trials_for_bootstrapping=3):
    """
    This function calculates the visual tuning for the given activity dataframe.

    Parameters:
    activity_df (DataFrame): The dataframe containing the activity data.
    activity_type (str): The type of activity.
    direction_label (str, optional): The label for the direction data column. Defaults to 'direction_wrapped'.
    tuning_fit (str, optional): The type of tuning fit. Defaults to 'von_mises'.
    metric_for_analysis (str, optional): The metric for analysis. Defaults to 'diff_auc'.
        Options are 'ratio_mean', 'diff_mean', 'ratio_auc', 'diff_auc', 'ratio_max', 'diff_max'
    bootstrap_shuffles (int, optional): The number of bootstrap shuffles. Defaults to 500.
    min_trials_for_bootstrapping (int, optional): The minimum number of trials for bootstrapping. Defaults to 3.

    Returns:
    DataFrame: The dataframe containing the visual tuning data.
    """

    # --- 0. Setup--- #

    # Set fitting function
    if tuning_fit == 'von_mises':
        fit_function = tuning.calculate_pref_von_mises
    else:
        fit_function = tuning.calculate_pref_gaussian

    # Parse trials and cells
    activity_df.reset_index(drop=True, inplace=True)
    activity_df = activity_df.fillna(0)
    cells = [col for col in activity_df.columns if 'cell' in col]

    # Drop trials that are poorly indexed or incomplete
    activity_df = drop_partial_or_long_trials(activity_df)
    trial_nums = activity_df.loc[activity_df.trial_num > 0].trial_num.unique()
    num_trials = len(trial_nums)
    directions = activity_df.loc[np.isin(activity_df.trial_num, trial_nums)].groupby('trial_num')[direction_label].first()
    orientations = activity_df.loc[np.isin(activity_df.trial_num, trial_nums)].groupby('trial_num')['orientation'].first()

    # Get the trial counts per angle and decide if we can do equal trial number bootstrapping
    angle_counts = activity_df.loc[activity_df.trial_num > 0].groupby(direction_label).apply(lambda x: x.trial_num.nunique())
    min_presentations = int(angle_counts.min())
    if min_presentations < min_trials_for_bootstrapping:
        print(f'    Not enough presentations per angle to bootstrap resultant/DSI/OSI with at least '
              f'{min_trials_for_bootstrapping} trials.')
        do_equal_trial_nums_boostrap = False
    else:
        do_equal_trial_nums_boostrap = True

    # Threshold data if using inferred spikes
    if activity_type == 'spikes':
        # define the clipping threshold in percentile of baseline and do the cell clipping
        clip_threshold = 8
        activity_df[cells] = activity_df.loc[:, cells].apply(clipping_function, threshold=clip_threshold,
                                                             raw=True, axis=1)

    # --- 1. Calculate Responsivity --- #

    # -- 1.0 Get std of each cell across whole experiment
    std_activity = activity_df.loc[:, cells].apply(np.std)

    # -- 1.1 Get the std or cell response, and mean, max and AUC of response during trials
    trial_max_activity = (activity_df.loc[activity_df.trial_num > 0, :]
                          .groupby(['trial_num', direction_label, 'orientation'])[cells]
                          .agg(np.max).copy().reset_index())
    trial_mean_activity = (activity_df.loc[activity_df.trial_num > 0, :]
                           .groupby(['trial_num', direction_label, 'orientation'])[cells]
                           .agg(np.mean).copy().reset_index())
    trial_auc_activity = (activity_df.loc[activity_df.trial_num > 0, :]
                          .groupby(['trial_num', direction_label, 'orientation'])[cells]
                          .agg(np.trapz).copy().reset_index())

    # -- 1.2 Get the std or cell response, and mean, max and AUC of response during ITI
    #    Parse the dataframe into trial frames, which are the trial + 1.5 sec of the preceding inter-trial interval
    #    This will be used for evaluating per-trial responsivity
    trial_frames_short_iti, _ = tuning.parse_trial_frames(activity_df, pre_trial=1.5)

    iti_std_activity = (trial_frames_short_iti.groupby('frame_num')
                        .apply(lambda x: x.loc[x.trial_num == 0, cells].std())
                        .reset_index(names='trial_num'))
    iti_std_activity.insert(1, direction_label, trial_max_activity[direction_label])
    iti_std_activity.insert(2, 'orientation', trial_max_activity['orientation'])

    iti_mean_activity = (trial_frames_short_iti.groupby('frame_num')
                         .apply(lambda x: x.loc[x.trial_num == 0, cells].mean())
                         .reset_index(names='trial_num'))
    iti_mean_activity.insert(1, direction_label, trial_max_activity[direction_label])
    iti_mean_activity.insert(2, 'orientation', trial_max_activity['orientation'])

    iti_max_activity = (trial_frames_short_iti.groupby('frame_num')
                        .apply(lambda x: x.loc[x.trial_num == 0, cells].max())
                        .reset_index(names='trial_num'))
    iti_max_activity.insert(1, direction_label, trial_max_activity[direction_label])
    iti_max_activity.insert(2, 'orientation', trial_max_activity['orientation'])

    iti_auc_activity = (trial_frames_short_iti.groupby('frame_num')
                        .apply(lambda x: x.loc[x.trial_num == 0, cells].apply(np.trapz))
                        .reset_index(names='trial_num'))
    iti_auc_activity.insert(1, direction_label, trial_max_activity[direction_label])
    iti_auc_activity.insert(2, 'orientation', trial_max_activity['orientation'])

    # -- 1.3 Determine if the cell is visually responsive.
    #    This is done by comparing the AUC activity during each 5 sec ITI to the AUC activity during the trial. If a
    #    cell passes a Mann-Whitney U test where the test checks if activity during the trials is greater than that
    #    during the ITI (i.e. alternative='greater'), then the cell is considered visually responsive.
    trial_frames_long_iti, _ = tuning.parse_trial_frames(activity_df, pre_trial=5.0)
    long_iti_auc_activity = (trial_frames_long_iti.groupby('frame_num')
                             .apply(lambda x: x.loc[x.trial_num == 0, cells]
                                    .apply(np.trapz))
                             .reset_index(names='trial_num'))
    long_iti_auc_activity.insert(1, 'direction_wrapped', trial_max_activity['direction_wrapped'])
    long_iti_auc_activity.insert(2, 'orientation', trial_max_activity['orientation'])

    stats, pvals = mannwhitneyu(trial_auc_activity[cells], long_iti_auc_activity[cells],
                                alternative='greater', axis=0)
    vis_drive_test = pd.DataFrame(index=cells, data={'vis_resp_statistic': stats, 'vis_resp_pval': pvals})

    vis_drive_test['is_vis_resp'] = vis_drive_test.vis_resp_pval < processing_parameters.responsivity_p_cutoff
    vis_drive_test['not_vis_resp'] = vis_drive_test.vis_resp_pval > 1 - processing_parameters.responsivity_p_cutoff
    vis_drive_test['mod_vis_resp'] = np.logical_and(vis_drive_test.vis_resp_pval >= processing_parameters.responsivity_p_cutoff,
                                                    vis_drive_test.vis_resp_pval <= 1 - processing_parameters.responsivity_p_cutoff)

    # --- 2. Generate Response Vectors --- #

    # -- 2.1 Calculate the ratios and differences of activity between trials and ITIs
    #    We can choose one of these metrics as the basis for all further analyses

    # Get the ratio of activity during the trial to  activity during the ITI
    ratio_mean_activity = calculate_response_ratios(trial_mean_activity, iti_mean_activity, cells)
    ratio_mean_activity = pd.concat(
        [trial_mean_activity[['trial_num', direction_label, 'orientation']], ratio_mean_activity], axis=1)

    ratio_max_activity = calculate_response_ratios(trial_max_activity, iti_max_activity, cells)
    ratio_max_activity = pd.concat(
        [trial_max_activity[['trial_num', direction_label, 'orientation']], ratio_max_activity], axis=1)

    ratio_auc_activity = calculate_response_ratios(trial_auc_activity, iti_auc_activity, cells)
    ratio_auc_activity = pd.concat(
        [trial_auc_activity[['trial_num', direction_label, 'orientation']], ratio_auc_activity], axis=1)

    # Get the difference of activity during the trial to the activity during the ITI
    diff_mean_activity = calculate_response_diff(trial_mean_activity, iti_mean_activity, cells)
    diff_mean_activity = pd.concat(
        [trial_mean_activity[['trial_num', direction_label, 'orientation']], diff_mean_activity], axis=1)

    diff_max_activity = calculate_response_diff(trial_max_activity, iti_max_activity, cells)
    diff_max_activity = pd.concat(
        [trial_max_activity[['trial_num', direction_label, 'orientation']], diff_max_activity], axis=1)

    diff_auc_activity = calculate_response_diff(trial_auc_activity, iti_auc_activity, cells)
    diff_auc_activity = pd.concat(
        [trial_auc_activity[['trial_num', direction_label, 'orientation']], diff_auc_activity], axis=1)

    # -- 2.2 Choose the metric to use for further analyses
    if metric_for_analysis == 'ratio_mean':
        used_activity_ds = ratio_mean_activity
    elif metric_for_analysis == 'ratio_max':
        used_activity_ds = ratio_max_activity
    elif metric_for_analysis == 'ratio_auc':
        used_activity_ds = ratio_auc_activity
    elif metric_for_analysis == 'diff_mean':
        used_activity_ds = diff_mean_activity
    elif metric_for_analysis == 'diff_max':
        used_activity_ds = diff_max_activity
    elif metric_for_analysis == 'diff_auc':
        used_activity_ds = diff_auc_activity
    else:
        raise ValueError('metric_for_analysis must be one of ratio_mean, ratio_max, ratio_auc, diff_mean, diff_max, diff_auc')

    # -- 2.3 Get the mean direction and orientation response per trial and drop the inter-trial interval from df
    mean_direction_activity_by_trial = (used_activity_ds.groupby([direction_label, 'trial_num'])[cells]
                                        .agg(np.mean).copy().reset_index())
    mean_direction_activity_by_trial = (mean_direction_activity_by_trial
                                        .drop(mean_direction_activity_by_trial[mean_direction_activity_by_trial.trial_num == 0].index)
                                        .sort_values([direction_label, 'trial_num']))

    # Make sure to explicitly represent 360 degrees in the direction data by duplicating the 0 degrees value
    dup = mean_direction_activity_by_trial.loc[mean_direction_activity_by_trial[direction_label] == 0].copy()
    dup.loc[:, direction_label] = 360.
    mean_direction_activity_by_trial = pd.concat([mean_direction_activity_by_trial, dup], ignore_index=True)

    # Get the mean orientation response per trial and drop the inter-trial interval from df
    mean_orientation_activity_by_trial = (used_activity_ds.groupby(['orientation', 'trial_num'])[cells]
                                          .agg(np.mean).copy().reset_index())
    mean_orientation_activity_by_trial = (mean_orientation_activity_by_trial
                                          .drop(mean_orientation_activity_by_trial[mean_orientation_activity_by_trial.trial_num == 0].index)
                                          .sort_values(['orientation', 'trial_num']))

    # -- 2.4 Create the response vectors and normalized response vectors
    mean_dir, sem_dir, std_dir, unique_dirs = generate_tcs_and_stats(mean_direction_activity_by_trial)
    mean_ori, sem_ori, std_ori, unique_oris = generate_tcs_and_stats(mean_orientation_activity_by_trial)

    # Normalize the responses of each cell to the maximum mean response of the cell on any given trial
    norm_direction_activity_by_trial = mean_direction_activity_by_trial.copy()
    norm_direction_activity_by_trial[cells] = norm_direction_activity_by_trial[cells].apply(tuning.normalize)
    norm_mean_dir, norm_sem_dir, norm_std_dir, _ = generate_tcs_and_stats(norm_direction_activity_by_trial)

    norm_orientation_activity_by_trial = mean_orientation_activity_by_trial.copy()
    norm_orientation_activity_by_trial[cells] = norm_orientation_activity_by_trial[cells].apply(tuning.normalize)
    norm_mean_ori, norm_sem_ori, norm_std_ori, _ = generate_tcs_and_stats(norm_orientation_activity_by_trial)

    # --- Run the main loop for all cells --- #
    cell_data_list = []
    for cell in cells:

        # -- 3. Calculate fit using all trials -- #
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

        dir_gof = tuning.goodness_of_fit(norm_direction_activity_by_trial[direction_label].to_numpy(),
                                         norm_direction_activity_by_trial[cell].to_numpy(),
                                         dir_fit_curve[:, 0], dir_fit_curve[:, 1],
                                         type=processing_parameters.gof_type)

        ori_gof = tuning.goodness_of_fit(norm_orientation_activity_by_trial['orientation'].to_numpy(),
                                         norm_orientation_activity_by_trial[cell].to_numpy(),
                                         ori_fit_curve[:, 0], ori_fit_curve[:, 1],
                                         type=processing_parameters.gof_type)

        # -- 3.1 Calculate the direction and orientation selectivity indices using the peak response found by the fit
        fit_dsi, fit_osi = tuning.calculate_dsi_osi_fit(unique_dirs, norm_mean_dir[cell].to_numpy(), real_pref_dir)

        # -- 3.2 Bootstrap fit and responsivity using all trials
        # Split the data with an 80-20 train-test split. Fit the tuning curve on the training data and calculate
        # goodness of fit on the test data.
        bootstrap_dir_gof, bootstrap_pref_dir, bootstrap_real_pref_dir = \
            tuning.bootstrap_tuning_curve(norm_direction_activity_by_trial[[direction_label, 'trial_num', cell]],
                                          fit_function,
                                          gof_type=processing_parameters.gof_type,
                                          num_shuffles=bootstrap_shuffles, mean=mean_guess_dir)
        p_dir_gof = percentileofscore(bootstrap_dir_gof[~np.isnan(bootstrap_dir_gof)], dir_gof, kind='mean') / 100.

        bootstrap_ori_gof, bootstrap_pref_ori, bootstrap_real_pref_ori = \
            tuning.bootstrap_tuning_curve(norm_orientation_activity_by_trial[['orientation', 'trial_num', cell]],
                                          fit_function,
                                          gof_type=processing_parameters.gof_type,
                                          num_shuffles=bootstrap_shuffles, mean=mean_guess_ori)
        p_ori_gof = percentileofscore(bootstrap_ori_gof[~np.isnan(bootstrap_ori_gof)], ori_gof, kind='mean') / 100.

        # --- 4. Get resultant vector, variance, DSI and OSI using the resultant vectors --- #

        # Use the direction dataset first
        theta_dirs = np.deg2rad(unique_dirs)
        dir_magnitudes = norm_mean_dir[cell].copy().to_numpy()

        resultant_dsi_nasal_temporal, resultant_dsi_abs, resultant_osi, resultant_dir_length, resultant_dir, resultant_null_dir = \
            tuning.calculate_dsi_osi_resultant(theta_dirs, dir_magnitudes)
        resultant_dir = fk.wrap(np.rad2deg(resultant_dir), bound=360.)
        resultant_null_dir = fk.wrap(np.rad2deg(resultant_null_dir), bound=360.)
        responsivity_dir = resultant_dir_length

        # For the orientation dataset
        theta_oris = np.deg2rad(unique_oris)
        ori_sep = np.mean(np.diff(theta_oris))
        ori_magnitudes = norm_mean_ori[cell].copy().to_numpy()

        resultant_ori_length, resultant_ori = tuning.resultant_vector(theta_oris, ori_magnitudes, 1)
        resultant_ori = fk.wrap(np.rad2deg(resultant_ori), bound=180.)
        null_ori = fk.wrap(resultant_ori + 90, bound=180.)

        circ_var_ori = circ.var(theta_oris, w=ori_magnitudes, d=ori_sep)
        responsivity_ori = 1 - circ_var_ori

        # --- 5. Run permutation tests using single trial mean responses --- #

        # -- 5.1  Bootstrap resultant vector while guaranteeing the same number of presentations per angle
        #    This is what Joel does for significant shifts in tuning curves

        if do_equal_trial_nums_boostrap:
            # For direction data
            bootstrap_dsi_nasal_temporal, bootstrap_dsi_abs, bootstrap_osi, bootstrap_resultant_dir, bootstrap_null_dir \
                = tuning.boostrap_dsi_osi_resultant(norm_direction_activity_by_trial[[direction_label, 'trial_num', cell]],
                                                    sampling_method='equal_trial_nums', num_shuffles=bootstrap_shuffles)

            bootstrap_resultant_dir[:, 1] = fk.wrap(bootstrap_resultant_dir[:, 1], bound=360.)
            bootstrap_null_dir = fk.wrap(bootstrap_null_dir, bound=360.)

            bootstrap_responsivity_dir = bootstrap_resultant_dir[:, 0]
            p_dsi_nasal_temporal_bootstrap = percentileofscore(bootstrap_dsi_nasal_temporal, resultant_dsi_nasal_temporal, kind='mean') / 100.
            p_dsi_abs_bootstrap = percentileofscore(bootstrap_dsi_abs, resultant_dsi_abs, kind='mean') / 100.
            p_osi_bootstrap = percentileofscore(bootstrap_osi, resultant_osi, kind='mean') / 100.
            p_responsivity_dir_bootstrap = percentileofscore(bootstrap_responsivity_dir, responsivity_dir, kind='mean') / 100.

            # For orientation data
            bootstrap_resultant_ori = \
                tuning.bootstrap_resultant_orientation(norm_orientation_activity_by_trial[['orientation', 'trial_num', cell]],
                                                       sampling_method='equal_trial_nums', multiplier=2, num_shuffles=bootstrap_shuffles)
            bootstrap_responsivity_ori = bootstrap_resultant_ori[:, 0]
            p_responsivity_ori_bootstrap = percentileofscore(bootstrap_responsivity_ori, responsivity_ori, kind='mean') / 100.

        else:
            # Fill everything with NaNs
            bootstrap_dsi_nasal_temporal = np.full(bootstrap_shuffles, np.nan)
            bootstrap_dsi_abs = np.full(bootstrap_shuffles, np.nan)
            bootstrap_osi = np.full(bootstrap_shuffles, np.nan)
            bootstrap_resultant_dir = np.full((bootstrap_shuffles, 2), np.nan)
            bootstrap_responsivity_dir = bootstrap_resultant_dir[:, 0]
            bootstrap_null_dir = np.full(bootstrap_shuffles, np.nan)

            p_dsi_nasal_temporal_bootstrap = np.full(bootstrap_shuffles, np.nan)
            p_dsi_abs_bootstrap = np.full(bootstrap_shuffles, np.nan)
            p_osi_bootstrap = np.full(bootstrap_shuffles, np.nan)
            p_responsivity_dir_bootstrap = np.full(bootstrap_shuffles, np.nan)

            bootstrap_resultant_ori = np.full((bootstrap_shuffles, 2), np.nan)
            bootstrap_responsivity_ori = bootstrap_resultant_ori[:, 0]
            p_responsivity_ori_bootstrap = np.full(bootstrap_shuffles, np.nan)

        # -- 5.2 Shuffle the trial IDs and compare the real selectivity indices to the bootstrapped distribution
        # For direction data
        shuffle_dsi_nasal_temporal, shuffle_dsi_abs, shuffle_osi, shuffle_resultant_dir, shuffle_null_dir \
            = tuning.boostrap_dsi_osi_resultant(norm_direction_activity_by_trial[[direction_label, 'trial_num', cell]],
                                                sampling_method='shuffle_trials', num_shuffles=bootstrap_shuffles)

        p_dsi_nasal_temporal_shuffle = percentileofscore(shuffle_dsi_nasal_temporal, resultant_dsi_nasal_temporal, kind='mean') / 100.
        shuffle_responsivity_dir = shuffle_resultant_dir[:, 0]
        p_dsi_abs_shuffle = percentileofscore(shuffle_dsi_abs, resultant_dsi_abs, kind='mean') / 100.
        p_osi_shuffle = percentileofscore(shuffle_osi, resultant_osi, kind='mean') / 100.
        p_responsivity_dir_shuffle = percentileofscore(shuffle_responsivity_dir, responsivity_dir, kind='mean') / 100.

        # For orientation data
        shuffle_resultant_ori = \
            tuning.bootstrap_resultant_orientation(norm_orientation_activity_by_trial[['orientation', 'trial_num', cell]],
                                                   sampling_method='shuffle_trials', multiplier=2,
                                                   num_shuffles=bootstrap_shuffles)
        shuffle_responsivity_ori = shuffle_resultant_ori[:, 0]
        p_responsivity_ori_shuffle = percentileofscore(shuffle_responsivity_ori, responsivity_ori, kind='mean') / 100.

        # -- Assemble data for saving -- #
        vis_resp_data = [trial_max_activity[cell].to_numpy(), iti_max_activity[cell].to_numpy(),
                         trial_mean_activity[cell].to_numpy(), iti_mean_activity[cell].to_numpy(),
                         trial_auc_activity[cell].to_numpy(), iti_auc_activity[cell].to_numpy(),
                         std_activity[cell], iti_std_activity[cell].to_numpy(),]

        direction_data = [mean_direction_activity_by_trial[[direction_label, cell]].to_numpy(),
                          norm_direction_activity_by_trial[[direction_label, cell]].to_numpy(),
                          np.vstack([unique_dirs, mean_dir[cell].to_numpy()]).T,
                          np.vstack([unique_dirs, norm_mean_dir[cell].to_numpy()]).T,
                          std_dir[cell].to_numpy(), norm_std_dir[cell].to_numpy(),
                          sem_dir[cell].to_numpy(), norm_sem_dir[cell].to_numpy(),
                          dir_fit, dir_fit_curve, dir_gof, bootstrap_dir_gof, p_dir_gof,
                          pref_dir, bootstrap_pref_dir, real_pref_dir, bootstrap_real_pref_dir, fit_dsi,
                          (resultant_dir_length, resultant_dir), bootstrap_resultant_dir, shuffle_resultant_dir,
                          responsivity_dir, bootstrap_responsivity_dir, p_responsivity_dir_bootstrap,
                          shuffle_responsivity_dir, p_responsivity_dir_shuffle,
                          resultant_null_dir, bootstrap_null_dir, shuffle_null_dir,
                          resultant_dsi_nasal_temporal, bootstrap_dsi_nasal_temporal, p_dsi_nasal_temporal_bootstrap,
                          shuffle_dsi_nasal_temporal, p_dsi_nasal_temporal_shuffle,
                          resultant_dsi_abs, bootstrap_dsi_abs, p_dsi_abs_bootstrap,
                          shuffle_dsi_abs, p_dsi_abs_shuffle]

        orientation_data = [mean_orientation_activity_by_trial[['orientation', cell]].to_numpy(),
                            norm_orientation_activity_by_trial[['orientation', cell]].to_numpy(),
                            np.vstack([unique_oris, mean_ori[cell].to_numpy()]).T,
                            np.vstack([unique_oris, norm_mean_ori[cell].to_numpy()]).T,
                            std_ori[cell].to_numpy(), norm_std_ori[cell].to_numpy(),
                            sem_ori[cell].to_numpy(), norm_sem_ori[cell].to_numpy(),
                            ori_fit, ori_fit_curve, ori_gof, bootstrap_ori_gof, p_ori_gof,
                            pref_ori, null_ori, bootstrap_pref_ori, real_pref_ori, bootstrap_real_pref_ori, fit_osi,
                            (resultant_ori_length, resultant_ori), bootstrap_resultant_ori, shuffle_resultant_ori,
                            responsivity_ori, bootstrap_responsivity_ori, p_responsivity_ori_bootstrap,
                            shuffle_responsivity_ori, p_responsivity_ori_shuffle,
                            resultant_osi, bootstrap_osi, p_osi_bootstrap, shuffle_osi, p_osi_shuffle]

        cell_data = vis_resp_data + direction_data + orientation_data
        cell_data_list.append(cell_data)

    # -- Assemble large dataframe for single cells data-- #
    vis_resp_columns = ['max_vis_activity', 'max_baseline_activity',
                        'mean_vis_activity', 'mean_baseline_activity',
                        'auc_vis_activity', 'auc_baseline_activity',
                        'std_vis_activity', 'std_baseline_activity',]

    direction_columns = ['resp_dir',
                         'resp_norm_dir',
                         'mean_dir',
                         'mean_norm_dir',
                         'std_dir', 'std_norm_dir',
                         'sem_dir', 'sem_norm_dir',
                         'fit_dir', 'fit_curve_dir', 'gof_dir', 'bootstrap_gof_dir', 'p_gof_dir',
                         'pref_dir', 'bootstrap_pref_dir', 'real_pref_dir', 'bootstrap_real_pref_dir', 'fit_dsi',
                         'resultant_dir', 'bootstrap_resultant_dir', 'shuffle_resultant_dir',
                         'responsivity_dir', 'bootstrap_responsivity_dir', 'bootstrap_p_responsivity_dir',
                         'shuffle_responsivity_dir', 'shuffle_p_responsivity_dir',
                         'resultant_null_dir', 'bootstrap_null_dir', 'shuffle_null_dir',
                         'resultant_dsi_nasal_temporal', 'bootstrap_dsi_nasal_temporal', 'bootstrap_p_dsi_nasal_temporal',
                         'shuffle_dsi_nasal_temporal', 'shuffle_p_dsi_nasal_temporal',
                         'resultant_dsi_abs', 'bootstrap_dsi_abs', 'bootstrap_p_dsi_abs',
                         'shuffle_dsi_abs', 'shuffle_p_dsi_abs']

    orientation_columns = ['resp_ori',
                           'resp_norm_ori',
                           'mean_ori',
                           'mean_norm_ori',
                           'std_ori', 'std_norm_ori',
                           'sem_ori', 'sem_norm_ori',
                           'fit_ori', 'fit_curve_ori', 'gof_ori', 'bootstrap_gof_ori', 'p_gof_ori',
                           'pref_ori', 'null_ori', 'bootstrap_pref_ori', 'real_pref_ori', 'bootstrap_real_pref_ori',
                           'fit_osi',
                           'resultant_ori', 'bootstrap_resultant_ori', 'shuffle_resultant_ori',
                           'responsivity_ori', 'bootstrap_responsivity_ori', 'bootstrap_p_responsivity_ori',
                           'shuffle_responsivity_ori', 'shuffle_p_responsivity_ori',
                           'resultant_osi', 'bootstrap_osi', 'bootstrap_p_osi', 'shuffle_osi', 'shuffle_p_osi']

    data_cols = vis_resp_columns + direction_columns + orientation_columns
    data_df = pd.DataFrame(index=cells, columns=data_cols, data=cell_data_list)

    # Append the responsivity dataframe to it
    data_df = pd.concat([vis_drive_test, data_df], axis=1)

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
        matched_calcium.loc[mask, 'orientation_rel_ground'] = matched_calcium.loc[mask, 'orientation_rel_ground'].apply(fk.wrap, bound=180.1)

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
    for col in ['mouse_y_m', 'mouse_z_m', 'mouse_x_m', 'head_height', 'mouse_speed', 'mouse_acceleration']:
        if col in kinematics.columns:
            kinematics[col] = kinematics[col] * 100.
        else:
            pass

    # Incorporate wheel speed absolute
    if exp_type == 'fixed':
        kinematics['wheel_speed_abs'] = np.abs(kinematics['wheel_speed'])
        kinematics['wheel_acceleration_abs'] = np.abs(kinematics['wheel_acceleration'])
        kinematics['norm_wheel_speed'] = tuning.normalize(kinematics['wheel_speed_abs'])

    raw_spikes = matched_calcium.loc[:, stimulus_cols + spikes_cols]
    raw_spikes.columns = [key.rsplit('_', 1)[0] if 'spikes' in key else key for key in raw_spikes.columns]
    raw_fluor = matched_calcium.loc[:, stimulus_cols + fluor_cols]
    raw_fluor.columns = [key.rsplit('_', 1)[0] if 'fluor' in key else key for key in raw_fluor.columns]

    return kinematics, raw_spikes, raw_fluor


def predict_running_gmm_hmm(running_trace, n_components=2):
    scores = list()
    models = list()

    for idx in range(10):  # ten different random starting states
        # define our hidden Markov model
        model = hmm.GMMHMM(n_components=n_components, random_state=idx, n_iter=100)
        model.fit(running_trace)
        models.append(model)
        scores.append(model.score(running_trace))

    # get the best model
    model = models[np.argmax(scores)]
    print(f'The best model had a score of {max(scores)} and '
          f'{model.n_components} components')

    # use the Viterbi algorithm to predict the most likely sequence of states
    # given the model
    states = model.predict(running_trace)
    return states


def get_running_modulated_cells(activity_df, running_bouts_idxs, ci_interval=0.95):
    cells = [col for col in activity_df.columns if 'cell' in col]

    # Find cells that are significantly modulated by running in general
    still_bouts_idxs = np.setdiff1d(activity_df.index, running_bouts_idxs)
    cell_running_activity = activity_df.loc[running_bouts_idxs, cells].apply(np.nanmean, axis=0)
    cell_still_activity = activity_df.loc[still_bouts_idxs, cells].apply(np.nanmean, axis=0)
    running_diff = cell_running_activity - cell_still_activity
    running_cis = t.interval(ci_interval, len(running_diff) - 1,
                             loc=np.mean(running_diff), scale=sem(running_diff))
    sig_running_modulated = (running_diff < running_cis[0]) | (running_diff > running_cis[1])

    # Find cells that are significantly modulated by running during visual stimulus
    vis_stim_idxs = activity_df[activity_df.trial_num >= 1].index
    vis_running_idxs = np.intersect1d(running_bouts_idxs, vis_stim_idxs)
    vis_still_idxs = np.intersect1d(still_bouts_idxs, vis_stim_idxs)
    vis_cell_running_activity = activity_df.loc[vis_running_idxs, cells].apply(np.nanmean, axis=0)
    vis_cell_still_activity = activity_df.loc[vis_still_idxs, cells].apply(np.nanmean, axis=0)
    vis_running_diff = vis_cell_running_activity - vis_cell_still_activity
    vis_running_cis = t.interval(ci_interval, len(vis_running_diff) - 1,
                                 loc=np.mean(vis_running_diff), scale=sem(vis_running_diff))
    sig_vis_running_modulated = (vis_running_diff < vis_running_cis[0]) | (vis_running_diff > vis_running_cis[1])

    df = pd.DataFrame({'cell': cells, 'run_activity': cell_running_activity, 'still_activity': cell_still_activity,
                       'run_diff': running_diff, 'sig_run_modulated': sig_running_modulated,
                       'vis_run_activity': cell_running_activity, 'vis_still_activity': cell_still_activity,
                       'vis_run_diff': vis_running_diff, 'sig_vis_run_modulated': sig_vis_running_modulated})

    return df


def calculate_kinematic_tuning(df, day, animal, rig):
    """ Process kinematic tuning
    This is lifted directly from tc_calculate.py
    Note here that the kinematic data that was used for the visual tuning is not fed to the kinematic tuning
    curve calculation below since the formatting is different.

    """

    print('Calculating kinematic tuning curves...')

    # Drop fluorescence columns since not used for this analysis
    fluor_cols = [col for col in df.columns if 'fluor' in col]
    df.drop(columns=fluor_cols, inplace=True)

    # get the number of bins
    bin_num = processing_parameters.bin_number
    shuffle_kind = processing_parameters.tc_shuffle_kind
    percentile = processing_parameters.tc_resp_qual_cutoff

    # define the pairs to quantify
    if rig in ['VWheel', 'VWheelWF']:
        variable_names = processing_parameters.variable_list_fixed
        df['wheel_speed_abs'] = np.abs(df['wheel_speed'].copy())
        df['wheel_acceleration_abs'] = np.abs(df['wheel_acceleration'].copy())
        df['norm_wheel_speed'] = tuning.normalize(df['wheel_speed_abs'])
    else:
        variable_names = processing_parameters.variable_list_free

    # Convert to cm or cm/s^2 (if acceleration)
    # Since this is the original dataset and not the visual tuning dataset, the conversion is also done here
    for col in ['mouse_y_m', 'mouse_z_m', 'mouse_x_m',
                'head_height', 'mouse_speed', 'mouse_acceleration']:
        if col in df.columns:
            df[col] = df[col] * 100.
        else:
            pass

    # clip the calcium traces
    clipped_data = clip_calcium([df])

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
                                 tt_split=processing_parameters.tc_train_test_split,
                                 percentile=percentile, bin_number=bin_num, shuffle_kind=shuffle_kind)

    # get the TC consistency
    tcs_cons = extract_consistency(tcs_half, variable_names, cell_num, shuffle_kind=shuffle_kind,
                                   percentile=processing_parameters.tc_consistency_cutoff)

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
            kinematics, inferred_spikes, deconvolved_fluor = parse_kinematic_data(raw_data[0][-1], rig)

            # Calculate normalized fluorescence and spikes
            activity_ds_dict = {}
            activity_ds_dict['deconvolved_fluor'] = deconvolved_fluor
            activity_ds_dict['inferred_spikes'] = inferred_spikes

            norm_spikes = tuning.normalize_responses(inferred_spikes)
            norm_fluor = tuning.normalize_responses(deconvolved_fluor)
            activity_ds_dict['norm_deconvolved_fluor'] = norm_fluor
            activity_ds_dict['norm_inferred_spikes'] = norm_spikes

            # dff = tuning.calculate_dff(deconvolved_fluor, baseline_type='quantile', quantile=0.08)
            # norm_dff = tuning.normalize_responses(dff)
            # activity_ds_dict['dff'] = dff
            # activity_ds_dict['norm_dff'] = norm_dff

            # Filter trials by head pitch if freely moving
            if rig in ['VTuningWF', 'VTuning']:
                pitch_lower_cutoff = processing_parameters.head_pitch_cutoff[0]
                pitch_upper_cutoff = processing_parameters.head_pitch_cutoff[1]
                view_fraction = processing_parameters.view_fraction
                kinematics['viewed'] = np.logical_and(kinematics['head_pitch'].to_numpy() >= pitch_lower_cutoff,
                                                      kinematics['head_pitch'].to_numpy() <= pitch_upper_cutoff)
                viewed_trials = kinematics.groupby('trial_num').filter(
                    lambda x: (x['viewed'].sum() / len(x['viewed'])) > view_fraction).trial_num.unique()

                viewed_activity_dict = {}
                for ds_key in activity_ds_dict.keys():
                    viewed_activity_dict[ds_key + '_viewed'] = activity_ds_dict[ds_key].loc[
                        activity_ds_dict[ds_key].trial_num.isin(viewed_trials)].copy()

            else:
                viewed_trials = inferred_spikes.trial_num.unique()

                viewed_activity_dict = {}
                for ds_key in activity_ds_dict.keys():
                    viewed_activity_dict[ds_key + '_viewed'] = activity_ds_dict[ds_key].copy()

            activity_ds_dict.update(viewed_activity_dict)

            # Filter trials by running speed
            if rig == 'VTuningWF':
                speed_column = 'mouse_speed'
            else:
                speed_column = 'wheel_speed_abs'

            # Use GMM - HMM to predict running state
            running_prediction = predict_running_gmm_hmm(kinematics[speed_column].to_numpy().reshape(-1,1),
                                                         n_components=2)
            running_idxs = np.argwhere(running_prediction > 0).flatten()
            still_idxs = np.argwhere(running_prediction == 0).flatten()
            kinematics['is_running'] = running_prediction > 0

            still_trials = kinematics.iloc[still_idxs, :].groupby('trial_num').trial_num.unique()
            still_trials = viewed_trials[np.in1d(viewed_trials, still_trials)]

            still_activity_dict = {}
            for ds_key in viewed_activity_dict.keys():
                still_activity_dict[ds_key + '_still'] = viewed_activity_dict[ds_key].loc[
                    viewed_activity_dict[ds_key].trial_num.isin(still_trials)].copy()

            activity_ds_dict.update(still_activity_dict)

            # Run the visual tuning loop and save to file
            print('Calculating visual tuning curves...')
            vis_prop_dict = {}
            for ds_name in processing_parameters.activity_datasets:

                if ds_name not in activity_ds_dict.keys():
                    raise ValueError(f'Activity dataset {ds_name} not found in the dataset.')

                if 'spikes' in ds_name:
                    activity_ds_type = 'spikes'
                elif 'dff' in ds_name:
                    activity_ds_type = 'dff'
                elif 'fluor' in ds_name:
                    activity_ds_type = 'fluor'
                else:
                    raise ValueError(f'Unknown activity dataset type: {ds_name}')

                activity_ds = activity_ds_dict[ds_name].copy()

                trial_params = activity_ds[['trial_num', 'direction_wrapped', 'orientation']].groupby(
                    'trial_num').first().reset_index()

                props = calculate_visual_tuning(activity_ds, activity_ds_type,
                                                metric_for_analysis=processing_parameters.analysis_metric,
                                                bootstrap_shuffles=processing_parameters.bootstrap_repeats)

                # Save visual features to hdf5 file
                props.to_hdf(out_file, f'{ds_name}_props')
                trial_params.to_hdf(out_file, f'{ds_name}_trial_params')

            # --- Process kinematic tuning --- #
            tcs_dict, tcs_counts_dict, tcs_bins_dict = calculate_kinematic_tuning(dataframe, day, animal, rig)

            # for all the features
            for feature in tcs_dict.keys():
                tcs_dict[feature].to_hdf(out_file, feature)
                tcs_counts_dict[feature].to_hdf(out_file, feature + '_counts')
                tcs_bins_dict[feature].to_hdf(out_file, feature + '_edges')

            # calculate locomotion modulated cells
            running_modulated_cells = get_running_modulated_cells(inferred_spikes, running_idxs)
            running_modulated_cells.to_hdf(out_file, 'running_modulated_cells')

            # --- save the cell matches --- #
            cell_matches.to_hdf(out_file, 'cell_matches')

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
