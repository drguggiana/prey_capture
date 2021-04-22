import pandas as pd
from datetime import datetime
from numpy import abs, mean, ones

from functions_bondjango import query_database
from functions_preprocessing import timed_event_finder


def load_h5_df(path):
    """Loads a Pandas data frame from an h5 file"""

    h5_obj = pd.HDFStore(path)

    # Add a column to the trial_params data set that will properly index the trials
    trial_params = h5_obj.get("trial_set")
    trial_params['trial_num'] = trial_params.index + 1

    data = {'traces': h5_obj.get("full_traces"),
            'session_params': h5_obj.get("params"),
            'trial_params': trial_params}

    return data


def load_VScreens_datasets(search_string, exclusion=None):

    valid_experiments = []

    data_all = query_database('analyzed_data', search_string)

    for ds in data_all:
        date = datetime.strptime(ds['date'], '%Y-%m-%dT%H:%M:%SZ').date()
        animal = "_".join(ds['slug'].split('_')[7:10])

        if (exclusion is None) or (exclusion not in ds['analysis_path']):
            exp = load_h5_df(ds['analysis_path'])
            exp['traces']['date'] = date
            exp['traces']['animal'] = animal
            valid_experiments.append(exp)
            print(ds['analysis_path'])
            print(date)
        else:
            # Skip the excluded experiment
            continue

    # load the data
    return valid_experiments


def calculate_bins(time_trace, bin_duration: int):
    """
    Calculates the time bin of a given time point in a vector
    :param time_trace: array-like
    :param bin_duration: int, duration in minutes
    :return: vector of time bins
    """
    return time_trace // (60 * bin_duration)


def get_trial_traces(df, trials):
    """
    Extracts slices of the input data frame by trial number
    :param df: [pd.DataFrame] Kinematic data from an experiment
    :param trials: [list] trial numbers to pull out
    :return: [pd.DataFrame] Kinematic traces for the selected trials aggregated to lists
    """
    trial_traces = pd.DataFrame()

    for trial in trials:
        trial_slice = df.loc[df['trial_num'] == trial].copy()
        trial_traces = trial_traces.append(trial_slice)

    grouped = trial_traces.groupby(['trial_num']).agg(list)

    return grouped


def approach_finder(data_in, key, threshold_function,
                    distance_threshold=0.05, window=1.5, start_distance=0.05, speed_minimum=0.15):
    all_approaches = []
    all_trials = []
    valid_approaches = []
    valid_trials = []

    for index, row in data_in.iterrows():

        trial = pd.DataFrame(row).T
        trial = trial.apply(pd.Series.explode)

        # This gets us all encounters within one trial
        trial_approaches = timed_event_finder(trial, key, distance_threshold, threshold_function, window=window)

        is_app = []
        good_app = []
        trial_traces = []
        valid_traces = []

        if len(trial_approaches) == 0:
            valid_approaches.append(good_app)
            valid_trials.append(valid_traces)
            all_approaches.append(is_app)
            all_trials.append(trial_traces)
        else:
            app_id_group = trial_approaches.groupby('event_id')
            for a, group in app_id_group:
                trial_traces.append(trial.copy())
                is_app.append(group.copy())
                # group.head()
                quarter_idx = len(group) // 4
                mid_idx = len(group) // 2

                # Qualifications for an approach as in Procacci (decreasing target_mouse_angle, start >= 5cm away, speed >= 15cm/s)
                # angle_start = fk.circmean_deg(group['target_delta_heading'][:quarter_idx].to_list())
                angle_start = group['target_delta_heading'][quarter_idx]
                angle_enc = group['target_delta_heading'][mid_idx]
                angle_change = angle_enc - angle_start
                avg_speed = mean(group['mouse_speed'][:mid_idx])
                start_dist = abs(group[key][mid_idx] - group[key][0])

                if (start_dist >= start_distance) and (avg_speed >= speed_minimum):  # and (angle_change <= 0):
                    # This is a good encounter
                    good_app.append(group.copy())
                    valid_traces.append(trial)
                else:
                    # This does not meet one or more of the criteria, so we remove it.
                    pass

            valid_approaches.append(good_app)
            valid_trials.append(valid_traces)
            all_approaches.append(is_app)
            all_trials.append(trial_traces)

    return all_approaches, all_trials, valid_approaches, valid_trials


def extract_experiment_approaches(data, grouping_params, approach_criteria, threshold_function, output_df):
    experiment_valid_approaches = []
    experiment_approaches = []

    grouped = data['trial_params'].groupby(grouping_params)

    for key, item in grouped:
        temp_storage_dict = dict.fromkeys(list(output_df.columns))

        trial_nums = item['trial_num'].to_list()
        parameter_traces = get_trial_traces(data['traces'], trial_nums)

        # Use an updated encounter finder that encorporates Hoy et al. criteria to find interactions with target during each trial
        all_trial_approaches, approach_trials, valid_approaches, valid_app_trials = approach_finder(parameter_traces,
                                                                                                    'target_mouse_distance_recalc',
                                                                                                    threshold_function,
                                                                                                    **approach_criteria)

        for trial_apps, trials in zip(valid_approaches, valid_app_trials):
            if len(trial_apps) != 0:
                for a, t in zip(trial_apps, trials):
                    experiment_valid_approaches.append([a, t, (key[0], key[1][0] * 100, key[2] * 100)])

        for trial_apps, trials in zip(all_trial_approaches, approach_trials):
            if len(trial_apps) != 0:
                for a, t in zip(trial_apps, trials):
                    experiment_approaches.append([a, t, (key[0], key[1][0] * 100, key[2] * 100)])

        temp_storage_dict['date'] = [pt['date'][0] for i, pt in parameter_traces.iterrows()]
        temp_storage_dict['animal'] = [pt['animal'][0] for i, pt in parameter_traces.iterrows()]
        temp_storage_dict['bin'] = [pt['bin'][0] for i, pt in parameter_traces.iterrows()]
        temp_storage_dict['trial_num'] = trial_nums[:len(parameter_traces)]
        temp_storage_dict['target_color'] = [key[0] for i in range(len(parameter_traces))]
        temp_storage_dict['scale'] = ones(len(parameter_traces)) * key[1][0] * 100  # Get into cm
        temp_storage_dict['speed'] = ones(len(parameter_traces)) * key[2] * 100  # Get into cm/s
        temp_storage_dict['approaches'] = [len(e) for e in valid_approaches]
        temp_storage_dict['has_approach'] = [1 if t > 0 else 0 for t in temp_storage_dict['approaches']]

        group_approaches = pd.DataFrame.from_dict(temp_storage_dict)
        output_df = output_df.append(group_approaches, ignore_index=True)

    return output_df, experiment_valid_approaches, experiment_approaches
