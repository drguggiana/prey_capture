import os
import yaml

import pandas as pd
import numpy as np

import paths
import processing_parameters
import functions_bondjango as bd
import functions_data_handling as fdh
import functions_misc as fm
import functions_tuning as tuning
from snakemake_scripts.wf_tc_calculate import parse_kinematic_data, calculate_visual_tuning, predict_running_gmm_hmm

if __name__ == '__main__':
    try:
        # get the input
        preproc_file = snakemake.input[0]
        tc_file = snakemake.input[1]
        dummy_out_file = snakemake.output[0]
        file_info = yaml.load(snakemake.params.file_info, Loader=yaml.FullLoader)

        # get the parts for the file naming
        rig = file_info['rig']
        day = file_info['slug'][:10]
        result = file_info['result']
        lighting = file_info['lighting']
        mouse = os.path.basename(tc_file).split('_')[7:10]
        mouse = '_'.join([mouse[0].upper()] + mouse[1:])

    except NameError:

        data_all = bd.query_database('analyzed_data', processing_parameters.search_string)
        parsed_search = fdh.parse_search_string(processing_parameters.search_string)

        # get the paths to the files
        preproc_data = [el for el in data_all if ('_preproc' in el['slug']) and
                        (parsed_search['mouse'].lower() in el['slug']) and
                        (parsed_search['rig'] in el['rig'])]
        preproc_file = preproc_data[0]['analysis_path']

        tc_data = [el for el in data_all if ('_tcday' in el['slug']) and
                        (parsed_search['mouse'].lower() in el['slug']) and
                        (parsed_search['rig'] in el['rig'])]
        tc_file = tc_data[0]['analysis_path']

        # get the parts for the file naming
        rig = parsed_search['rig']
        day = tc_data[0]['slug'][:10]
        result = tc_data[0]['result']
        lighting = tc_data[0]['lighting']
        mouse = parsed_search['mouse']

        dummy_out_file = tc_file.replace('_tcday.hdf5', '_update_dummy.txt')

    # --- Process kinematic tuning --- #

    # load the preprocessed data
    raw_data = []
    with pd.HDFStore(preproc_file, mode='r') as h:
        if '/matched_calcium' in h.keys():
            # concatenate the latents
            dataframe = h['matched_calcium']
            raw_data.append(dataframe)

    # process if file is not empty
    if len(raw_data) != 0:

        # --- Process visual tuning --- #
        kinematics, inferred_spikes, deconvolved_fluor = parse_kinematic_data(raw_data[0], rig)

        # Calculate normalized fluorescence and spikes
        activity_ds_dict = {}
        activity_ds_dict['deconvolved_fluor'] = deconvolved_fluor
        activity_ds_dict['inferred_spikes'] = inferred_spikes

        norm_spikes = tuning.normalize_responses(inferred_spikes)
        norm_fluor = tuning.normalize_responses(deconvolved_fluor)
        activity_ds_dict['norm_deconvolved_fluor'] = norm_fluor
        activity_ds_dict['norm_inferred_spikes'] = norm_spikes

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
        running_prediction = predict_running_gmm_hmm(kinematics[speed_column].to_numpy().reshape(-1, 1),
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

            trial_params = activity_ds[['trial_num', 'direction_wrapped', 'orientation']].groupby('trial_num').first().reset_index()

            props = calculate_visual_tuning(activity_ds, activity_ds_type,
                                            metric_for_analysis=processing_parameters.analysis_metric,
                                            bootstrap_shuffles=processing_parameters.bootstrap_repeats)

            # Update visual features to hdf5 files
            with pd.HDFStore(tc_file, mode='r+') as f:
                feature = f'{ds_name}_props'
                feature_key = '/' + feature

                # update the file
                if feature_key in f.keys():
                    f.remove(feature)
                    props.to_hdf(tc_file, feature)
                else:
                    props.to_hdf(tc_file, feature)
                    trial_params.to_hdf(tc_file, f'{ds_name}_trial_params')

        print(f'Updated visual tuning in {os.path.basename(tc_file)}')

    else:
        print(f'No calcium data in {os.path.basename(preproc_file)}')

    # assemble the entry data
    entry_data = {
        'analysis_type': 'tc_analysis',
        'analysis_path': tc_file,
        'date': '',
        'pic_path': '',
        'result': str(result),
        'rig': str(rig),
        'lighting': str(lighting),
        'imaging': 'wirefree',
        'slug': fm.slugify(os.path.basename(tc_file).split('.')[0]),
    }

    # check if the entry already exists, if so, update it, otherwise, create it
    update_url = '/'.join((paths.bondjango_url, 'analyzed_data', entry_data['slug'], ''))
    output_entry = bd.update_entry(update_url, entry_data)
    if output_entry.status_code == 404:
        # build the url for creating an entry
        create_url = '/'.join((paths.bondjango_url, 'analyzed_data', ''))
        output_entry = bd.create_entry(create_url, entry_data)

    print(f'The output status was {output_entry.status_code}, reason {output_entry.reason}')
    if output_entry.status_code in [500, 400]:
        print(entry_data)

    # Write the dummy file
    with open(dummy_out_file, 'w') as f:
        f.writelines([preproc_file, tc_file])
