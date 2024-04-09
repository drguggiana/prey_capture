import os
import pandas as pd
import numpy as np
import yaml

import paths
import processing_parameters
import functions_bondjango as bd
import functions_data_handling as fdh
import functions_misc as fm
import functions_tuning as tuning
from snakemake_scripts.wf_tc_calculate import parse_kinematic_data, calculate_visual_tuning

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
        preproc_data = [el for el in data_all if '_preproc' in el['slug'] and
                        parsed_search['mouse'].lower() in el['slug']]
        preproc_file = preproc_data[0]['analysis_path']

        tc_data = [el for el in data_all if '_tcday' in el['slug'] and
                   parsed_search['mouse'].lower() in el['slug']]
        tc_file = tc_data[0]['analysis_path']

        # get the parts for the file naming
        rig = tc_data[0]['rig']
        day = tc_data[0]['slug'][:10]
        result = tc_data[0]['result']
        lighting = tc_data[0]['lighting']
        mouse = os.path.basename(tc_file).split('_')[7:10]
        mouse = '_'.join([mouse[0].upper()] + mouse[1:])

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
        kinematics, raw_spikes, raw_fluor = parse_kinematic_data(raw_data[0], rig)

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

        for ds_name in processing_parameters.activity_datasets:
            activity_ds = activity_ds_dict[ds_name]
            props = calculate_visual_tuning(activity_ds, bootstrap_shuffles=processing_parameters.bootstrap_repeats)

            # Update visual features to hdf5 files
            with pd.HDFStore(tc_file, mode='r+') as f:
                feature = f'{ds_name}_props'
                feature_key = '/' + feature

                # update the file
                if feature_key in f.keys():
                    f.remove(feature)
                    props.to_hdf(tc_file, feature)

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
