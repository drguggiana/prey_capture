import os
import pandas as pd
import yaml

import paths
import processing_parameters
import functions_bondjango as bd
import functions_data_handling as fdh
import functions_misc as fm
from snakemake_scripts.wf_tc_calculate import calculate_kinematic_tuning


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

        tcs_dict, tcs_counts_dict, tcs_bins_dict = calculate_kinematic_tuning(dataframe, day, mouse, rig)

        # for all the features
        with pd.HDFStore(tc_file, mode='r+') as f:
            for feature in tcs_dict.keys():
                feature_key = '/' + feature

                # update the file
                if feature_key in f.keys():
                    f.remove('feature')
                    tcs_dict[feature].to_hdf(tc_file, feature)

                    f.remove(feature + '_counts')
                    tcs_counts_dict[feature].to_hdf(tc_file, feature + '_counts')

                    f.remove(feature + '_edges')
                    tcs_bins_dict[feature].to_hdf(tc_file, feature + '_edges')

        print(f'Updated kinematic tuning in {os.path.basename(tc_file)}')

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
