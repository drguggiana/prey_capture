import os
import numpy as np
import pandas as pd

import paths
import processing_parameters
import functions_bondjango as bd
import functions_misc as fm
import snakemake_scripts.wf_tc_consolidate

try:
    # get the input
    input_paths = snakemake.input[0]
    rigs = [path.split('_')[6] for path in input_paths]
    day = np.unique([path[:4] for path in input_paths])
    mouse = np.unique([path[7:10] for path in input_paths])
    # read the output path and the input file urls
    out_path = os.path.join(paths.analysis_path, f'{day[0]}_{mouse[0]}_tcconsolidate.hdf5')
    dummy_out = snakemake.output
except NameError:
    # get the search string
    search_string = processing_parameters.search_string

    # get the paths from the database
    all_path = bd.query_database('analyzed_data', search_string)
    input_paths = [el['analysis_path'] for el in all_path if '_tcday' in el['slug']]
    rigs = [el['rig'] for el in all_path if '_tcday' in el['slug']]

    # assemble the output path
    out_path = os.path.join(paths.analysis_path, 'test_tcconsolidate.hdf5')
    dummy_out = os.path.join(paths.analysis_path, 'test_tcdummy.txt')

with open(dummy_out, 'w') as f:
    f.writelines(input_paths)

# cycle through the files
matches = None
empty_flag = False
data_list = []
for file in input_paths:
    # Load the data
    file_dict = {}
    with pd.HDFStore(file, mode='r') as h:

        for key in h.keys():
            if key == '/no_ROIs':
                empty_flag = True
                break
            elif key == '/cell_matches':
                matches = h[key].dropna().reset_index(drop=True)
            else:
                file_dict[key.split('/')[-1]] = h[key]

        data_list.append(file_dict)


if empty_flag:
    empty = pd.DataFrame([])
    empty.to_hdf(out_path, 'no_ROIs')

else:
    # Save cell matches
    if matches is not None:
        matches.to_hdf(out_path, f'cell_matches')

    matched_data_list = []
    for data, rig in zip(data_list, rigs):
        # get matches and save
        match_col = np.where([rig in el for el in matches.columns])[0][0]
        match_idxs = matches.iloc[:, match_col].to_numpy(dtype=int)

        for feature in data.keys():
            # Save the whole dataset
            data[feature].to_hdf(out_path, f'{rig}/{feature}')

            # Save matched TCs
            if 'counts' not in feature:
                matched_feature = data[feature].iloc[match_idxs, :].reset_index(names=['original_cell_id'])
                matched_feature.to_hdf(out_path, f'matched/{rig}/{feature}')



# generate database entry
# assemble the entry data
entry_data = {
    'analysis_type': 'tc_consolidate',
    'analysis_path': out_path,
    'date': '',
    'pic_path': '',
    'result': 'multi',
    'rig': 'multi',
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