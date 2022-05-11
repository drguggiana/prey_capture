import processing_parameters
import functions_bondjango as bd
import os
import paths
import pandas as pd
import functions_misc as fm
import numpy as np

try:
    # get the input
    input_path = snakemake.input
    # read the output path and the input file urls
    out_path = snakemake.output[0]
except NameError:
    # get the search string
    search_string = processing_parameters.search_string

    # get the paths from the database
    all_path = bd.query_database('analyzed_data', search_string)
    input_path = [el['analysis_path'] for el in all_path if '_tcday' in el['slug']]

    # assemble the output path
    out_path = os.path.join(paths.analysis_path, 'test_tcconsolidate.hdf5')

# get the rig
if 'miniscope' in input_path[0]:
    rig = 'miniscope'
else:
    rig = 'VR'

# set a flag for an empty file
empty_flag = True

# get a list of the features
feature_list = processing_parameters.variable_list

# allocate the metadata list
meta_list = []

# for all the features
for idx, feature in enumerate(feature_list):
    # allocate a list for the individual dataframes
    file_list = []
    counts_list = []
    # add a file counter
    file_counter = 0
    # cycle through the files
    for files in input_path:
        try:
            # read in the data and store
            file_list.append(pd.read_hdf(files, feature))
            counts_list.append(pd.read_hdf(files, feature+'_counts'))
            # only save the ID for the first feature
            if idx == 0:
                meta_list.append(pd.read_hdf(files, 'meta_data'))
            # add the trial id to the trial TCs
            # TODO: using only first trial, need to find better way so trial info is not lost
            file_list[-1]['id'] = meta_list[file_counter].loc[0, 'id']

            # add a cell_id column
            cell_id = np.arange(file_list[-1].shape[0])
            file_list[-1]['cell_id'] = cell_id
            # print(feature, file_list[-1].shape, files)
            # set the empty flag to false
            empty_flag = False
            # update the file counter
            file_counter += 1
        except KeyError:
            print(f'feature:{feature} is not present in file: {os.path.basename(files)}')
            continue
    if len(file_list) > 0:
        # store the concatenated dataframe in the output file
        joint_df = pd.concat(file_list, axis=0).reset_index(drop=True)
        joint_df.to_hdf(out_path, feature)
        joint_counts = pd.concat(counts_list, axis=0).reset_index(drop=True)
        joint_counts.to_hdf(out_path, feature+'_counts')
        if idx == 0:
            meta_data = pd.concat(meta_list, axis=0).reset_index(drop=True)
            meta_data.to_hdf(out_path, 'meta_data')

# if there were no tcs, save a no-ROIs file
if empty_flag:
    empty = pd.DataFrame([])
    empty.to_hdf(out_path, 'no_ROIs')

# generate database entry
# assemble the entry data
entry_data = {
    'analysis_type': 'tc_consolidate',
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

