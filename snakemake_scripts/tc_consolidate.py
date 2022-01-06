import h5py
import processing_parameters
import functions_bondjango as bd
import os
import paths
import pandas as pd
import functions_misc as fm

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

# get a list of all the features from the first file
with h5py.File(input_path[0], 'r') as f:
    feature_list = list(f.keys())
# for all the features
for feature in feature_list:
    # allocate a list for the individual dataframes
    file_list = []
    # cycle through the files
    for files in input_path:
        try:
            # read in the data and store
            file_list.append(pd.read_hdf(files, feature))
        except KeyError:
            continue
    # store the concatenated dataframe in the output file
    joint_df = pd.concat(file_list, axis=0).reset_index(drop=True)
    joint_df.to_hdf(out_path, feature)

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

