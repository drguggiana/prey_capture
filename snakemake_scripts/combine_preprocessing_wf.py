# imports
import functions_bondjango as bd
import os
import processing_parameters
import paths
import yaml
import h5py
import pandas as pd
import numpy as np
import shutil


# check if launched from snakemake, otherwise, prompt user
try:
    # get the path to the file, parse and turn into a dictionary
    preproc_path = snakemake.input[0]
    files = yaml.load(snakemake.params.info, Loader=yaml.FullLoader)
    # get the save paths
    save_path = snakemake.output[0]
except NameError:
    # USE FOR DEBUGGING ONLY (need to edit the search query and the object selection)
    # define the search string
    search_string = processing_parameters.search_string

    # define the target model
    if 'miniscope' in search_string:
        target_model = 'video_experiment'
    else:
        target_model = 'vr_experiment'

    # get the queryset
    files = bd.query_database(target_model, search_string)[0]

    # assemble the input paths
    preproc_path = files['avi_path'].replace('.avi', '_rawcoord.hdf5').replace('VRExperiment', 'AnalyzedData')
    # assemble the save paths
    save_path = os.path.join(paths.analysis_path,
                             os.path.basename(files['avi_path'][:-4]))+'_preproc.hdf5'

# copy the preprocessing file as the output file
shutil.copy(preproc_path, save_path)

# assemble the entry data
entry_data = {
    'analysis_type': 'preprocessing',
    'analysis_path': save_path,
    'date': files['date'],
    'result': files['result'],
    'rig': files['rig'],
    'lighting': files['lighting'],
    'imaging': files['imaging'],
    'slug': files['slug'] + '_preprocessing',
    'video_analysis': [files['url']] if files['rig'] == 'miniscope' else [],
    'vr_analysis': [] if files['rig'] == 'miniscope' else [files['url']],
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

print('<3')
