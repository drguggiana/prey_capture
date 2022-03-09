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
    input_path = [el['analysis_path'] for el in all_path if '_preproc' in el['slug']]

    # assemble the output path
    out_path = os.path.join(paths.analysis_path, 'test_latentconsolidate.hdf5')

# get the rig
if 'miniscope' in input_path[0]:
    rig = 'miniscope'
else:
    rig = 'VR'




