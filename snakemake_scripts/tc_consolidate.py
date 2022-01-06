import h5py
import processing_parameters
import functions_bondjango as bd
import os
import paths

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

# allocate memory for the individual days
all_days = []

# cycle through the files
for files in input_path:
    #