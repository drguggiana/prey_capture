import os
import numpy as np
import pandas as pd

import paths
import processing_parameters
import functions_bondjango as bd
import functions_misc as fm

try:
    # get the input
    input_path = list(snakemake.input)
    # get the slugs
    slug_list = [os.path.basename(el).replace('_tcday.hdf5', '') for el in input_path]
    # read the output path and the input file urls
    out_path = [snakemake.output[0]]
    data_all = [yaml.load(snakemake.params.file_info, Loader=yaml.FullLoader)]
    # get the parts for the file naming
    rigs = np.unique([d['rig'] for d in data_all])
    animals = [slug.split('_')[7:10] for slug in slug_list]
    animals = np.unique(['_'.join([animal[0].upper()] + animal[1:]) for animal in animals])
    days = [slug[:10] for slug in slug_list]

except NameError:

    # get the paths from the database
    data_all = bd.query_database('analyzed_data', processing_parameters.search_string)
    data_all = [el for el in data_all if '_tcday' in el['slug']]
    input_path = [el['analysis_path'] for el in data_all]

    # get the day, animal and rig
    days = ['_'.join(d['slug'].split('_')[0:3]) for d in data_all]
    rigs = np.unique([el['rig'] for el in data_all])
    animals = [d['slug'].split('_')[7:10] for d in data_all]
    animals = np.unique(['_'.join([animal[0].upper()] + animal[1:]) for animal in animals])

    # assemble the output path
    out_path = os.path.join(paths.analysis_path, f'test_tcconsolidate.hdf5')
