# find the activity triggered average for the neural activity
import yaml
import functions_data_handling as fd
import paths
import os
import pandas as pd
import numpy as np

try:
    # get the raw output_path
    raw_path = snakemake.output[0]
    # get the parsed path
    dict_path = yaml.load(snakemake.params.output_info, Loader=yaml.FullLoader)
    # get the input paths
    paths_all = snakemake.input
    # get the parsed path info
    path_info = [yaml.load(el, Loader=yaml.FullLoader) for el in snakemake.params.file_info]
    # get the analysis type
    analysis_type = dict_path['analysis_type']
    # get a list of all the animals and dates involved
    animal_list = [el['mouse'] for el in path_info]
    date_list = [el['date'] for el in path_info]
except NameError:
    # define the analysis type
    analysis_type = 'TrigAve'
    # define the search query
    search_query = 'result:succ,lighting:normal,rig:miniscope,imaging:doric'
    # define the origin model
    ori_type = 'preprocessing'
    # get a dictionary with the search terms
    dict_path = fd.parse_search_string(search_query)

    # get the info and paths
    path_info, paths_all, parsed_query, date_list, animal_list = \
        fd.fetch_preprocessing(search_query + ', =analysis_type:' + ori_type)
    # get the raw output_path
    dict_path['analysis_type'] = analysis_type
    basic_name = '_'.join(dict_path.values())
    raw_path = os.path.join(paths.analysis_path, '_'.join((ori_type, basic_name))+'.hdf5')

# read the data
data = [pd.read_hdf(el, 'matched_calcium') for el in paths_all]
# get the unique animals and dates
unique_mice = np.unique(animal_list)
unique_dates = np.unique(date_list)

# for all the mice
for mouse in unique_mice:
    # for all the dates
    for date in unique_dates:
        # get the data for this mouse and day
        sub_data = [el for idx, el in enumerate(data)
                    if (date_list[idx] == date) & (animal_list[idx] == mouse)]
        if len(sub_data) == 0:
            continue
        elif len(sub_data) > 1:
            sub_data = pd.concat(sub_data)

        # separate the calcium data from the rest
        labels = sub_data.columns
        cells = [el for el in labels if 'cell' in el]
        others = [el for el in labels if 'cell' not in el]
        calcium_data = sub_data.loc[:, cells]
        variables = sub_data.loc[:, others]

        # for each cell
        for cell in cells:
            # get the cells data
            cell_data = calcium_data.loc[:, cell]
            # find the peaks in activity


print('yay')
