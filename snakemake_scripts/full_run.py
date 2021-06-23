import paths
import os
import yaml

# Main script
try:
    # get the raw output_path
    raw_path = snakemake.output[0]
    # get the parsed path
    dict_path = yaml.load(snakemake.params.output_info, Loader=yaml.FullLoader)
    # get the input paths
    paths_all = snakemake.input
    # # get the parsed path info
    # path_info = [yaml.load(el, Loader=yaml.FullLoader) for el in snakemake.params.file_info]
    # # get the analysis type
    # analysis_type = dict_path['analysis_type']
    # # get a list of all the animals and dates involved
    # animal_list = [el['mouse'] for el in path_info]
    # date_list = [datetime.datetime.strptime(el['date'], '%Y-%m-%dT%H:%M:%SZ').date() for el in path_info]
except NameError:
    # define the output path
    raw_path = os.path.join(paths.analysis_path, 'full_run.txt')
    # define the string
    paths_all = ['test']
#     # define the analysis type
#     analysis_type = 'aggEnc'
#     # define the search query
#     search_query = 'result:succ,lighting:normal,rig:vr'
#     # define the origin model
#     ori_type = 'preprocessing'
#     # get a dictionary with the search terms
#     dict_path = fd.parse_search_string(search_query)
#
#     # get the info and paths
#     path_info, paths_all, parsed_query, date_list, animal_list = \
#         fd.fetch_preprocessing(search_query + ', =analysis_type:' + ori_type)
#     # get the raw output_path
#     dict_path['analysis_type'] = analysis_type
#     basic_name = '_'.join(dict_path.values())
#     raw_path = os.path.join(paths.analysis_path, '_'.join((ori_type, basic_name))+'.hdf5')

# save a text file
with open(raw_path, 'w') as f:
    f.writelines(paths_all)
