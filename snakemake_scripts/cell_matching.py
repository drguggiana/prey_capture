import yaml
import functions_data_handling as fd
import functions_bondjango as bd
import paths
import os
from caiman.base.rois import register_multisession
from caiman.utils import visualization
from caiman.source_extraction import cnmf as cnmf

# Main script
try:
    # get the raw output_path
    raw_path = snakemake.output[0]
    # get the parsed path
    dict_path = yaml.load(snakemake.params.output_info, Loader=yaml.FullLoader)
    # # get the input paths
    # paths_all = snakemake.input
    # get the parsed path info
    path_info = [yaml.load(el, Loader=yaml.FullLoader) for el in snakemake.params.file_info]
    # get the analysis type
    analysis_type = dict_path['analysis_type']

except NameError:
    # define the analysis type
    analysis_type = 'cellMatch'
    # define the target mouse
    target_mouse = 'DG_200701_a'
    # define the search query
    search_query = 'slug:' + target_mouse
    # # define the origin model
    # ori_type = 'preprocessing'
    # # get a dictionary with the search terms
    # dict_path = fd.parse_search_string(search_query)

    path_info = bd.query_database('video_experiment', search_query)

    # # get the info and paths
    # path_info, paths_all, parsed_query, date_list, animal_list = \
    #     fd.fetch_preprocessing(search_query + ', =analysis_type:' + ori_type)
    # # get the raw output_path
    # dict_path['analysis_type'] = analysis_type
    raw_path = os.path.join(paths.analysis_path, 'cellMatch_' + search_query + '.hdf5')

    # paths_all = os.path.join(paths.analysis_path, '')

# load the data for the matching
# for all the files
for files in path_info:
    current_file = cnmf.online_cnmf.load_OnlineCNMF(files['fluo_path'])

# run the matching software
spatial_union, assignments, matchings = register_multisession(
    A=footprint_list, dims=size_list[0], templates=template_list)
# save the matching results
# create the appropriate bondjango entry
