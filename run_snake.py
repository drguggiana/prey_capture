import subprocess as sp
import yaml
import functions_bondjango as bd
import paths
import functions_data_handling as fd
import os


# define the type of analysis
input_dictionary = {
    # 'analysis_type': ['aggBin', 'aggFull', 'aggEnc', 'aggBinCA', 'aggFullCA', 'aggEncCA', 'trigAveCA'],
    'analysis_type': ['aggFull'],
    # 'analysis_type': ['trigAveCA'],
    # 'analysis_type': ['trigAveCA'],
    'result': ['test', ],
    # 'result': ['test'],
    'rig': ['VPrey', ],
    'lighting': ['normal', ],
    # 'gtdate': ['2020-03-01T00-00-00'],
    # 'notes': ['crickets_0_vrcrickets_1'],
}
# assemble the possible search query
search_queries = fd.combinatorial_query(input_dictionary)
# for all the search queries
for search_query in search_queries:

    # pick the target model based on the search query
    if 'rig:miniscope' in search_query:
        # define the target model and search query
        target_model = 'video_experiment'
        target_path = paths.videoexperiment_path
    else:
        # define the target model and search query
        target_model = 'vr_experiment'
        target_path = paths.vrexperiment_path
        if 'rig' not in search_query:
            # if the rig argument wasn't give it, add it
            search_query += ',rig:vr'

    # parse the search string
    parsed_search = fd.parse_search_string(search_query)
    # if the analysis type requires CA data, make sure notes=BLANK
    if 'CA' in parsed_search['analysis_type']:

        parsed_search['imaging'] = 'doric'

        if 'imaging' not in search_query:
            search_query += ',imaging:doric'

    # also get the target database entries
    target_entries = bd.query_database(target_model, fd.remove_query_field(search_query, 'analysis_type'))
    # if there are no entries, skip the iteration
    if len(target_entries) == 0:
        print('No entries: ' + search_query)
        continue
    else:
        print(str(len(target_entries)) + ' entries: ' + search_query)

    # create the config file
    config_dict = {'files': {os.path.basename(el['bonsai_path'])[:-4]: os.path.basename(el['bonsai_path'])[:-4]
                             for el in target_entries},
                   'file_info': {os.path.basename(el['bonsai_path'])[:-4]: yaml.dump(el)
                                 for el in target_entries},
                   # 'dlc_flag': {os.path.basename(el['bonsai_path'])[:-4]: True if len(el['avi_path']) > 0
                   #              else False for el in target_entries},
                   'dlc_flag': {os.path.basename(el['bonsai_path'])[:-4]: False for el in target_entries},
                   'output_info': yaml.dump(parsed_search),
                   'target_path': target_path,
                   'dlc_path': paths.dlc_script,
                   'interval': [2, 3],
                   }
    # write the file
    with open(paths.snakemake_config, 'w') as f:
        target_file = yaml.dump(config_dict, f)

    # assemble the output path
    out_path = os.path.join(paths.analysis_path, '_'.join(('preprocessing', *parsed_search.values())) + '.hdf5')

    # out_path = os.path.join(paths.figures_path, '_'.join(('averages', *parsed_search.values())) + '.html')

    # run snakemake
    preprocess_sp = sp.Popen(['snakemake', out_path, out_path, '--cores', '1',
                              '-s', paths.snakemake_scripts,
                              '-d', paths.snakemake_working],
                             stdout=sp.PIPE)

    stdout = preprocess_sp.communicate()[0]
    print(stdout.decode())
print('yay')
