import subprocess as sp
import yaml
import functions_bondjango as bd
import paths
import functions_data_handling as fd
import os


# define the type of analysis
input_dictionary = {
    'analysis_type': ['aggBin', 'aggFull', 'aggEnc', 'aggBinCA', 'aggFullCA', 'aggEncCA'],
    'result': ['succ', 'fail'],
    'rig': ['miniscope'],
    'lighting': ['normal', 'dark']
}
# assemble the possible search query
search_queries = fd.combinatorial_query(input_dictionary)
# for all the search queries
for search_query in search_queries:
    # search_query = 'result:succ,rig:miniscope,lighting:normal'

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
    # TODO: update the model definition and the remaining fields to add calcium explicitly
    if 'CA' in parsed_search['analysis_type']:
        parsed_search['notes'] = 'BLANK'
        search_query += ',notes:BLANK'
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
                             for el in target_entries}, 'target_path': target_path}
    with open(paths.snakemake_config, 'w') as f:
        target_file = yaml.dump(config_dict, f)

    # assemble the output path
    # out_path = os.path.join(paths.analysis_path, '_'.join(('preprocessing', *parsed_search.values())) + '.hdf5')

    out_path = os.path.join(paths.figures_path, '_'.join(('averages', *parsed_search.values())) + '.html')

    # run snakemake
    preprocess_sp = sp.Popen(['snakemake', out_path, '--cores', '1',
                              '-s', paths.snakemake_scripts,
                              '-d', paths.snakemake_working],
                             stdout=sp.PIPE)

    stdout = preprocess_sp.communicate()[0]
    print(stdout.decode())
print('yay')
