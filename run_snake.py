import subprocess as sp
import yaml
import functions_bondjango as bd
import paths
import functions_data_handling as fd
import os
import numpy as np
import processing_parameters


# define the type of analysis
input_dictionary = processing_parameters.input_dictionary

# assemble the possible search query
search_queries = fd.combinatorial_query(input_dictionary)
# allocate a list to store the full queries
full_queries = []
full_paths = []
full_parsed = []
# for all the search queries
for search_query in search_queries:

    # pick the target model based on the search query
    if 'rig:miniscope' in search_query:
        # if 'analysis_type:combinedanalysis' in search_query:
        #     target_model = 'analyzed_data'
        #     target_path = paths.analysis_path
        # else:
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
    # add the queries to the list
    full_queries.append(target_entries)
    full_paths.append(target_path)
    full_parsed.append(parsed_search)

# allocate a list for the mice
new_queries = []
new_paths = []
new_parsed = []
mouse_list = []
# for all the search queries
for idx, search_query in enumerate(full_parsed):
    # modify query if cellMatch
    if search_query['analysis_type'] == 'cellMatch':
        # extract only the mice from the search query that have calcium
        mouse_list.append(np.unique([el['mouse'] for el in full_queries[idx] if len(el['fluo_path']) > 0]))
    else:
        new_queries.append(full_queries[idx])
        new_paths.append(full_paths[idx])
        new_parsed.append(search_query)
# consolidate the mice in the list
mouse_list = np.unique([el for sublist in mouse_list for el in sublist])

# get the entries
for idx, mouse in enumerate(mouse_list):
    target_entries = bd.query_database('vr_experiment', 'slug:' + mouse)
    target_entries.append(bd.query_database('video_experiment', 'slug:' + mouse))

    target_entries = [el for sublist in target_entries for el in sublist]
    # filter out the no fluo
    target_entries = [el for el in target_entries if len(el['fluo_path']) > 0]
    # append to the query list
    new_queries.append(target_entries)
    new_parsed.append({'analysis_type': 'cellMatch'})
    new_paths.append([])

# for all the full queries
for idx, target_entries in enumerate(new_queries):

    parsed_search = new_parsed[idx]
    target_path = new_paths[idx]
    # create the config file
    config_dict = {'files': {os.path.basename(el['bonsai_path'])[:-4]: os.path.basename(el['bonsai_path'])[:-4]
                             for el in target_entries},
                   'file_info': {os.path.basename(el['bonsai_path'])[:-4]: yaml.dump(el)
                                 for el in target_entries},
                   'dlc_flag': {os.path.basename(el['bonsai_path'])[:-4]: True if len(el['avi_path']) > 0
                                else False for el in target_entries},
                   # 'dlc_flag': {os.path.basename(el['bonsai_path'])[:-4]: False for el in target_entries},
                   'calcium_flag': {os.path.basename(el['bonsai_path'])[:-4]: True if len(el['tif_path']) > 0
                                    else False for el in target_entries},
                   # 'calcium_flag': {os.path.basename(el['bonsai_path'])[:-4]: False for el in target_entries},
                   'output_info': yaml.dump(parsed_search),
                   'target_path': target_path,
                   'dlc_path': paths.dlc_script,
                   'cnmfe_path': paths.calcium_script,
                   'interval': [2, 3],
                   'analysis_type': parsed_search['analysis_type'],
                   }
    # write the file
    with open(paths.snakemake_config, 'w') as f:
        target_file = yaml.dump(config_dict, f)

    # assemble the output path
    if (parsed_search['analysis_type'] == 'full_run') or (parsed_search['analysis_type'] == 'combinedanalysis'):
        # feed the generic txt file for preprocessing
        out_path = os.path.join(paths.analysis_path, 'full_run.txt')
    # elif parsed_search['analysis_type'] == 'combinedanalysis':
    #     out_path = os.path.join(paths.analysis_path, 'combinedanalysis.hdf5')
    else:
        # feed the aggregation path
        out_path = os.path.join(paths.analysis_path, '_'.join(('preprocessing', *parsed_search.values())) + '.hdf5')

    # out_path = os.path.join(paths.figures_path, '_'.join(('averages', *parsed_search.values())) + '.html')

    # run snakemake
    preprocess_sp = sp.Popen(['snakemake', out_path, out_path, '--cores', '1',
                              # '-F',         # (hard) force rerun everything
                              # '-f',         # (soft) force rerun last step
                              # '--unlock',   # unlocks the files after force quit
                              # '--rerun-incomplete',
                              # '--verbose',  # make the output more verbose for debugging
                              '-s', paths.snakemake_scripts,
                              '-d', paths.snakemake_working],
                             stdout=sp.PIPE)

    stdout = preprocess_sp.communicate()[0]
    print(stdout.decode())

    # assemble the output path
    if ((parsed_search['analysis_type'] == 'full_run') or (parsed_search['analysis_type'] == 'combinedanalysis')) and \
            os.path.isfile(os.path.join(paths.analysis_path, 'full_run.txt')):
        # delete the just_preprocess file (specify de novo to no run risks)
        os.remove(os.path.join(paths.analysis_path, 'full_run.txt'))
print('yay')
