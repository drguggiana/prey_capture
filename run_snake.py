import subprocess as sp
import yaml
import functions_bondjango as bd
import paths
import functions_data_handling as fd
import os
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
        # if 'rig' not in search_query:
        #     # if the rig argument wasn't give it, add it
        #     search_query += ',rig:vr'

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

    # remove virtual mouse entries
    target_entries = [el for el in target_entries if 'DG_virtual_mouse' != el['mouse']]
    # add the queries to the list
    full_queries.append(target_entries)
    full_paths.append(target_path)
    full_parsed.append(parsed_search)

# for all the full queries
for idx, target_entries in enumerate(full_queries):
    # # allow only the desired dates, for testing purposes only
    # target_entries = [el for el in target_entries if el['slug'][:10] in
    #                   ['03_23_2021', '03_24_2021', '03_29_2021', '03_30_2021', '03_31_2021']]

    parsed_search = full_parsed[idx]
    target_path = full_paths[idx]
    # create the config file
    config_dict = {'files': {os.path.basename(el['avi_path'])[:-4]: os.path.basename(el['avi_path'])[:-4]
                             for el in target_entries},
                   'file_info': {os.path.basename(el['avi_path'])[:-4]: yaml.dump(el)
                                 for el in target_entries},
                   'dlc_flag': {os.path.basename(el['avi_path'])[:-4]: True if len(el['avi_path']) > 0
                                else False for el in target_entries},
                   # 'dlc_flag': {os.path.basename(el['avi_path'])[:-4]: False for el in target_entries},
                   'calcium_flag': {os.path.basename(el['avi_path'])[:-4]: True if len(el['tif_path']) > 0
                                    else False for el in target_entries},
                   # 'calcium_flag': {os.path.basename(el['avi_path'])[:-4]: False for el in target_entries},
                   'output_info': yaml.dump(parsed_search),
                   'target_path': target_path,
                   'dlc_path': paths.dlc_script,
                   'denoising_path': paths.ca_denoising_script,
                   'cnmfe_path': paths.calcium_script,
                   'ca_reg_path': paths.update_motion_path,
                   'interval': [2, 3],
                   'analysis_type': parsed_search['analysis_type'],
                   }

    # write the file
    with open(paths.snakemake_config, 'w') as f:
        target_file = yaml.dump(config_dict, f)

    # assemble the output path
    if parsed_search['analysis_type'] == 'preprocessing_run':
        # feed the generic txt file for preprocessing
        out_path = os.path.join(paths.analysis_path, 'preprocessing_run.txt')
    elif parsed_search['analysis_type'] == 'tuning_run':
        out_path = os.path.join(paths.analysis_path, 'tuning_run.txt')
    elif parsed_search['analysis_type'] == 'mine_run':
        out_path = os.path.join(paths.analysis_path, 'mine_run.txt')
    elif parsed_search['analysis_type'] == 'aggregate_run':
        out_path = os.path.join(paths.analysis_path, 'aggregate_run.txt')
    elif parsed_search['analysis_type'] == 'update_matches_run':
        out_path = os.path.join(paths.analysis_path, 'update_cells_match_run.txt')
    elif parsed_search['analysis_type'] == 'update_vis_tcs_run':
        out_path = os.path.join(paths.analysis_path, 'update_vis_tc_run.txt')
    elif parsed_search['analysis_type'] == 'update_kinem_tcs_run':
        out_path = os.path.join(paths.analysis_path, 'update_kinem_tc_run.txt')
    elif parsed_search['analysis_type'] == 'update_ca_reg_run':
        out_path = os.path.join(paths.analysis_path, 'ca_reg_run.txt')
    else:
        # feed the aggregation path
        out_path = os.path.join(paths.analysis_path, '_'.join(('preprocessing', *parsed_search.values())) + '.hdf5')

    # run snakemake
    preprocess_sp = sp.Popen(['snakemake', out_path, out_path, '--cores', #'1',
                              '-s', paths.snakemake_scripts,
                              '-d', paths.snakemake_working,
                              # '--use-conda',
                              # '-F',         # (hard) force rerun everything
                              '-f',         # (soft) force rerun last step
                              # '--unlock',   # unlocks the files after force quit
                              # '--rerun-incomplete',
                              # '--touch',    # updates output file timestamp, but doesn't process
                              # '--verbose',    # make the output more verbose for debugging
                              # '--debug-dag',  # show the file selection operation, also for debugging
                              # '--dryrun',   # generates the DAG and everything, but doesn't process,
                              # '--reason'  ,   # print the reason for executing each job
                              ],
                             stdout=sp.PIPE,
                             )

    stdout = preprocess_sp.communicate()[0]
    print(stdout.decode())

    # assemble the output path
    if (parsed_search['analysis_type'] == 'preprocessing_run') &  \
            (os.path.isfile(os.path.join(paths.analysis_path, 'preprocessing_run.txt'))):
        # delete the txt file (specify de novo to not run risks)
        os.remove(os.path.join(paths.analysis_path, 'preprocessing_run.txt'))
    elif (parsed_search['analysis_type'] == 'tuning_run') & \
            (os.path.isfile(os.path.join(paths.analysis_path, 'tuning_run.txt'))):
        # delete the txt file (specify de novo to not run risks)
        os.remove(os.path.join(paths.analysis_path, 'tuning_run.txt'))
    elif (parsed_search['analysis_type'] == 'mine_run') & \
            (os.path.isfile(os.path.join(paths.analysis_path, 'mine_run.txt'))):
        # delete the txt file (specify de novo to not run risks)
        os.remove(os.path.join(paths.analysis_path, 'mine_run.txt'))
    elif (parsed_search['analysis_type'] == 'aggregate_run') &  \
            (os.path.isfile(os.path.join(paths.analysis_path, 'aggregate_run.txt'))):
        # delete the txt file (specify de novo to not run risks)
        os.remove(os.path.join(paths.analysis_path, 'aggregate_run.txt'))
    elif (parsed_search['analysis_type'] == 'update_matches_run') & \
         (os.path.isfile(os.path.join(paths.analysis_path, 'update_cells_match_run.txt'))):
        os.remove(os.path.join(paths.analysis_path, 'update_cells_match_run.txt'))
    elif (parsed_search['analysis_type'] == 'update_vis_tcs_run') & \
            (os.path.isfile(os.path.join(paths.analysis_path, 'update_vis_tc_run.txt'))):
        os.remove(os.path.join(paths.analysis_path, 'update_vis_tc_run.txt'))
    elif (parsed_search['analysis_type'] == 'update_kinem_tcs_run') & \
            (os.path.isfile(os.path.join(paths.analysis_path, 'update_kinem_tc_run.txt'))):
        os.remove(os.path.join(paths.analysis_path, 'update_kinem_tc_run.txt'))
    elif (parsed_search['analysis_type'] == 'update_ca_reg_run') & \
            (os.path.isfile(os.path.join(paths.analysis_path, 'ca_reg_run.txt'))):
        os.remove(os.path.join(paths.analysis_path, 'ca_reg_run.txt'))

print('yay')
