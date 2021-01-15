import subprocess as sp
import yaml
import functions_bondjango as bd
import paths
import functions_data_handling as fd
import os


# define the type of analysis
input_dictionary = {
    # 'analysis_type': ['aggBin', 'aggFull', 'aggEnc', 'aggBinCA', 'aggFullCA', 'aggEncCA', 'trigAveCA'],
    # 'analysis_type': ['trigAveCA'],
    # 'analysis_type': ['aggBin', 'aggFull', 'aggEnc'],
    'analysis_type': ['just_preprocess'],
    'result': ['test', ],
    # 'result': ['test', 'succ'],
    # 'rig': ['VPrey', 'VR', ],
    # 'rig': ['miniscope', ],
    'rig': ['VScreen'],
    # 'lighting': ['normal', ],
    # 'gtdate': ['2020-08-24T00-00-00'],
    # 'gtdate': ['2020-06-23T00-00-00'],
    # 'ltdate': ['2020-07-06T00-00-00'],
    # 'notes': ['crickets_1_vrcrickets_1', 'crickets_1_vrcrickets_3',
    #           'crickets_0_vrcrickets_1', 'crickets_0_vrcrickets_3'
    #           ]
    # 'notes': ['blackCr_crickets_1_vrcrickets_1', 'blackCr_rewarded_crickets_0_vrcrickets_1',
    #           'blackCr_nonrewarded_crickets_0_vrcrickets_1',
    #           'blackCr_crickets_1_vrcrickets_3', 'blackCr_rewarded_crickets_0_vrcrickets_3',
    #           'blackCr_nonrewarded_crickets_0_vrcrickets_3',
    #           'blackCr_crickets_1_vrcrickets_0',
    #           'blackCr_grayBG_crickets_1_vrcrickets_1', 'blackCr_grayBG_rewarded_crickets_0_vrcrickets_1',
    #           'blackCr_grayBG_crickets_1_vrcrickets_3', 'blackCr_grayBG_rewarded_crickets_0_vrcrickets_3',
    #           'whiteCr_blackBG_crickets_1_vrcrickets_1', 'whiteCr_blackBG_rewarded_crickets_0_vrcrickets_1',
    #           'whiteCr_blackBG_crickets_1_vrcrickets_3', 'whiteCr_blackBG_rewarded_crickets_0_vrcrickets_3',
    #           'whiteCr_grayBG_crickets_1_vrcrickets_1', 'whiteCr_grayBG_rewarded_crickets_0_vrcrickets_1',
    #           'whiteCr_grayBG_crickets_1_vrcrickets_3', 'whiteCr_grayBG_rewarded_crickets_0_vrcrickets_3',
    #           'obstacle_crickets_1_vrcrickets_1', 'obstacle_rewarded_crickets_0_vrcrickets_1',
    #           'obstacle_crickets_1_vrcrickets_3', 'obstacle_rewarded_crickets_0_vrcrickets_3',
    #           'blackCr_crickets_1',
    #           'blackCr_rewarded',
    #           'blackCr_grayBG_crickets_1',
    #           'blackCr_grayBG_rewarded',
    #           'whiteCr_blackBG_crickets_1',
    #           'whiteCr_blackBG_rewarded',
    #           'whiteCr_grayBG_crickets_1',
    #           'whiteCr_grayBG_rewarded',
    #           'crickets_1',
    #           'crickets_0',
    #           ],
    # 'notes': ['obstacle_crickets_1_vrcrickets_1', 'obstacle_rewarded_crickets_0_vrcrickets_1',
    #           'obstacle_crickets_1_vrcrickets_3', 'obstacle_rewarded_crickets_0_vrcrickets_3',
    #           ]
    # 'notes': ['blackCr_crickets_1',
    #           'blackCr_rewarded',
    #           'blackCr_grayBG_crickets_1',
    #           'blackCr_grayBG_rewarded',
    #           'whiteCr_blackBG_crickets_1',
    #           'whiteCr_blackBG_rewarded',
    #           'whiteCr_grayBG_crickets_1',
    #           'whiteCr_grayBG_rewarded',
    #           ]

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
                   'dlc_flag': {os.path.basename(el['bonsai_path'])[:-4]: True if len(el['avi_path']) > 0
                                else False for el in target_entries},
                   # 'dlc_flag': {os.path.basename(el['bonsai_path'])[:-4]: False for el in target_entries},
                   'output_info': yaml.dump(parsed_search),
                   'target_path': target_path,
                   'dlc_path': paths.dlc_script,
                   'interval': [2, 3],
                   }
    # write the file
    with open(paths.snakemake_config, 'w') as f:
        target_file = yaml.dump(config_dict, f)

    # assemble the output path
    if parsed_search['analysis_type'] == 'just_preprocess':
        # feed the generic txt file for preprocessing
        out_path = os.path.join(paths.analysis_path, 'just_preprocess.txt')
    else:
        # feed the aggregation path
        out_path = os.path.join(paths.analysis_path, '_'.join(('preprocessing', *parsed_search.values())) + '.hdf5')

    # out_path = os.path.join(paths.figures_path, '_'.join(('averages', *parsed_search.values())) + '.html')

    # run snakemake
    preprocess_sp = sp.Popen(['snakemake', out_path, out_path, '--cores', '1',
                              # '-F',         # (hard) force rerun everything
                              '-f',         # (soft) force rerun last step
                              # '--unlock',   # unlocks the files after force quit
                              # '--rerun-incomplete',
                              '-s', paths.snakemake_scripts,
                              '-d', paths.snakemake_working],
                             stdout=sp.PIPE)

    stdout = preprocess_sp.communicate()[0]
    print(stdout.decode())

    # assemble the output path
    if parsed_search['analysis_type'] == 'just_preprocess':
        # delete the just_preprocess file (specify de novo to no run risks)
        os.remove(os.path.join(paths.analysis_path, 'just_preprocess.txt'))
print('yay')
