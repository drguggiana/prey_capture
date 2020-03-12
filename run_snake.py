import subprocess as sp
import yaml
import functions_bondjango as bd
import paths
import functions_data_handling as fd
import os


# define the type of analysis
analysis_type = 'aggBin'
search_query = 'result:succ,rig:miniscope'

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

# get the target database entries
target_entries = bd.query_database(target_model, search_query)
# also parse the search string
parsed_search = fd.parse_search_string('analysis_type:'+analysis_type+','+search_query)

# create the config file
config_dict = {'files': {os.path.basename(el['bonsai_path'])[:-4]: os.path.basename(el['bonsai_path'])[:-4]
                         for el in target_entries}, 'target_path': target_path}
with open(paths.snakemake_config, 'w') as f:
    target_file = yaml.dump(config_dict, f)

# assemble the output path
out_path = os.path.join(paths.analysis_path, '_'.join(('preprocessing', *parsed_search.values())) + '.hdf5')

# run snakemake
preprocess_sp = sp.Popen(['snakemake', out_path, '-F',
                          '-s', paths.snakemake_scripts,
                          '-d', paths.snakemake_working],
                         stdout=sp.PIPE)

stdout = preprocess_sp.communicate()[0]
print(stdout.decode())
print('yay')
