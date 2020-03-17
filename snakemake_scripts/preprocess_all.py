# imports
import functions_matching
import paths
import snakemake_scripts.sub_preprocess_S1 as s1
import snakemake_scripts.sub_preprocess_S2 as s2
import datetime
import functions_bondjango as bd
import functions_data_handling as fd

# define the search string
# # search_string = 'result=succ, lighting=normal'
# target_model = 'video_experiment'
# # get the queryset
# file_path_bonsai = bd.query_database(target_model)
# file_path_bonsai = bd.query_database(target_model, search_string)

# get the path to the file, parse and turn into a dictionary
raw_path = snakemake.input[0]

# find the entry based on the path
# files = bd.query_database(target_model, 'bonsai_path:'+raw_path)[0]
files = fd.parse_experiment_name(raw_path)
# get the target model from the path
target_model = 'video_experiment' if files['rig'] == 'miniscope' else 'vr_experiment'

# run through the files and select the appropriate analysis script

# get the file date
# file_date = functions_io.get_file_date(files)
# file_date = datetime.datetime.strptime(files['date'], '%Y-%m-%dT%H:%M:%SZ')
file_date = datetime.datetime.strptime(files['date'], '%m_%d_%Y_%H_%M_%S')

# check for the nomini flag
if ('nomini' in files['notes']) or ('nofluo' in files['notes']):
    nomini_flag = True
else:
    nomini_flag = False

# assemble the save path
# save_path = os.path.join(paths.analysis_path, os.path.basename(files['bonsai_path']))
save_path = snakemake.output[0]
# decide the analysis path based on the file name and date
# if miniscope with the _nomini flag, run bonsai only
if files['rig'] == 'miniscope' and nomini_flag:
    # run the first stage of preprocessing
    out_path, filtered_traces = s1.run_preprocess(files['bonsai_path'],
                                                            save_path,
                                                            ['cricket_x', 'cricket_y'])
    # TODO: add corner detection to calibrate the coordinate to real size
    # in the meantime, add a rough manual correction based on the size of the arena and the number of pixels

    # run the preprocessing kinematic calculations
    kinematics_data = s2.kinematic_calculations(out_path, filtered_traces)

# if miniscope regular, run with the matching of miniscope frames
elif files['rig'] == 'miniscope' and not nomini_flag:
    # run the first stage of preprocessing
    out_path, filtered_traces = s1.run_preprocess(files['bonsai_path'],
                                                            save_path,
                                                            ['cricket_x', 'cricket_y'])

    # run the preprocessing kinematic calculations
    kinematics_data = s2.kinematic_calculations(out_path, filtered_traces)

    # get the calcium file path
    calcium_path = files['fluo_path']

    # find the sync file
    sync_path = files['sync_path']

    # get a dataframe with the calcium data matched to the bonsai data
    matched_calcium = functions_matching.match_calcium(calcium_path, sync_path, kinematics_data)

    matched_calcium.to_hdf(out_path, key='matched_calcium', mode='a', format='table')

elif files['rig'] != 'miniscope' and file_date <= datetime.datetime(year=2019, month=11, day=10):
    # run the first stage of preprocessing
    out_path, filtered_traces = s1.run_preprocess(files['bonsai_path'],
                                                            save_path,
                                                            ['cricket_x', 'cricket_y'])

    # TODO: add the old motive-bonsai alignment as a function

    # run the preprocessing kinematic calculations
    kinematics_data = s2.kinematic_calculations(out_path, paths.kinematics_path)
else:
    # TODO: make sure the constants are set to values that make sense for the vr arena
    # run the first stage of preprocessing
    out_path, filtered_traces = s1.run_preprocess(files['bonsai_path'],
                                                            save_path,
                                                            ['cricket_x', 'cricket_y'])

    # run the preprocessing kinematic calculations
    kinematics_data = s2.kinematic_calculations(out_path, paths.kinematics_path)

    # pic_path = ''
    # out_path = []

# assemble the entry data
# TODO: make sure it's compatible also with VR experiments
entry_data = {
    'analysis_type': 'preprocessing',
    'analysis_path': out_path,
    'pic_path': '',
    'result': files['result'],
    'rig': files['rig'],
    'lighting': files['lighting'],
    'slug': files['slug'] + '_preprocessing',
    'notes': files['notes'],
    'video_analysis': [files['url']] if files['rig'] == 'miniscope' else [],
    'vr_analysis': [] if files['rig'] == 'miniscope' else [files['url']],
}
# check if the entry already exists, if so, update it, otherwise, create it
update_url = '/'.join((paths.bondjango_url, 'analyzed_data', entry_data['slug'], ''))
output_entry = bd.update_entry(update_url, entry_data)
if output_entry.status_code == 404:
    # build the url for creating an entry
    create_url = '/'.join((paths.bondjango_url, 'analyzed_data', ''))
    output_entry = bd.create_entry(create_url, entry_data)

print('The output status was %i, reason %s' %
      (output_entry.status_code, output_entry.reason))
if output_entry.status_code in [500, 400]:
    print(entry_data)

# print('yay')
print('<3')
