# imports
import functions_matching
import paths
import snakemake_scripts.sub_preprocess_S1 as s1
import snakemake_scripts.sub_preprocess_S2 as s2
import datetime
import functions_bondjango as bd
import os
import yaml
import matplotlib.pyplot as plt

# check if launched from snakemake, otherwise, prompt user
try:
    # get the path to the file, parse and turn into a dictionary
    raw_path = snakemake.input[0]
    files = yaml.load(snakemake.params.info, Loader=yaml.FullLoader)
except NameError:
    # define the search string
    search_string = 'result=succ, lighting=normal'
    # define the target model
    target_model = 'video_experiment'
    # get the queryset
    files = bd.query_database(target_model, search_string)
    raw_path = files[0]['bonsai_path']

# get the target model from the path
target_model = 'video_experiment' if files['rig'] == 'miniscope' else 'vr_experiment'

# get the file date
file_date = files['date']

# assemble the save path
try:
    save_path = snakemake.output[0]
    pic_path = snakemake.output[1]
except NameError:
    save_path = os.path.join(paths.analysis_path,
                             os.path.basename(files['bonsai_path'][:-4]))+'_preproc.hdf5'
    pic_path = os.path.join(save_path[:-13] + '.png')

# decide the analysis path based on the file name and date
# if miniscope but no imaging, run bonsai only
if (files['rig'] == 'miniscope') and (files['imaging'] == 'no'):
    # run the first stage of preprocessing
    out_path, filtered_traces = s1.run_preprocess(files['bonsai_path'],
                                                  save_path,
                                                  ['cricket_x', 'cricket_y'])
    # TODO: add corner detection to calibrate the coordinate to real size
    # in the meantime, add a rough manual correction based on the size of the arena and the number of pixels

    # run the preprocessing kinematic calculations
    kinematics_data = s2.kinematic_calculations(out_path, filtered_traces)

# if miniscope regular, run with the matching of miniscope frames
elif files['rig'] == 'miniscope' and (files['imaging'] == 'doric'):
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

elif files['rig'] == 'vr' and file_date <= datetime.datetime(year=2019, month=11, day=10):
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

# save the filtered trace
fig_final = plt.figure()
ax = fig_final.add_subplot(111)
plt.gca().invert_yaxis()

# plot the filtered trace
ax.plot(filtered_traces.mouse_x,
        filtered_traces.mouse_y, marker='o', linestyle='-')
ax.plot(filtered_traces.cricket_x,
        filtered_traces.cricket_y, marker='o', linestyle='-')

# define the path for the figure
fig_final.savefig(pic_path, bbox_inches='tight')

# assemble the entry data
entry_data = {
    'analysis_type': 'preprocessing',
    'analysis_path': out_path,
    'pic_path': pic_path,
    'result': files['result'],
    'rig': files['rig'],
    'lighting': files['lighting'],
    'imaging': files['imaging'],
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

print('<3')
