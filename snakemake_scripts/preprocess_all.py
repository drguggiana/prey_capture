# imports
import functions_matching
import functions_preprocessing as fp
import paths
import snakemake_scripts.sub_preprocess_S1 as s1
import snakemake_scripts.sub_preprocess_S2 as s2
import datetime
import functions_bondjango as bd
import os
import yaml
import matplotlib.pyplot as plt


def preprocess_selector(csv_path, saving_path, file_info):
    """functions that selects the preprocessing function for the first step, either dlc or not"""
    # check if the input has a dlc path or not
    if len(file_info['dlc_path']) > 0:
        # if there's a dlc file, use this preprocessing
        output_path, traces = s1.run_dlc_preprocess(csv_path, file_info['dlc_path'], saving_path)
    else:
        # if not, use the legacy non-dlc preprocessing
        output_path, traces = s1.run_preprocess(csv_path, saving_path)
    return output_path, traces


# check if launched from snakemake, otherwise, prompt user
try:
    # get the path to the file, parse and turn into a dictionary
    raw_path = snakemake.input[0]
    files = yaml.load(snakemake.params.info, Loader=yaml.FullLoader)
    # get the save paths
    save_path = snakemake.output[0]
    pic_path = snakemake.output[1]
except NameError:
    # USE FOR DEBUGGING ONLY (need to edit the search query and the object selection)
    # define the search string
    search_string = 'slug:03_04_2020_16_16_18_miniscope_MM_200129_b_succ'
    # search_string = 'result:succ, lighting:normal, rig:miniscope'
    # define the target model
    target_model = 'video_experiment'
    # get the queryset
    files = bd.query_database(target_model, search_string)[0]
    raw_path = files['bonsai_path']
    # assemble the save paths
    save_path = os.path.join(paths.analysis_path,
                             os.path.basename(files['bonsai_path'][:-4]))+'_preproc.hdf5'
    pic_path = os.path.join(save_path[:-13] + '.png')

# get the file date
file_date = files['date']

# decide the analysis path based on the file name and date
# if miniscope but no imaging, run bonsai only
if (files['rig'] == 'miniscope') and (files['imaging'] == 'no'):
    # run the first stage of preprocessing
    # out_path, filtered_traces = s1.run_preprocess(files['bonsai_path'],
    #                                               save_path)
    out_path, filtered_traces = preprocess_selector(files['bonsai_path'], save_path, files)
    # TODO: add corner detection to calibrate the coordinate to real size
    # in the meantime, add a rough manual correction based on the size of the arena and the number of pixels
    # run the preprocessing kinematic calculations
    kinematics_data = s2.kinematic_calculations(out_path, filtered_traces)

# if miniscope regular, run with the matching of miniscope frames
elif files['rig'] == 'miniscope' and (files['imaging'] == 'doric'):
    # run the first stage of preprocessing
    # out_path, filtered_traces = s1.run_preprocess(files['bonsai_path'],
    #                                               save_path)
    out_path, filtered_traces = preprocess_selector(files['bonsai_path'], save_path, files)
    [] = fp.rescale_pixels(filtered_traces, files)

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
    # out_path, filtered_traces = s1.run_preprocess(files['bonsai_path'],
    #                                               save_path)
    out_path, filtered_traces = preprocess_selector(files['bonsai_path'], save_path, files)

    # TODO: add the old motive-bonsai alignment as a function

    # run the preprocessing kinematic calculations
    kinematics_data = s2.kinematic_calculations(out_path, paths.kinematics_path)
else:
    # TODO: make sure the constants are set to values that make sense for the vr arena
    # run the first stage of preprocessing
    # out_path, filtered_traces = s1.run_preprocess(files['bonsai_path'],
    #                                               save_path)
    out_path, filtered_traces = preprocess_selector(files['bonsai_path'], save_path, files)

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
    'date': files['date'],
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
