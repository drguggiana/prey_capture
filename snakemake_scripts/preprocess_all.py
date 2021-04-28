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
from pandas import read_hdf
import processing_parameters


def preprocess_selector(csv_path, saving_path, file_info):
    """functions that selects the preprocessing function for the first step, either dlc or not"""
    # check if the input has a dlc path or not
    if (len(file_info['dlc_path']) > 0 and file_info['dlc_path'] != 'N/A') or \
            os.path.isfile(file_info['bonsai_path'].replace('.csv', '_dlc.h5')):
        # assemble the path here, in case the file wasn't in the database
        dlc_path = file_info['bonsai_path'].replace('.csv', '_dlc.h5')
        # if there's a dlc file, use this preprocessing
        output_path, traces, corner_out, frame_b = \
            s1.run_dlc_preprocess(csv_path, dlc_path, saving_path, file_info)
    else:
        # if not, use the legacy non-dlc preprocessing
        output_path, traces = s1.run_preprocess(csv_path, saving_path, file_info)
        # set corners to empty
        corner_out = []
        # set frame bounds to empty
        frame_b = []
    return output_path, traces, corner_out, frame_b


# check if launched from snakemake, otherwise, prompt user
try:
    # get the path to the file, parse and turn into a dictionary
    raw_path = snakemake.input[0]
    calcium_path = snakemake.input[1]
    files = yaml.load(snakemake.params.info, Loader=yaml.FullLoader)
    # get the save paths
    save_path = snakemake.output[0]
    pic_path = snakemake.output[1]
except NameError:
    # USE FOR DEBUGGING ONLY (need to edit the search query and the object selection)
    # define the search string
    search_string = processing_parameters.search_string

    # define the target model
    if 'miniscope' in search_string:
        target_model = 'video_experiment'
    else:
        target_model = 'vr_experiment'

    # get the queryset
    files = bd.query_database(target_model, search_string)[0]
    raw_path = files['bonsai_path']
    calcium_path = files['bonsai_path'][:-4] + '_calcium.hdf5'
    # assemble the save paths
    save_path = os.path.join(paths.analysis_path,
                             os.path.basename(files['bonsai_path'][:-4]))+'_preproc.hdf5'
    pic_path = os.path.join(save_path[:-13] + '.png')

# get the file date
file_date = datetime.datetime.strptime(files['date'], '%Y-%m-%dT%H:%M:%SZ')

# decide the analysis path based on the file name and date
# if miniscope but no imaging, run bonsai only
if (files['rig'] == 'miniscope') and (files['imaging'] == 'no'):
    # run the first stage of preprocessing
    out_path, filtered_traces, corners, frame_bounds = preprocess_selector(files['bonsai_path'], save_path, files)
    # save the bounds
    frame_bounds.to_hdf(out_path, key='frame_bounds', mode='a', format='table')
    # define the dimensions of the arena
    reference_coordinates = paths.arena_coordinates[files['rig']]
    # scale the traces accordingly
    filtered_traces, corners = fp.rescale_pixels(filtered_traces, files, reference_coordinates, corners)

    # run the preprocessing kinematic calculations
    kinematics_data, real_crickets, vr_crickets = s2.kinematic_calculations(out_path, filtered_traces)

# if miniscope regular, run with the matching of miniscope frames
elif files['rig'] == 'miniscope' and (files['imaging'] == 'doric'):
    # run the first stage of preprocessing
    # out_path, filtered_traces = s1.run_preprocess(files['bonsai_path'],
    #                                               save_path)
    out_path, filtered_traces, corners, frame_bounds = preprocess_selector(files['bonsai_path'], save_path, files)

    # save the bounds
    frame_bounds.to_hdf(out_path, key='frame_bounds', mode='a', format='table')

    # define the dimensions of the arena
    reference_coordinates = paths.arena_coordinates[files['rig']]
    # scale the traces accordingly
    filtered_traces, corners = fp.rescale_pixels(filtered_traces, files, reference_coordinates, corners)

    # run the preprocessing kinematic calculations
    kinematics_data, real_crickets, vr_crickets = s2.kinematic_calculations(out_path, filtered_traces)

    # # trim the succ trials after cricket capture (define threshold in cm)
    # filtered_traces = fp.trim_at_last_encounter(files['result'], filtered_traces, threshold=4)

    # find the sync file
    sync_path = files['sync_path']

    # get a dataframe with the calcium data matched to the bonsai data
    matched_calcium = functions_matching.match_calcium(calcium_path, sync_path, kinematics_data, frame_bounds)

    matched_calcium.to_hdf(out_path, key='matched_calcium', mode='a', format='table')

# elif files['rig'] == 'VR' and file_date <= datetime.datetime(year=2019, month=11, day=10):
elif files['rig'] in ['VR', 'VPrey'] and file_date <= datetime.datetime(year=2020, month=6, day=22):

    # run the first stage of preprocessing
    # out_path, filtered_traces = s1.run_preprocess(files['bonsai_path'],
    #                                               save_path)
    out_path, filtered_traces, corners, _ = preprocess_selector(files['bonsai_path'], save_path, files)

    # define the dimensions of the arena
    reference_coordinates = paths.arena_coordinates['VR']

    # TODO: add the old motive-bonsai alignment as a function

    # # placeholder corners list
    # corners = []

    # run the preprocessing kinematic calculations
    kinematics_data, real_crickets, vr_crickets = s2.kinematic_calculations(out_path, filtered_traces)

elif files['rig'] in ['VR', 'VPrey'] and \
        datetime.datetime(year=2020, month=6, day=23) <= file_date <= datetime.datetime(year=2020, month=7, day=20):

    # TODO: make sure the constants are set to values that make sense for the vr arena

    # get the video tracking data
    out_path, filtered_traces, _, _ = preprocess_selector(files['bonsai_path'], save_path, files)

    # get the motive tracking data
    motive_traces, _, _ = s1.extract_motive(files['track_path'], files['rig'])

    # define the dimensions of the arena
    reference_coordinates = paths.arena_coordinates['VR']
    manual_coordinates = paths.arena_coordinates['VR_manual']

    # scale the traces accordingly
    filtered_traces, corners = \
        fp.rescale_pixels(filtered_traces, files, reference_coordinates, manual_coordinates=manual_coordinates)

    # align them temporally based on the sync file
    filtered_traces = functions_matching.match_motive(motive_traces, files['sync_path'], filtered_traces)

    # align the data spatially
    filtered_traces = functions_matching.align_spatial(filtered_traces)

    # run the preprocessing kinematic calculations
    kinematics_data, real_crickets, vr_crickets = s2.kinematic_calculations(out_path, filtered_traces)

elif files['rig'] in ['VScreen']:

    # load the data for the trial structure and parameters
    trials = read_hdf(files['screen_path'], key='trial_set')
    params = read_hdf(files['screen_path'], key='params')

    # get the video tracking data
    out_path, filtered_traces, _, _ = preprocess_selector(files['bonsai_path'], save_path, files)

    # define the dimensions of the arena
    manual_coordinates = paths.arena_coordinates['VR_manual']

    # get the motive tracking data
    motive_traces, reference_coordinates, obstacle_coordinates = \
        s1.extract_motive(files['track_path'], files['rig'])

    # scale the traces accordingly
    filtered_traces, corners = \
        fp.rescale_pixels(filtered_traces, files, reference_coordinates, manual_coordinates=manual_coordinates)

    # align them temporally based on the sync file
    filtered_traces = functions_matching.match_motive(motive_traces, files['sync_path'], filtered_traces)

    # run the preprocessing kinematic calculations
    # also saves the data
    kinematics_data, real_crickets, vr_crickets = s2.kinematic_calculations(out_path, filtered_traces)

    # For these trials, save the trial set and the trial parameters to the output file
    trials.to_hdf(out_path, key='trial_set', mode='a')
    params.to_hdf(out_path, key='params', mode='a')

else:
    # TODO: make sure the constants are set to values that make sense for the vr arena
    # run the first stage of preprocessing
    # out_path, filtered_traces = s1.run_preprocess(files['bonsai_path'],
    #                                               save_path)

    # get the video tracking data
    out_path, filtered_traces, _, _ = preprocess_selector(files['bonsai_path'], save_path, files)

    # define the dimensions of the arena
    reference_coordinates = paths.arena_coordinates['VR']
    manual_coordinates = paths.arena_coordinates['VR_manual']

    # scale the traces accordingly
    filtered_traces, corners = \
        fp.rescale_pixels(filtered_traces, files, reference_coordinates, manual_coordinates=manual_coordinates)

    # get the motive tracking data
    motive_traces, _, _ = s1.extract_motive(files['track_path'], files['rig'])

    # align them temporally based on the sync file
    filtered_traces = functions_matching.match_motive(motive_traces, files['sync_path'], filtered_traces)

    # run the preprocessing kinematic calculations
    kinematics_data, real_crickets, vr_crickets = s2.kinematic_calculations(out_path, filtered_traces)


# save the filtered trace
fig_final = plt.figure()
ax = fig_final.add_subplot(111)
# plt.gca().invert_xaxis()
# plt.gca().invert_yaxis()

# plot the filtered trace
a = ax.scatter(filtered_traces.mouse_x, filtered_traces.mouse_y,
               c=filtered_traces.time_vector, marker='o', linestyle='-', cmap='Blues')
cbar = fig_final.colorbar(a, ax=ax)
cbar.set_label('Time (s)')
ax.axis('equal')

# for all the real crickets
for real_cricket in range(real_crickets):
    ax.scatter(filtered_traces['cricket_'+str(real_cricket)+'_x'],
               filtered_traces['cricket_'+str(real_cricket)+'_y'],
               c=filtered_traces.time_vector, marker='o', linestyle='-', cmap='Oranges')

# for all the virtual crickets or virtual targets
for vr_cricket in range(vr_crickets):
    try:
        ax.scatter(filtered_traces['vrcricket_' + str(vr_cricket) + '_x'],
                   filtered_traces['vrcricket_' + str(vr_cricket) + '_y'],
                   c=filtered_traces.time_vector, marker='o', linestyle='-', cmap='Greens')
    except:
        ax.scatter(filtered_traces['target_x_m'], filtered_traces['target_y_m'],
                   c=filtered_traces.time_vector, marker='o', linestyle='-', cmap='Greens')

# plot the found corners if existent
if len(corners) > 0:
    for corner in corners:
        ax.scatter(corner[0], corner[1], c='black')

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
    'video_analysis': [files['url']] if files['rig'] == 'miniscope' else [],
    'vr_analysis': [] if files['rig'] == 'miniscope' else [files['url']],
}

# Assemble the notes section for the entry data
if '_vrcrickets_' in files['notes']:
    # Add nothing to the notes section. It's already been done
    entry_data['notes'] = files['notes']
elif files['notes'] == 'BLANK':
    # If it's blank, then add
    entry_data['notes'] = 'crickets_' + str(real_crickets) + '_vrcrickets_' + str(vr_crickets)
else:
    entry_data['notes'] = files['notes'] + '_crickets_' + str(real_crickets) + '_vrcrickets_' + str(vr_crickets),

# if the notes field hasn't been updated
if '_vrcrickets_' not in files['notes']:
    # update the notes field from the original file
    url_original = files['url']
    update_notes = {'notes': entry_data['notes'],
                    'mouse': '/'.join((paths.bondjango_url, 'mouse', files['mouse']))+'/',
                    'experiment_type': ['/'.join((paths.bondjango_url, 'experiment_type',
                                                  files['experiment_type'][0]))+'/'],
                    }
    update_original = bd.update_entry(url_original, update_notes)
    if update_original.status_code == 404 or update_original.status_code == 400:
        print('Original entry for {} was not updated'.format(files['slug']))

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
