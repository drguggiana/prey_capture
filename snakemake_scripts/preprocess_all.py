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
    if len(file_info['dlc_path']) > 0 and file_info['dlc_path'] != 'N/A':
        # if there's a dlc file, use this preprocessing
        output_path, traces = s1.run_dlc_preprocess(csv_path, file_info['dlc_path'], saving_path, file_info)
    else:
        # if not, use the legacy non-dlc preprocessing
        output_path, traces = s1.run_preprocess(csv_path, saving_path, file_info)
    return output_path, traces


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
    # search_string = 'slug:11_11_2019_15_02_31_DG_190417_a_succ'
    # search_string = 'slug:03_04_2020_15_54_26_miniscope_mm_200129_a_succ'
    # search_string = 'slug:07_17_2020_16_24_31_dg_200526_d_fail_dark'
    # search_string = 'slug:08_24_2020_11_24_27_VPrey_DG_200526_b_test_whiteCr_blackBG_rewarded'
    search_string = 'slug:03_12_2020_16_52_33_miniscope_MM_200129_b_succ'

    # search_string = 'result:fail, lighting:normal, rig:miniscope'

    # search_string = 'slug:11_25_2019_15_28_57_miniscope_MM_191106_a_fail_nomini'
    # search_string = 'slug:03_05_2020_15_56_12_miniscope_MM_200129_b_succ'
    # search_string = 'slug:03_13_2020_13_20_21_miniscope_MM_200129_a_succ'
    # search_string = 'slug:12_04_2019_15_56_34_miniscope_MM_191107_a_succ_nomini'

    # define the target model
    target_model = 'video_experiment'
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
    out_path, filtered_traces = preprocess_selector(files['bonsai_path'], save_path, files)

    # define the dimensions of the arena
    reference_coordinates = paths.arena_coordinates[files['rig']]
    # scale the traces accordingly
    filtered_traces, corners = fp.rescale_pixels(filtered_traces, files, reference_coordinates)
    # corners = miniscope_arena

    # run the preprocessing kinematic calculations
    kinematics_data, real_crickets, vr_crickets = s2.kinematic_calculations(out_path, filtered_traces)

# if miniscope regular, run with the matching of miniscope frames
elif files['rig'] == 'miniscope' and (files['imaging'] == 'doric'):
    # run the first stage of preprocessing
    # out_path, filtered_traces = s1.run_preprocess(files['bonsai_path'],
    #                                               save_path)
    out_path, filtered_traces = preprocess_selector(files['bonsai_path'], save_path, files)

    # define the dimensions of the arena
    reference_coordinates = paths.arena_coordinates[files['rig']]
    # scale the traces accordingly
    # filtered_traces, corners = fp.rescale_pixels(filtered_traces, files, reference_coordinates)
    corners = []
    # run the preprocessing kinematic calculations
    kinematics_data, real_crickets, vr_crickets = s2.kinematic_calculations(out_path, filtered_traces)

    # # get the calcium file path
    # calcium_path = files['fluo_path']

    # find the sync file
    sync_path = files['sync_path']

    # get a dataframe with the calcium data matched to the bonsai data
    matched_calcium = functions_matching.match_calcium(calcium_path, sync_path, kinematics_data)

    matched_calcium.to_hdf(out_path, key='matched_calcium', mode='a', format='table')

# elif files['rig'] == 'VR' and file_date <= datetime.datetime(year=2019, month=11, day=10):
elif files['rig'] in ['VR', 'VPrey'] and file_date <= datetime.datetime(year=2020, month=6, day=23):

    # run the first stage of preprocessing
    # out_path, filtered_traces = s1.run_preprocess(files['bonsai_path'],
    #                                               save_path)
    out_path, filtered_traces = preprocess_selector(files['bonsai_path'], save_path, files)

    # define the dimensions of the arena
    reference_coordinates = paths.arena_coordinates['VR']

    # TODO: add the old motive-bonsai alignment as a function

    # placeholder corners list
    corners = []

    # run the preprocessing kinematic calculations
    kinematics_data, real_crickets, vr_crickets = s2.kinematic_calculations(out_path, filtered_traces)

else:
    # TODO: make sure the constants are set to values that make sense for the vr arena
    # run the first stage of preprocessing
    # out_path, filtered_traces = s1.run_preprocess(files['bonsai_path'],
    #                                               save_path)

    # get the video tracking data
    out_path, filtered_traces = preprocess_selector(files['bonsai_path'], save_path, files)

    # define the dimensions of the arena
    reference_coordinates = paths.arena_coordinates['VR']
    manual_coordinates = paths.arena_coordinates['VR_manual']

    # scale the traces accordingly
    filtered_traces, corners = \
        fp.rescale_pixels(filtered_traces, files, reference_coordinates, manual_coordinates=manual_coordinates)

    # get the motive tracking data
    motive_traces = s1.extract_motive(files['track_path'], files['rig'])
    # align them temporally based on the sync file
    # TODO add spatial matching
    filtered_traces = functions_matching.match_motive(motive_traces, files['sync_path'], filtered_traces)
    # align the data spatially
    # filtered_traces = functions_matching.align_spatial(filtered_traces)

    # run the preprocessing kinematic calculations
    kinematics_data, real_crickets, vr_crickets = s2.kinematic_calculations(out_path, filtered_traces)


# save the filtered trace
fig_final = plt.figure()
ax = fig_final.add_subplot(111)
plt.gca().invert_yaxis()

# plot the filtered trace
ax.plot(filtered_traces.mouse_x,
        filtered_traces.mouse_y, marker='o', linestyle='-')
ax.axis('equal')
# plot the found corners if existent
if len(corners) > 0:
    for corner in corners:
        ax.scatter(corner[0], corner[1], c='black')
# for all the real crickets
for real_cricket in range(real_crickets):
    ax.plot(filtered_traces['cricket_'+str(real_cricket)+'_x'],
            filtered_traces['cricket_'+str(real_cricket)+'_y'], marker='o', linestyle='-')

# for all the virtual crickets
for vr_cricket in range(vr_crickets):
    ax.plot(filtered_traces['vrcricket_'+str(vr_cricket)+'_x'],
            filtered_traces['vrcricket_'+str(vr_cricket)+'_y'], marker='o', linestyle='-')

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
    'notes': 'crickets_' + str(real_crickets) + '_vrcrickets_' + str(vr_crickets) if files['notes'] == 'BLANK' else
             files['notes'] + '_crickets_' + str(real_crickets) + '_vrcrickets_' + str(vr_crickets),
    'video_analysis': [files['url']] if files['rig'] == 'miniscope' else [],
    'vr_analysis': [] if files['rig'] == 'miniscope' else [files['url']],
}

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
