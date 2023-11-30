# imports
import functions_matching as fm
import functions_preprocessing as fp
import functions_plotting as fplot
import paths
import snakemake_scripts.sub_preprocess_S1 as s1
import snakemake_scripts.sub_preprocess_S2 as s2
import datetime
import functions_bondjango as bd
import os
import yaml
import matplotlib.pyplot as plt
import h5py
from pandas import read_hdf, DataFrame
import processing_parameters


def preprocess_selector(ref_path, file_info):
    """functions that selects the preprocessing function for the first step, either dlc or not"""
    # check if the input has a dlc path or not
    if (len(file_info['dlc_path']) > 0 and file_info['dlc_path'] != 'N/A') or \
            os.path.isfile(file_info['avi_path'].replace('.avi', '_dlc.h5')):
        # assemble the path here, in case the file wasn't in the database
        dlc_path = file_info['avi_path'].replace('.avi', '_dlc.h5')
        # select function depending on the rig
        if files['rig'] in ['VWheel', 'VWheelWF'] :
            # use the eye specific function
            traces, corner_out, frame_b = s1.run_preprocess_eye(ref_path, dlc_path, file_info)
        else:
            # if there's a dlc file, use this preprocessing
            traces, corner_out, frame_b = s1.run_dlc_preprocess(ref_path, dlc_path, file_info)
    else:
        # if not, use the legacy non-dlc preprocessing
        output_path, traces = s1.run_preprocess(ref_path, file_info)
        # set corners to empty
        corner_out = []
        # set frame bounds to empty
        frame_b = []
    return traces, corner_out, frame_b


# check if launched from snakemake, otherwise, prompt user
try:
    # get the path to the file, parse and turn into a dictionary
    raw_path = snakemake.input[0]
    calcium_path = snakemake.input[1]
    match_path = snakemake.input[2]
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
    raw_path = files['avi_path']
    calcium_path = files['avi_path'][:-4] + '_calcium.hdf5'
    # match_path = os.path.join(paths.analysis_path, '_'.join((files['mouse'], files['rig'], 'cellMatch.hdf5')))
    match_path = os.path.join(paths.analysis_path, '_'.join((files['mouse'], 'cellMatch.hdf5')))
    # assemble the save paths
    save_path = os.path.join(paths.analysis_path,
                             os.path.basename(files['avi_path'][:-4]))+'_rawcoord.hdf5'
    pic_path = save_path[:-14] + '.png'

# get the file date
file_date = datetime.datetime.strptime(files['date'], '%Y-%m-%dT%H:%M:%SZ')

# initialize the cricket and trial variables
real_crickets = 0
vr_crickets = 0
trials = None
params = None

# decide the analysis path based on the file name and date
if files['rig'] == 'miniscope':
    # run the first stage of preprocessing
    # out_path, filtered_traces = s1.run_preprocess(files['bonsai_path'],
    #                                               save_path)
    filtered_traces, px_corners, frame_bounds = preprocess_selector(files['avi_path'], files)

    # scale the traces accordingly
    filtered_traces, corners = fp.rescale_pixels(filtered_traces, files,
                                                 paths.arena_coordinates[files['rig']], px_corners.to_numpy().T)
    # save the bounds and the matrix
    frame_bounds.to_hdf(save_path, key='frame_bounds', mode='w', format='fixed')
    px_corners.to_hdf(save_path, key='corners', mode='a', format='fixed')

    # run the preprocessing kinematic calculations
    kinematics_data, real_crickets, vr_crickets = s2.kinematic_calculations(filtered_traces)

    if files['imaging'] == 'doric':

        # get a dataframe with the calcium data matched to the bonsai data
        matched_calcium, roi_info = fm.match_calcium_2(calcium_path, files['sync_path'], kinematics_data)

        # if there is a calcium output, write to the file
        if matched_calcium is not None:
            matched_calcium.to_hdf(save_path, key='matched_calcium', mode='a', format='fixed')
            roi_info.to_hdf(save_path, key='roi_info', mode='a', format='fixed')
            # also get the cell matching if it exists
            cell_matches = fm.match_cells(match_path)
            cell_matches.to_hdf(save_path, key='cell_matches', mode='a', format='fixed')

elif files['rig'] in ['VTuning']:
    # load the data for the trial structure and parameters
    trials = read_hdf(files['screen_path'], key='trial_set')
    params = read_hdf(files['screen_path'], key='params')

    # get the video tracking data
    filtered_traces, px_corners, frame_bounds = preprocess_selector(files['avi_path'], files)

    # define the dimensions of the arena
    manual_coordinates = paths.arena_coordinates['VTuning']

    # get the motive tracking data
    motive_traces, reference_coordinates, obstacle_coordinates = \
        s1.extract_motive(files['track_path'], files['rig'])

    # scale the traces accordingly
    filtered_traces, corners = \
        fp.rescale_pixels(filtered_traces, files, manual_coordinates, px_corners.to_numpy().T)

    # align them temporally based on the sync file
    filtered_traces = fm.match_motive_2(motive_traces, files['sync_path'], filtered_traces)

    # run the preprocessing kinematic calculations
    # also saves the data
    kinematics_data, real_crickets, vr_crickets = s2.kinematic_calculations(filtered_traces)

    # For these trials, save the trial set and the trial parameters to the output file
    trials.to_hdf(save_path, key='trial_set', mode='a')
    params.to_hdf(save_path, key='params', mode='a')

    # calculate only if calcium is present
    if files['imaging'] == 'doric':
        # get a dataframe with the calcium data matched to the bonsai data
        matched_calcium, roi_info = fm.match_calcium_2(calcium_path, files['sync_path'], kinematics_data, trials=trials)
        # if there is a calcium output, write to the file
        if matched_calcium is not None:
            matched_calcium.to_hdf(save_path, key='matched_calcium', mode='a', format='fixed')
            roi_info.to_hdf(save_path, key='roi_info', mode='a', format='fixed')
            # also get the cell matching if it exists
            cell_matches = fm.match_cells(match_path)
            cell_matches.to_hdf(save_path, key='cell_matches', mode='a', format='fixed')

elif files['rig'] in ['VWheel']:
    # load the data for the trial structure and parameters
    trials = read_hdf(files['screen_path'], key='trial_set')
    params = read_hdf(files['screen_path'], key='params')

    # run the first stage of preprocessing
    filtered_traces, corners, frame_bounds = preprocess_selector(files['avi_path'], files)

    # compute the eye metrics
    filtered_traces = fm.match_eye(filtered_traces)

    # get the wheel info
    filtered_traces = fm.match_wheel(files, filtered_traces)

    # get the motive tracking data
    motive_traces, reference_coordinates, obstacle_coordinates = \
        s1.extract_motive(files['track_path'], files['rig'])

    # align them temporally based on the sync file
    kinematics_data = fm.match_motive_2(motive_traces, files['sync_path'], filtered_traces)

    # For these trials, save the trial set and the trial parameters to the output file
    trials.to_hdf(save_path, key='trial_set', mode='a')
    params.to_hdf(save_path, key='params', mode='a')

    # calculate only if calcium is present
    if files['imaging'] == 'doric':
        # get a dataframe with the calcium data matched to the bonsai data
        matched_calcium, roi_info = fm.match_calcium_2(calcium_path, files['sync_path'], kinematics_data, trials=trials)
        # if there is a calcium output, write to the file
        if matched_calcium is not None:
            matched_calcium.to_hdf(save_path, key='matched_calcium', mode='a', format='fixed')
            roi_info.to_hdf(save_path, key='roi_info', mode='a', format='fixed')
            # also get the cell matching if it exists
            cell_matches = fm.match_cells(match_path)
            cell_matches.to_hdf(save_path, key='cell_matches', mode='a', format='fixed')

elif files['rig'] in ['VTuningWF']:
    # load the data for the trial structure and parameters
    trials = read_hdf(files['screen_path'], key='trial_set')
    params = read_hdf(files['screen_path'], key='params')

    # get the video tracking data
    filtered_traces, px_corners, frame_bounds = preprocess_selector(files['avi_path'], files)

    # define the dimensions of the arena
    manual_coordinates = paths.arena_coordinates['VTuningWF']

    # get the motive tracking data
    motive_traces, reference_coordinates, obstacle_coordinates = \
        s1.extract_motive(files['track_path'], files['rig'])

    # scale the traces accordingly
    filtered_traces, corners = \
        fp.rescale_pixels(filtered_traces, files, manual_coordinates, px_corners.to_numpy().T)

    # align them temporally based on the sync file
    filtered_traces = fm.match_motive_2(motive_traces, files['sync_path'], filtered_traces)

    # run the preprocessing kinematic calculations
    # also saves the data
    kinematics_data, real_crickets, vr_crickets = s2.kinematic_calculations(filtered_traces)

    # For these trials, save the trial set and the trial parameters to the output file
    trials.to_hdf(save_path, key='trial_set', mode='a')
    params.to_hdf(save_path, key='params', mode='a')

    # calculate only if calcium is present
    if files['imaging'] == 'wirefree':
        # get a dataframe with the calcium data matched to the bonsai data
        matched_calcium, roi_info = fm.match_calcium_2(calcium_path, files['sync_path'], kinematics_data, trials=trials)
        # if there is a calcium output, write to the file
        if matched_calcium is not None:
            matched_calcium.to_hdf(save_path, key='matched_calcium', mode='a', format='fixed')
            roi_info.to_hdf(save_path, key='roi_info', mode='a', format='fixed')
            # also get the cell matching if it exists
            cell_matches = fm.match_cells(match_path)
            cell_matches.to_hdf(save_path, key='cell_matches', mode='a', format='fixed')

elif files['rig'] in ['VWheelWF']:
    # load the data for the trial structure and parameters
    trials = read_hdf(files['screen_path'], key='trial_set')
    params = read_hdf(files['screen_path'], key='params')

    # run the first stage of preprocessing
    filtered_traces, corners, frame_bounds = preprocess_selector(files['avi_path'], files)

    # compute the eye metrics
    filtered_traces = fm.match_eye(filtered_traces)

    # get the wheel info
    filtered_traces = fm.match_wheel(files, filtered_traces)

    # get the motive tracking data
    motive_traces, reference_coordinates, obstacle_coordinates = \
        s1.extract_motive(files['track_path'], files['rig'])

    # align them temporally based on the sync file
    kinematics_data = fm.match_motive_2(motive_traces, files['sync_path'], filtered_traces)

    # For these trials, save the trial set and the trial parameters to the output file
    trials.to_hdf(save_path, key='trial_set', mode='a')
    params.to_hdf(save_path, key='params', mode='a')

    # calculate only if calcium is present
    if files['imaging'] == 'wirefree':
        # get a dataframe with the calcium data matched to the bonsai data
        matched_calcium, roi_info = fm.match_calcium_2(calcium_path, files['sync_path'], kinematics_data, trials=trials)
        # if there is a calcium output, write to the file
        if matched_calcium is not None:
            matched_calcium.to_hdf(save_path, key='matched_calcium', mode='a', format='fixed')
            roi_info.to_hdf(save_path, key='roi_info', mode='a', format='fixed')
            # also get the cell matching if it exists
            cell_matches = fm.match_cells(match_path)
            cell_matches.to_hdf(save_path, key='cell_matches', mode='a', format='fixed')

else:
    # return all empty outputs and print a warning
    # TODO: replace with logging
    filtered_traces = fm.empty_dataframe()
    kinematics_data = fm.empty_dataframe()
    corners = []
    print(f'File {files["slug"]} has an invalid rig type')

# save the kinematics and arena corner data
kinematics_data.to_hdf(save_path, key='full_traces', mode='a', format='fixed')
corners_df = DataFrame(data=corners, columns=['x', 'y'])
corners_df.to_hdf(save_path, key='arena_corners', mode='a')

# # For these trials, save the trial set and the trial parameters to the output file
# if files['rig'] in ['VTuning', 'VWheel']:
#     trials.to_hdf(save_path, key='trial_set', mode='a', format='table')
#     params.to_hdf(save_path, key='params', mode='a', format='table')

# generate the output figure
fig_final = fplot.preprocessing_figure(filtered_traces, real_crickets, vr_crickets, corners)

# define the path for the figure
fig_final.savefig(pic_path, bbox_inches='tight')

# assemble the entry data
entry_data = {
    'analysis_type': 'preprocessing',
    'analysis_path': save_path,
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
