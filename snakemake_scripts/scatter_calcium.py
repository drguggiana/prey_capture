import yaml
import os
import h5py
import functions_bondjango as bd
import paths
import numpy as np
import datetime
import processing_parameters
import cv2

try:

    # get the path to the file, parse and turn into a dictionary
    calcium_path = snakemake.input[0]
    video_data = yaml.load(snakemake.params.info, Loader=yaml.FullLoader)
    # get the save paths
    out_path = snakemake.output[0]

    # get the parts to assemble the input file path
    animal = video_data['mouse']
    rig = video_data['rig']
    trial_date = datetime.datetime.strptime(video_data['date'], '%Y-%m-%dT%H:%M:%SZ')
    day = trial_date.strftime('%m_%d_%Y')
    time = trial_date.strftime('%H_%M_%S')

    # print('scatter calcium:'+calcium_path)
    # print('scatter out:'+out_path)
except NameError:

    # define the target file
    search_string = processing_parameters.search_string

    # query the database for data to plot
    data_all = bd.query_database('video_experiment', search_string)
    video_data = data_all[0]

    # get the parts to assemble the input file path
    animal = video_data['mouse']
    rig = video_data['rig']
    trial_date = datetime.datetime.strptime(video_data['date'], '%Y-%m-%dT%H:%M:%SZ')
    day = trial_date.strftime('%m_%d_%Y')
    time = trial_date.strftime('%H_%M_%S')

    # assemble the output path
    out_path = video_data['tif_path'].replace('.tif', '_calcium.hdf5')

    # get the input file path
    calcium_path = os.path.join(paths.analysis_path, '_'.join((day, animal, 'calciumday.hdf5')))

# get the model of origin
if rig == 'miniscope':
    target_model = 'video_experiment'
else:
    target_model = 'vr_experiment'

# if there are no ROIs detected, skip the file and print the name
try:
    # load the contents of the ca file
    with h5py.File(calcium_path, 'r') as f:

        frame_list = np.array(f['frame_list'])
        # if there are no ROIs, raise to generate an empty file
        if frame_list == 'no_ROIs':
            raise ValueError('empty file')
        calcium_data = np.array(f['S'])
        # calcium_data = np.array(f['C'])
        footprints = np.array(f['A'])

    # get the trials in the file
    trials_list = [str(el)[2:-1] for el in frame_list[:, 0]]
    # get the frame numbers
    frame_numbers = [int(el) for el in frame_list[:, 1]]
    # based on the time, find the corresponding calcium data
    trial_idx = np.argwhere(np.array(trials_list) == '_'.join((day, time)))[0][0]
    # get the frame start and end
    frame_start = int(np.sum(frame_numbers[:trial_idx]))
    frame_end = int(frame_start + frame_numbers[trial_idx])
    # extract the frames
    current_calcium = calcium_data[:, frame_start:frame_end]

    # allocate memory for the centroids
    roi_info = []
    # get the centroid coordinates of each roi
    for roi in footprints:
        # binarize the image
        bin_roi = (roi > 0).astype(np.int8)
        # define the connectivity
        connectivity = 8
        # Perform the operation
        output = cv2.connectedComponentsWithStats(bin_roi, connectivity, cv2.CV_32S)
        # store the centroid x and y, the l, t, w, h of the bounding box and the area
        roi_info.append(np.hstack((output[3][1, :], output[2][1, :])))

    # concatenate the centroids
    roi_info = np.vstack(roi_info)
    # save the data as an h5py
    with h5py.File(out_path, 'w') as file:
        file.create_dataset('calcium_data', data=current_calcium)
        file.create_dataset('roi_info', data=roi_info)

except (KeyError, ValueError):
    print('This file did not contain any ROIs: ' + calcium_path)
    # create a dummy empty file
    with h5py.File(out_path, 'w') as file:
        file.create_dataset('calcium_data', data='no_ROIs')
        file.create_dataset('centroids', data='no_ROIs')

# update the bondjango entry (need to sort out some fields)
ori_data = video_data.copy()
ori_data['fluo_path'] = out_path
mouse = ori_data['mouse']
ori_data['mouse'] = '/'.join((paths.bondjango_url, 'mouse', mouse, ''))
ori_data['experiment_type'] = '/'.join((paths.bondjango_url, 'experiment_type', 'Free_behavior', ''))

# fix the preprocessing file field too
for idx, el in enumerate(ori_data['preproc_files']):
    ori_data['preproc_files'][idx] = \
        '/'.join((paths.bondjango_url, 'analyzed_data', ori_data['preproc_files'][idx], ''))

update_url = '/'.join((paths.bondjango_url, target_model, ori_data['slug'], ''))
output_entry = bd.update_entry(update_url, ori_data)

print(output_entry.status_code)
