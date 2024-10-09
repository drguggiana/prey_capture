import yaml
import os
import h5py
import numpy as np
import datetime
import cv2

import paths
import functions_misc as fm
import processing_parameters
import functions_bondjango as bd


def get_footprint_contours(calcium_data):
    contour_list = []
    contour_stats = []
    for frame in calcium_data:
        frame = frame * 255.
        frame = frame.astype(np.uint8)
        thresh = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        # get contours and filter out small defects
        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        # Only take the largest contour
        cntr = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cntr)
        perimeter = cv2.arcLength(cntr, True)
        compactness = 4 * np.pi * area / (perimeter + 1e-16) ** 2

        contour_list.append(cntr)
        contour_stats.append((area, perimeter, compactness))

    return contour_list, np.array(contour_stats)


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
    data_all = bd.query_database('vr_experiment', search_string)
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
    calcium_path = os.path.join(paths.analysis_path, '_'.join((day, animal, rig, 'calciumraw.hdf5')))

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

        raw_fluor_data = np.array(f['YrA'])  # this is raw fluorescence
        spikes_data = np.array(f['S'])     # these are inferred spikes
        deconv_fluor_data = np.array(f['C'])      # This is deconvolved fluorescence
        footprints = np.array(f['A'])
        processed_frames = np.array(f['processed_frames'])

    # We aren't scattering calcium data, per se. Instead, we want to use this
    # function to pad the trimmed Ca2+ data
    # get the total frame number from the raw calcium movie
    raw_frame_count = [int(el) for el in frame_list[:, 1]][0]

    # get the first and last frame number from the trimmed Ca2+ data
    frame_start = processed_frames[0]
    frame_end = processed_frames[-1]

    # create NaN-padded arrays in order to match raw movie frame count
    prepend_pad = np.empty((spikes_data.shape[0], frame_start))
    postpend_pad = np.empty((spikes_data.shape[0], raw_frame_count - frame_end))
    prepend_pad[:] = np.NaN
    postpend_pad[:] = np.NaN
    current_spikes = np.concatenate((prepend_pad, spikes_data, postpend_pad), axis=1)
    current_deconv_fluor = np.concatenate((prepend_pad, deconv_fluor_data, postpend_pad), axis=1)
    current_raw_fluor = np.concatenate((prepend_pad, raw_fluor_data, postpend_pad), axis=1)

    # clear the rois that don't pass the size criteria
    roi_info = fm.get_roi_stats(footprints)
    contours, contour_stats = get_footprint_contours(footprints)

    if len(roi_info.shape) == 1:
        roi_stats = roi_info.reshape(1, -1)
        contour_stats = contour_stats.reshape(1, -1)

    areas = roi_info[:, -1]
    compactness = contour_stats[:, -1]

    keep_vector = (areas > processing_parameters.roi_parameters['area_min']) & \
                  (areas < processing_parameters.roi_parameters['area_max']) & \
                  (compactness > processing_parameters.roi_parameters['compactness'])

    # remove from the calcium
    current_spikes = current_spikes[keep_vector, :]
    current_deconv_fluor = current_deconv_fluor[keep_vector, :]
    current_raw_fluor = current_raw_fluor[keep_vector, :]
    roi_info = roi_info[keep_vector, :]

    # save the data as an h5py
    with h5py.File(out_path, 'w') as file:
        file.create_dataset('roi_info', data=roi_info)
        file.create_dataset('raw_fluor_data', data=current_raw_fluor)
        file.create_dataset('deconv_fluor_data', data=current_deconv_fluor)
        # Ideally, this would be called spikes, but keep the name to work with the existing code
        file.create_dataset('calcium_data', data=current_spikes)    

except (KeyError, ValueError):
    print('This file did not contain any ROIs: ' + calcium_path)
    
    # create a dummy empty file
    with h5py.File(out_path, 'w') as file:
        file.create_dataset('roi_info', data='no_ROIs')
        file.create_dataset('raw_fluor_data', data='no_ROIs')
        file.create_dataset('calcium_data', data='no_ROIs')
        file.create_dataset('deconv_fluor_data', data='no_ROIs')

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
